"""Background pipeline runner.

Single asyncio task started by the app lifespan. Two stages, polled in
priority order:

    recording → pending_stt --(ffmpeg + whisper)--> pending_llm
                                                     |
                pending_llm --(llama-server + parse)--> done

Idles when both stages are starved or their backends aren't configured.

Concurrency: one in-flight call per stage. whisper-server and llama-server are
both single-flight per process; for a single Omi user there's no benefit to
parallelising. If we ever add multiple devices, wrap process_* in semaphores.
"""

import array
import asyncio
import json
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from sqlmodel import Session, select

from ..config import settings
from ..db import engine
from ..models import (
    ActionItem,
    AudioSession,
    CalendarEvent,
    Conversation,
    Decision,
    PersonMention,
    SessionStatus,
    Speaker,
    Transcript,
    WakeAction,
    WakeInvocation,
)
from . import diarize as diarize_mod
from . import extract, vad
from . import wake as wake_mod
from .audio import TranscodeError, transcode_to_wav_bytes
from .diarize import DiarizationError
from .llm import LLMError, chat_json
from .stt import STTError, collapse_repeated_segments, transcribe_wav
from .vad import VADError

logger = logging.getLogger("omilog.pipeline.runner")


# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────

async def run_forever(stop_event: asyncio.Event) -> None:
    _log_startup()

    while not stop_event.is_set():
        try:
            did_work = await _tick()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("pipeline: tick crashed")
            did_work = False

        # Periodic audio rotation. Cheap, hourly, only does anything when
        # the retention setting is positive — guarded so the default zero-
        # config deployment never deletes anything unexpected.
        try:
            _maybe_rotate_audio()
        except Exception:
            logger.exception("pipeline: audio rotation crashed")

        if did_work:
            continue
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=settings.pipeline_poll_seconds
            )
        except asyncio.TimeoutError:
            pass


# Track when we last ran rotation so we don't redo it on every pipeline
# tick. None on startup → fires on the first tick (catches data that aged
# while the server was down).
_LAST_ROTATION_AT: float | None = None
_ROTATION_INTERVAL_S: float = 3600.0  # once per hour


def _maybe_rotate_audio() -> None:
    """Run audio rotation if it's been long enough since the last sweep.

    Called from the main loop. Cheap when there's nothing to do (one SELECT
    that returns zero rows). When the retention setting is 0 we skip the
    query entirely.
    """
    global _LAST_ROTATION_AT
    if settings.audio_retention_days <= 0:
        return
    now = monotonic()
    if _LAST_ROTATION_AT is not None and now - _LAST_ROTATION_AT < _ROTATION_INTERVAL_S:
        return
    _LAST_ROTATION_AT = now
    deleted = _rotate_old_audio()
    if deleted > 0:
        logger.info(
            "pipeline: audio rotation deleted %d file(s) older than %d days",
            deleted,
            settings.audio_retention_days,
        )


def _rotate_old_audio() -> int:
    """Delete .opus files for done-or-segmented sessions older than
    ``audio_retention_days``, with two exemptions:

    - ``archived=True`` sessions are always kept (user explicitly pinned).
    - We never touch in-flight sessions (recording / pending_*) — those
      need their audio for processing. Status filter handles this.

    The DB row and any associated Transcript / Conversation stay intact;
    only the file goes and ``audio_path`` gets cleared so the UI knows to
    hide the audio player. Path-traversal guard rejects any audio_path
    outside ``storage_dir``.

    Returns the count of files successfully deleted.
    """
    if settings.audio_retention_days <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.audio_retention_days)
    storage_root = settings.storage_dir.resolve()
    deleted = 0
    with Session(engine) as db:
        rows = list(
            db.exec(
                select(AudioSession)
                .where(AudioSession.archived == False)  # noqa: E712 — SQLA
                .where(AudioSession.audio_path != None)  # noqa: E711 — SQLA
                .where(AudioSession.started_at < cutoff)
                # Only sessions that finished processing — never yank audio
                # out from under an active STT/diarize/LLM call.
                .where(
                    AudioSession.status.in_(
                        [SessionStatus.done, SessionStatus.segmented]
                    )
                )
            ).all()
        )
        for row in rows:
            try:
                p = Path(row.audio_path).resolve()
                p.relative_to(storage_root)  # path-traversal guard
                p.unlink(missing_ok=True)
            except (ValueError, OSError) as e:
                # Couldn't delete (permission, gone, weird path) — leave the
                # row's audio_path alone so the next sweep retries it.
                logger.warning(
                    "rotation: failed to unlink %s: %s", row.audio_path, e
                )
                continue
            row.audio_path = None
            db.add(row)
            deleted += 1
        db.commit()
    return deleted


def _log_startup() -> None:
    if settings.vad_enabled:
        logger.info(
            "pipeline: VAD enabled (gap=%.0fs, threshold=%.0fdB)",
            settings.vad_gap_seconds,
            settings.vad_threshold_db,
        )
    else:
        logger.warning(
            "pipeline: VAD disabled — parent sessions in pending_vad will not "
            "be segmented (set OMILOG_VAD_ENABLED=true to enable)."
        )
    if settings.stt_base_url:
        logger.info("pipeline: STT enabled (%s)", settings.stt_base_url)
    else:
        logger.warning(
            "pipeline: STT disabled — sessions will pile up in pending_stt "
            "until OMILOG_STT_BASE_URL is set."
        )
    if settings.diarization_enabled:
        if diarize_mod.DIARIZATION_AVAILABLE:
            logger.info(
                "pipeline: diarization enabled (sherpa-onnx, models=%s, %s)",
                settings.diarization_segmentation_model.name,
                settings.diarization_embedding_model.name,
            )
        else:
            err = diarize_mod.DIARIZATION_IMPORT_ERROR or "no details captured"
            hint = ""
            err_lower = err.lower()
            if "libsndfile" in err_lower or "sndfile" in err_lower:
                hint = (
                    " Likely cause: libsndfile1 is missing system-wide. Fix: "
                    "`sudo apt install libsndfile1` (Debian/Ubuntu/Raspberry Pi)."
                )
            elif "libonnxruntime" in err_lower or "onnxruntime" in err_lower:
                hint = (
                    " Likely cause: sherpa-onnx's bundled onnxruntime didn't land. "
                    "Fix: `uv sync --extra diarization` after a `git pull` "
                    "(the extra now installs onnxruntime explicitly)."
                )
            elif "sherpa_onnx" in err_lower or "sherpa-onnx" in err_lower:
                hint = (
                    " Likely cause: sherpa-onnx not installed in the active venv. "
                    "Fix: `uv sync --extra diarization`."
                )
            logger.warning(
                "pipeline: diarization enabled but the deps failed to import: %s.%s",
                err,
                hint,
            )
    if settings.llm_base_url:
        logger.info("pipeline: LLM enabled (%s)", settings.llm_base_url)
    else:
        logger.warning(
            "pipeline: LLM disabled — transcripts will pile up in pending_llm "
            "until OMILOG_LLM_BASE_URL is set."
        )
    if settings.audio_retention_days > 0:
        logger.info(
            "pipeline: audio rotation enabled (delete after %d day(s); "
            "archived 📌 sessions exempt)",
            settings.audio_retention_days,
        )


async def _tick() -> bool:
    """Process at most one session. Priority: VAD > STT > LLM."""
    if settings.vad_enabled:
        sid = _claim_next(SessionStatus.pending_vad)
        if sid:
            await _safe_process(sid, process_vad)
            return True
    if settings.stt_base_url:
        sid = _claim_next(SessionStatus.pending_stt)
        if sid:
            await _safe_process(sid, process_stt)
            return True
    if settings.llm_base_url:
        sid = _claim_next(SessionStatus.pending_llm)
        if sid:
            await _safe_process(sid, process_llm)
            return True
    return False


async def _safe_process(sid: UUID, fn) -> None:
    try:
        await fn(sid)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception("pipeline: unhandled error on %s", sid)
        _mark_failed(sid, f"runner: {type(e).__name__}: {e}")


def _claim_next(status: SessionStatus) -> UUID | None:
    with Session(engine) as db:
        stmt = (
            select(AudioSession)
            .where(AudioSession.status == status)
            .order_by(AudioSession.started_at)
            .limit(1)
        )
        row = db.exec(stmt).first()
        return row.id if row else None


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: STT (pending_stt → pending_llm)
# ──────────────────────────────────────────────────────────────────────────────

async def process_vad(session_id: UUID) -> None:
    """Carve a parent capture into N child sessions, one per conversation.

    Status transitions:
      parent (pending_vad)
          → all silence detected:     parent.status=silent, file deleted
          → no silence at all:        treat the whole thing as 1 conversation
          → 1+ conversation regions:  spawn N children (pending_stt),
                                       parent.status=segmented, file deleted
    """
    logger.info("pipeline: VAD processing %s", session_id)
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None:
            return
        audio_path_str = sess.audio_path
        user_id = sess.user_id
        parent_started_at = sess.started_at

    if not audio_path_str:
        _mark_failed(session_id, "no audio_path on parent")
        return
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        _mark_failed(session_id, f"parent audio missing: {audio_path}")
        return

    try:
        duration_s, silences, backend_used = await vad.analyse_with_backend(
            audio_path,
            backend=settings.vad_backend,
            threshold_db=settings.vad_threshold_db,
            min_silence_s=settings.vad_min_silence_seconds,
            silero_model_path=settings.vad_silero_model,
            silero_threshold=settings.vad_silero_threshold,
            silero_min_speech_s=settings.vad_silero_min_speech_seconds,
        )
    except VADError as e:
        logger.error("pipeline: VAD analyse failed %s err=%s", session_id, e)
        _mark_failed(session_id, f"vad: {e}")
        return

    convs = vad.segment_by_silence_gaps(
        duration_s,
        silences,
        gap_threshold_s=settings.vad_gap_seconds,
        pad_s=settings.vad_pad_seconds,
    )

    if not convs:
        logger.info("pipeline: VAD %s all silence, dropping", session_id)
        _mark_silent_and_delete(session_id, audio_path)
        return

    logger.info(
        "pipeline: VAD %s → %d conversation(s) over %.1fs (silences=%d, backend=%s)",
        session_id,
        len(convs),
        duration_s,
        len(silences),
        backend_used,
    )

    children: list[UUID] = []
    try:
        for idx, (start_s, end_s) in enumerate(convs):
            child_id = uuid4()
            child_path = settings.storage_dir / f"{child_id}.opus"
            await vad.extract_segment_to_opus(
                audio_path,
                child_path,
                start_s=start_s,
                end_s=end_s,
                bitrate=settings.vad_child_bitrate,
            )
            child_started_at = parent_started_at + timedelta(seconds=start_s)
            child_ended_at = parent_started_at + timedelta(seconds=end_s)
            with Session(engine) as db:
                db.add(
                    AudioSession(
                        id=child_id,
                        user_id=user_id,
                        parent_id=session_id,
                        codec="opus",
                        sample_rate_hz=16000,
                        audio_path=str(child_path),
                        bytes_written=child_path.stat().st_size,
                        started_at=child_started_at,
                        ended_at=child_ended_at,
                        duration_s=end_s - start_s,
                        status=SessionStatus.pending_stt,
                    )
                )
                db.commit()
            children.append(child_id)
            logger.info(
                "pipeline: VAD %s → child[%d] %s [%.1fs..%.1fs] %d bytes",
                session_id,
                idx,
                child_id,
                start_s,
                end_s,
                child_path.stat().st_size,
            )
    except VADError as e:
        logger.error("pipeline: VAD extract failed %s err=%s", session_id, e)
        _mark_failed(session_id, f"vad-extract: {e}")
        return

    # All children extracted: mark parent segmented and free the disk.
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is not None:
            sess.status = SessionStatus.segmented
            sess.error_msg = None
            db.add(sess)
            db.commit()
    _try_delete(audio_path)
    logger.info("pipeline: VAD %s done, spawned %d children", session_id, len(children))


def _mark_silent_and_delete(session_id: UUID, audio_path: Path) -> None:
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is not None:
            sess.status = SessionStatus.silent
            sess.error_msg = None
            db.add(sess)
            db.commit()
    _try_delete(audio_path)


def _try_delete(audio_path: Path) -> None:
    try:
        audio_path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("pipeline: could not delete %s: %s", audio_path, e)


async def process_stt(session_id: UUID) -> None:
    logger.info("pipeline: STT processing %s", session_id)
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None:
            logger.warning("pipeline: session %s vanished", session_id)
            return
        audio_path_str = sess.audio_path
        user_id = sess.user_id

    if not audio_path_str:
        _mark_failed(session_id, "no audio_path on session")
        return
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        _mark_failed(session_id, f"audio file missing: {audio_path}")
        return

    try:
        wav_bytes = await transcode_to_wav_bytes(audio_path)
    except TranscodeError as e:
        logger.error("pipeline: transcode failed %s err=%s", session_id, e)
        _mark_failed(session_id, f"ffmpeg: {e}")
        return

    try:
        result = await transcribe_wav(
            wav_bytes,
            base_url=settings.stt_base_url,
            inference_path=settings.stt_inference_path,
            language=settings.stt_language,
            timeout_s=settings.stt_timeout_s,
            initial_prompt=settings.stt_initial_prompt,
            temperature=settings.stt_temperature,
        )
    except STTError as e:
        logger.error("pipeline: STT failed %s err=%s", session_id, e)
        _mark_failed(session_id, f"stt: {e}")
        return

    # Whisper "repeats the previous output" loops are common on noisy /
    # silent audio segments. Collapse runs of identical text before doing
    # anything else with the segments, so they don't pollute the transcript
    # storage, the diarization input, or the LLM prompt.
    segments = collapse_repeated_segments(list(result.segments or []))
    if settings.diarization_enabled and segments:
        segments = await _diarize_or_continue(
            session_id, wav_bytes, segments, user_id
        )

    with Session(engine) as db:
        db.add(
            Transcript(
                audio_session_id=session_id,
                text=result.text,
                segments_json=json.dumps(segments) if segments else None,
                language=result.language,
                model=settings.stt_model_name,
            )
        )
        sess = db.get(AudioSession, session_id)
        if sess is not None:
            sess.status = SessionStatus.pending_llm
            sess.error_msg = None
            db.add(sess)
        db.commit()

    logger.info(
        "pipeline: STT done %s (%d chars, lang=%s) → pending_llm",
        session_id,
        len(result.text),
        result.language,
    )


async def _diarize_or_continue(
    session_id: UUID,
    wav_bytes: bytes,
    segments: list[dict],
    user_id: str,
) -> list[dict]:
    """Run sherpa-onnx diarization directly on the in-memory WAV bytes,
    then cross-conversation-link the detected speakers to known Speaker
    rows. Any failure is logged-and-swallowed — diarization (and linking)
    is a quality enhancement, never a pipeline blocker."""
    try:
        turns = await diarize_mod.diarize(
            wav_bytes,
            seg_path=settings.diarization_segmentation_model,
            emb_path=settings.diarization_embedding_model,
            min_speech_s=settings.diarization_min_speech_seconds,
            min_silence_s=settings.diarization_min_silence_seconds,
            num_threads=settings.diarization_num_threads,
            num_clusters=settings.diarization_num_clusters,
            cluster_threshold=settings.diarization_cluster_threshold,
        )
        # Post-merge: in-Python second pass that folds clusters whose
        # embeddings cosine-agree above the threshold. The opt-in fix for
        # over-split conversations (e.g. 2 actual people landing as 9
        # sherpa-onnx clusters). Disabled when threshold >= 1.0.
        if settings.diarization_post_merge_threshold < 1.0:
            turns = await diarize_mod.post_merge_clusters(
                wav_bytes,
                turns,
                emb_path=settings.diarization_embedding_model,
                threshold=settings.diarization_post_merge_threshold,
                num_threads=settings.diarization_num_threads,
            )
        segments = diarize_mod.assign_speakers_to_segments(segments, turns)
        segments = diarize_mod.relabel_user_and_others(segments)
        logger.info(
            "pipeline: diarize %s found %d turns, %d speakers",
            session_id,
            len(turns),
            len({t["speaker"] for t in turns}),
        )
    except DiarizationError as e:
        logger.warning(
            "pipeline: diarize %s failed (%s) — continuing without speaker labels",
            session_id,
            e,
        )
        return segments
    except Exception:
        logger.exception(
            "pipeline: diarize %s raised — continuing without speaker labels",
            session_id,
        )
        return segments

    # Phase 5: link the detected clusters to known voices across conversations.
    # Embeddings are computed in a thread (sherpa-onnx releases the GIL but the
    # numpy slicing doesn't); matching itself is fast.
    try:
        loop = asyncio.get_event_loop()
        embeddings_by_label = await loop.run_in_executor(
            None,
            lambda: diarize_mod.compute_speaker_embeddings(
                wav_bytes,
                segments,
                emb_path=settings.diarization_embedding_model,
                num_threads=settings.diarization_num_threads,
            ),
        )
        if embeddings_by_label:
            segments = _link_speakers_to_segments(
                user_id=user_id,
                segments=segments,
                embeddings_by_label=embeddings_by_label,
                audio_session_id=session_id,
            )
            logger.info(
                "pipeline: speaker linking %s linked %d cluster(s)",
                session_id,
                len(embeddings_by_label),
            )
            # Post-link USER promotion: the per-conversation 'loudest =
            # USER' heuristic mis-picks when you're not the most-talkative
            # speaker in a meeting. If any cluster in this conversation
            # links to a known is_user=True Speaker, swap that label into
            # the USER slot — overrides the heuristic with cross-conv truth.
            segments = _promote_known_user_label(segments, user_id)
    except Exception:
        logger.exception(
            "pipeline: speaker linking %s raised — segments keep their "
            "per-conversation labels but lose cross-conversation linkage",
            session_id,
        )
    return segments


def _emb_to_bytes(emb: list[float]) -> bytes:
    """Serialize a 1-D float list to bytes for SQLite BLOB storage.

    Uses ``array.array("f", …)`` rather than numpy so the matching code in
    this module stays importable on hosts without the diarization extra
    installed (CI, mobile-style deploys, etc.). 4 bytes per float — a
    192-D TitaNet embedding is 768 bytes.
    """
    return array.array("f", emb).tobytes()


def _emb_from_bytes(b: bytes) -> list[float]:
    arr = array.array("f")
    arr.frombytes(b)
    return list(arr)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. 192-D × ~100 speakers is ≪ 1 ms — no
    point pulling in numpy just for this."""
    if len(a) != len(b):
        return 0.0
    dot = 0.0
    na2 = 0.0
    nb2 = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na2 += x * x
        nb2 += y * y
    if na2 == 0.0 or nb2 == 0.0:
        return 0.0
    return dot / (math.sqrt(na2) * math.sqrt(nb2))


def _running_average(
    stored: list[float], new: list[float], n: int
) -> list[float]:
    """Update the running centroid with one more observation: avg = (avg·N + x) / (N+1)."""
    return [(s * n + x) / (n + 1) for s, x in zip(stored, new)]


# Cap preview snippets so the audio player on /speakers doesn't try to
# stream a multi-minute slice. 15 s is enough to recognise a voice;
# longer is wasteful + jarring.
_PREVIEW_MAX_SECONDS = 15.0


def _link_speakers_to_segments(
    *,
    user_id: str,
    segments: list[dict],
    embeddings_by_label: dict[str, list[float]],
    audio_session_id: UUID | None = None,
) -> list[dict]:
    """Match each (label → embedding) against the user's known speakers.

    For each label:
    - Find the existing Speaker whose stored embedding has the highest cosine
      similarity, above ``speaker_match_threshold``. If found, update its
      running-averaged embedding with the new one and bump ``mention_count``.
    - Otherwise create a new Speaker row with this embedding as the initial
      centroid (``mention_count=1``).
    - In either case, annotate every segment for this label with ``speaker_id``
      so the UI can resolve display name from the Speaker table later.

    A label of ``USER`` flips ``is_user=True`` on the matched/created speaker
    (sticky — once a voice has been the wearer in any conversation, we keep
    that mark even if a future conversation has someone else as longest-talker).

    When ``audio_session_id`` is supplied, also picks the longest segment per
    speaker label and updates the Speaker's preview pointer (clipped to
    ``_PREVIEW_MAX_SECONDS``) if it's longer than the currently-stored
    preview. UI uses this for the per-speaker audio snippet on /speakers
    so the user can hear who they're about to name / merge.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    threshold = settings.speaker_match_threshold
    label_to_id: dict[str, UUID] = {}

    # Compute longest segment per label up front — used after the linking
    # pass to update each Speaker.preview_* if the new clip beats the
    # current stored one.
    longest_per_label: dict[str, tuple[float, float]] = {}
    for seg in segments:
        label = seg.get("speaker")
        if not label:
            continue
        start = float(seg.get("start", 0) or 0)
        end = float(seg.get("end", start) or start)
        if end <= start:
            continue
        cur = longest_per_label.get(label)
        if cur is None or (end - start) > (cur[1] - cur[0]):
            longest_per_label[label] = (start, end)

    with Session(engine) as db:
        existing = list(
            db.exec(select(Speaker).where(Speaker.user_id == user_id)).all()
        )
        # Decode embeddings once so we don't pay the bytes→list cost twice
        # per (label, speaker) pair.
        existing_emb: dict[UUID, list[float]] = {
            sp.id: _emb_from_bytes(sp.embedding) for sp in existing
        }
        # Map id → Speaker so the preview-update pass below can locate a
        # row whether it was matched (already in existing) or newly created.
        by_id: dict[UUID, Speaker] = {sp.id: sp for sp in existing}

        for label, new_emb in embeddings_by_label.items():
            if not any(x != 0.0 for x in new_emb):
                continue  # degenerate, can't match

            best: Speaker | None = None
            best_score = threshold
            for sp in existing:
                score = _cosine_similarity(new_emb, existing_emb[sp.id])
                if score > best_score:
                    best = sp
                    best_score = score

            if best is not None:
                n = best.mention_count
                avg = _running_average(existing_emb[best.id], new_emb, n)
                best.embedding = _emb_to_bytes(avg)
                best.mention_count = n + 1
                if label == "USER":
                    best.is_user = True
                best.updated_at = now
                existing_emb[best.id] = avg  # so subsequent labels see fresh centroid
                db.add(best)
                label_to_id[label] = best.id
            else:
                new_sp = Speaker(
                    user_id=user_id,
                    embedding=_emb_to_bytes(new_emb),
                    is_user=(label == "USER"),
                    mention_count=1,
                    created_at=now,
                    updated_at=now,
                )
                db.add(new_sp)
                db.flush()  # populate id
                label_to_id[label] = new_sp.id
                existing.append(new_sp)
                existing_emb[new_sp.id] = list(new_emb)
                by_id[new_sp.id] = new_sp

        # Preview-snippet update: only when we know which audio session
        # the segments came from. For each speaker we just linked, see if
        # the longest segment of theirs in this conversation beats the
        # currently-stored preview, and if so, replace it. Clip length is
        # capped so we don't stream a multi-minute slice when the user
        # clicks play on /speakers.
        if audio_session_id is not None:
            for label, sp_id in label_to_id.items():
                best_span = longest_per_label.get(label)
                if best_span is None:
                    continue
                seg_start, seg_end = best_span
                preview_end = min(seg_end, seg_start + _PREVIEW_MAX_SECONDS)
                new_dur = preview_end - seg_start
                if new_dur <= 0:
                    continue
                sp = by_id.get(sp_id)
                if sp is None:
                    continue
                cur_dur = 0.0
                if (
                    sp.preview_start_s is not None
                    and sp.preview_end_s is not None
                ):
                    cur_dur = sp.preview_end_s - sp.preview_start_s
                if new_dur > cur_dur:
                    sp.preview_audio_session_id = audio_session_id
                    sp.preview_start_s = seg_start
                    sp.preview_end_s = preview_end
                    db.add(sp)

        db.commit()

    # Annotate the segments. Done outside the DB session — pure mutation.
    for seg in segments:
        sp_label = seg.get("speaker")
        if sp_label and sp_label in label_to_id:
            seg["speaker_id"] = str(label_to_id[sp_label])

    return segments


def _promote_known_user_label(
    segments: list[dict], user_id: str
) -> list[dict]:
    """Swap labels so the cluster matching a known is_user=True Speaker
    ends up labeled "USER", regardless of what the talk-time heuristic
    decided.

    The cross-conversation linker already sets ``speaker_id`` on each
    segment. This pass:

    1. Collects distinct speaker_ids referenced by the segments.
    2. Finds any of them with is_user=True in the DB (the wearer's voice
       as previously identified — either auto-flagged in an earlier
       conversation where they WERE the longest talker, or manually
       flagged via the /speakers UI).
    3. If the matched speaker's current label isn't already "USER", swaps
       the two labels: old USER (whoever the heuristic mis-picked) gets
       the matched speaker's label, and the matched speaker gets "USER".

    No-ops in three cases:
    - No segments have speaker_id (linking didn't run or pre-Phase-5 data).
    - No matched speaker_id has is_user=True (first-ever conversation; the
      heuristic gets to pick).
    - The is_user speaker is already labeled "USER" (heuristic agreed).

    Pure swap means S-labels stay contiguous (S1..SN); only USER moves.
    """
    sp_id_strs = {
        seg["speaker_id"] for seg in segments if seg.get("speaker_id")
    }
    if not sp_id_strs:
        return segments

    # Resolve the strings to UUIDs for the IN clause. Drop unparseable
    # ones quietly — same defensive posture as the rest of the linker.
    sp_uuids: list[UUID] = []
    for s in sp_id_strs:
        try:
            sp_uuids.append(UUID(s))
        except (ValueError, TypeError):
            continue
    if not sp_uuids:
        return segments

    with Session(engine) as db:
        known_user_rows = list(
            db.exec(
                select(Speaker)
                .where(Speaker.user_id == user_id)
                .where(Speaker.is_user.is_(True))  # noqa: E712
                .where(Speaker.id.in_(sp_uuids))
            ).all()
        )
    if not known_user_rows:
        return segments

    # If somehow multiple is_user speakers appear in one conversation
    # (shouldn't normally happen — one wearer per device), pick the one
    # whose cluster talked the most so the swap goes to a substantive
    # row, not a stray short utterance.
    user_speaker_ids = {str(r.id) for r in known_user_rows}
    talk_by_sp: dict[str, float] = {}
    for seg in segments:
        sp_id = seg.get("speaker_id")
        if sp_id not in user_speaker_ids:
            continue
        start = float(seg.get("start", 0) or 0)
        end = float(seg.get("end", start) or start)
        if end > start:
            talk_by_sp[sp_id] = talk_by_sp.get(sp_id, 0.0) + (end - start)
    if not talk_by_sp:
        return segments
    chosen_sp_id = max(talk_by_sp, key=lambda k: talk_by_sp[k])

    # Find that speaker's current label (whatever the heuristic gave it).
    current_user_label: str | None = None
    for seg in segments:
        if seg.get("speaker_id") == chosen_sp_id and seg.get("speaker"):
            current_user_label = seg["speaker"]
            break
    if not current_user_label or current_user_label == "USER":
        return segments  # already correct, no swap needed

    # Swap the two labels everywhere. Don't touch other labels (S1..SN
    # stay contiguous; we're just exchanging USER and current_user_label).
    swap_from, swap_to = current_user_label, "USER"
    for seg in segments:
        sp = seg.get("speaker")
        if sp == swap_to:
            seg["speaker"] = swap_from
        elif sp == swap_from:
            seg["speaker"] = swap_to
    logger.info(
        "pipeline: USER label swapped from heuristic pick to "
        "cross-conv is_user speaker (was=%s, now=USER)",
        swap_from,
    )
    return segments


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: LLM extraction (pending_llm → done)
# ──────────────────────────────────────────────────────────────────────────────

async def process_llm(session_id: UUID) -> None:
    logger.info("pipeline: LLM processing %s", session_id)
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None:
            return
        transcript = db.exec(
            select(Transcript)
            .where(Transcript.audio_session_id == session_id)
            .order_by(Transcript.created_at.desc())
            .limit(1)
        ).first()
        if transcript is None:
            _mark_failed(session_id, "no transcript for LLM stage")
            return
        # Snapshot the values we need outside the session.
        text = transcript.text
        segments_json = transcript.segments_json
        user_id = sess.user_id
        started_at = sess.started_at
        ended_at = sess.ended_at

    try:
        segments = json.loads(segments_json) if segments_json else []
    except json.JSONDecodeError:
        segments = []

    try:
        tz = ZoneInfo(settings.local_timezone)
    except Exception:
        tz = ZoneInfo("UTC")
    now = datetime.now(tz)

    messages = extract.build_messages(
        transcript_text=text,
        transcript_segments=segments,
        now=now,
        timezone_label=settings.local_timezone,
        primary_language=settings.llm_primary_language,
        system_prompt_override_path=settings.llm_system_prompt_file,
        enabled=_extraction_flags(),
    )

    try:
        chat = await chat_json(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.llm_timeout_s,
            disable_thinking=settings.llm_disable_thinking,
        )
    except LLMError as e:
        logger.error("pipeline: LLM call failed %s err=%s", session_id, e)
        _mark_failed(session_id, f"llm: {e}")
        return

    try:
        extraction = extract.parse(chat.text)
    except ValueError as e:
        logger.error("pipeline: LLM parse failed %s err=%s", session_id, e)
        _mark_failed(session_id, f"llm-parse: {e}")
        return

    conv_id = _save_extraction(
        session_id=session_id,
        user_id=user_id,
        started_at=started_at,
        ended_at=ended_at,
        extraction=extraction,
    )
    logger.info(
        "pipeline: LLM done %s → done (events=%d action_items=%d people=%d)",
        session_id,
        len(extraction.calendar_events),
        len(extraction.action_items),
        len(extraction.people_mentioned),
    )

    # Wake-word actions are additive: they fire alongside the saved
    # conversation, never replacing the LLM extraction. Failures are
    # logged-and-swallowed inside _run_wake_actions.
    if conv_id is not None:
        await _run_wake_actions(
            user_id=user_id,
            conversation_id=conv_id,
            transcript_text=text,
        )


def _extraction_flags() -> dict[str, bool]:
    """Snapshot the per-category extraction toggles from settings.

    Keys match the dict shape that ``extract.build_messages`` /
    ``extract.render_default_system_prompt`` accept. Used both to build the
    prompt (categories with False are omitted from the schema) and to gate
    DB writes in ``_save_extraction`` (so a model that ignores the prompt
    and returns disabled categories anyway doesn't get them persisted).
    """
    return {
        "calendar_events": settings.extract_calendar_events,
        "action_items": settings.extract_action_items,
        "decisions": settings.extract_decisions,
        "people_mentioned": settings.extract_people_mentioned,
        "topics": settings.extract_topics,
    }


def _save_extraction(
    *,
    session_id: UUID,
    user_id: str,
    started_at: datetime,
    ended_at: datetime | None,
    extraction: extract.Extraction,
) -> UUID | None:
    flags = _extraction_flags()
    with Session(engine) as db:
        # Default 0.5 when the LLM didn't return a score (older prompt
        # overrides, parse hiccup, etc.) — that keeps the conversation
        # showing up in the default "normal+" filter rather than getting
        # silently buried because of a missing field.
        score = extraction.quality_score if extraction.quality_score is not None else 0.5
        # If the JSON was repaired (likely truncated), bias slightly down —
        # we don't trust an incomplete extraction's self-rating as much.
        if extraction.was_repaired:
            score = max(0.0, score - 0.1)
        conv = Conversation(
            audio_session_id=session_id,
            user_id=user_id,
            title=extraction.title,
            summary=extraction.summary,
            # Topics are kept as a JSON list on Conversation directly; respect
            # the toggle by null'ing the field when disabled even if the model
            # snuck a topics array into its output anyway.
            topics_json=(
                json.dumps(extraction.topics)
                if extraction.topics and flags["topics"]
                else None
            ),
            extraction_repaired=extraction.was_repaired,
            quality_score=score,
            quality_reasoning=extraction.quality_reasoning,
            started_at=started_at,
            ended_at=ended_at or started_at,
        )
        db.add(conv)
        db.flush()  # populate conv.id before we reference it

        if flags["calendar_events"]:
            for evt in extraction.calendar_events:
                db.add(
                    CalendarEvent(
                        conversation_id=conv.id,
                        title=(evt.get("title") or "")[:200] or "(untitled)",
                        description=evt.get("description"),
                        starts_at=extract.parse_iso8601(evt.get("starts_at")),
                        ends_at=extract.parse_iso8601(evt.get("ends_at")),
                        location=evt.get("location"),
                        attendees_json=json.dumps(evt.get("attendees", []))
                        if evt.get("attendees")
                        else None,
                        confidence=_clamp01(evt.get("confidence")),
                    )
                )
        if flags["action_items"]:
            for ai in extraction.action_items:
                text_ = (ai.get("text") or "").strip()
                if not text_:
                    continue
                db.add(
                    ActionItem(
                        conversation_id=conv.id,
                        text=text_,
                        owner=ai.get("owner"),
                        due_at=extract.parse_iso8601(ai.get("due_at")),
                    )
                )
        if flags["decisions"]:
            for d in extraction.decisions:
                text_ = (d.get("text") or "").strip()
                if not text_:
                    continue
                db.add(
                    Decision(
                        conversation_id=conv.id,
                        text=text_,
                        made_by=d.get("made_by"),
                        confidence=_clamp01(d.get("confidence")),
                    )
                )
        if flags["people_mentioned"]:
            for p in extraction.people_mentioned:
                name = (p.get("name") or "").strip()
                if not name:
                    continue
                db.add(
                    PersonMention(
                        conversation_id=conv.id,
                        name=name,
                        context=p.get("context"),
                    )
                )
        sess = db.get(AudioSession, session_id)
        if sess is not None:
            sess.status = SessionStatus.done
            sess.error_msg = None
            db.add(sess)
        db.commit()
        return conv.id


async def _run_wake_actions(
    *,
    user_id: str,
    conversation_id: UUID,
    transcript_text: str,
) -> None:
    """Match the transcript against the user's enabled WakeActions, fire any
    that hit, persist a WakeInvocation row per fire.

    Iteration is sequential per match. If you want parallel, swap to
    ``asyncio.gather`` — but the typical case is at most one or two matches
    per conversation, and most user-installed commands are fast.
    """
    with Session(engine) as db:
        actions = list(
            db.exec(
                select(WakeAction)
                .where(WakeAction.user_id == user_id)
                .where(WakeAction.enabled == True)  # noqa: E712 — SQLAlchemy
            ).all()
        )
    if not actions:
        return

    for action in actions:
        try:
            phrases = json.loads(action.phrases_json)
        except json.JSONDecodeError:
            logger.warning(
                "wake: action %s has malformed phrases_json, skipping", action.id
            )
            continue
        stop_phrases: list[str] = []
        if action.stop_phrases_json:
            try:
                stop_phrases = json.loads(action.stop_phrases_json) or []
            except json.JSONDecodeError:
                logger.warning(
                    "wake: action %s has malformed stop_phrases_json, ignoring",
                    action.id,
                )
        matches = wake_mod.find_wake_matches(
            transcript_text, phrases, stop_phrases=stop_phrases
        )
        for match in matches:
            variables = {
                "transcript": match["transcript"],
                "transcript_full": transcript_text,
                "conversation_id": str(conversation_id),
                "wake_phrase": match["phrase"],
            }
            try:
                resolved = wake_mod.resolve_command(action.command, variables)
                result = await wake_mod.execute_command(
                    resolved, timeout_s=action.timeout_seconds
                )
            except Exception as e:
                logger.exception("wake: action %s execution crashed", action.id)
                result = {
                    "exit_code": None,
                    "stdout": "",
                    "stderr": f"runner crashed: {e}",
                    "duration_ms": 0,
                }
                resolved = action.command  # store the raw template if resolve failed

            with Session(engine) as db:
                db.add(
                    WakeInvocation(
                        wake_action_id=action.id,
                        conversation_id=conversation_id,
                        matched_phrase=match["phrase"],
                        input_text=match["transcript"],
                        command_resolved=resolved,
                        exit_code=result["exit_code"],
                        stdout=result["stdout"],
                        stderr=result["stderr"],
                        duration_ms=result["duration_ms"],
                    )
                )
                db.commit()
            logger.info(
                "wake: %s '%s' → exit=%s in %dms",
                action.name,
                match["phrase"],
                result["exit_code"],
                result["duration_ms"],
            )


def _clamp01(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, x))


def _mark_failed(session_id: UUID, error_msg: str) -> None:
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None:
            return
        sess.status = SessionStatus.failed
        sess.error_msg = error_msg[:500]
        db.add(sess)
        db.commit()
