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

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
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
    PersonMention,
    SessionStatus,
    Transcript,
)
from . import extract, vad
from .audio import TranscodeError, transcode_to_wav_bytes
from .llm import LLMError, chat_json
from .stt import STTError, transcribe_wav
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

        if did_work:
            continue
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=settings.pipeline_poll_seconds
            )
        except asyncio.TimeoutError:
            pass


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
    if settings.llm_base_url:
        logger.info("pipeline: LLM enabled (%s)", settings.llm_base_url)
    else:
        logger.warning(
            "pipeline: LLM disabled — transcripts will pile up in pending_llm "
            "until OMILOG_LLM_BASE_URL is set."
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
        codec = sess.codec
        sample_rate_hz = sess.sample_rate_hz

    if not audio_path_str:
        _mark_failed(session_id, "no audio_path on parent")
        return
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        _mark_failed(session_id, f"parent audio missing: {audio_path}")
        return

    try:
        duration_s, silences = await vad.analyse(
            audio_path,
            threshold_db=settings.vad_threshold_db,
            min_silence_s=settings.vad_min_silence_seconds,
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
        "pipeline: VAD %s → %d conversation(s) over %.1fs (silences=%d)",
        session_id,
        len(convs),
        duration_s,
        len(silences),
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
        )
    except STTError as e:
        logger.error("pipeline: STT failed %s err=%s", session_id, e)
        _mark_failed(session_id, f"stt: {e}")
        return

    with Session(engine) as db:
        db.add(
            Transcript(
                audio_session_id=session_id,
                text=result.text,
                segments_json=json.dumps(result.segments) if result.segments else None,
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
    )

    try:
        chat = await chat_json(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.llm_timeout_s,
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

    _save_extraction(
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


def _save_extraction(
    *,
    session_id: UUID,
    user_id: str,
    started_at: datetime,
    ended_at: datetime | None,
    extraction: extract.Extraction,
) -> None:
    with Session(engine) as db:
        conv = Conversation(
            audio_session_id=session_id,
            user_id=user_id,
            title=extraction.title,
            summary=extraction.summary,
            topics_json=json.dumps(extraction.topics) if extraction.topics else None,
            started_at=started_at,
            ended_at=ended_at or started_at,
        )
        db.add(conv)
        db.flush()  # populate conv.id before we reference it

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
