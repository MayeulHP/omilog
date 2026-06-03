"""Background pipeline runner.

Single asyncio task started by the app lifespan. Polls SQLite for sessions in
`pending_stt`, processes them one at a time:

    pending_stt --[ffmpeg → whisper.cpp]--> pending_llm (or done if no LLM yet)

When STT_BASE_URL is unset we log once and idle — handy for dev on the Mac
without the GPU box reachable.

Concurrency: deliberately one-at-a-time. whisper.cpp is single-flight per
process, and a single Omi user generates one session at a time. If we ever
add a second device or a backfill, throw a semaphore in process_one().
"""

import asyncio
import json
import logging
from pathlib import Path
from uuid import UUID

from sqlmodel import Session, select

from ..config import settings
from ..db import engine
from ..models import AudioSession, SessionStatus, Transcript
from .audio import TranscodeError, transcode_to_wav_bytes
from .stt import STTError, transcribe_wav

logger = logging.getLogger("omilog.pipeline.runner")


async def run_forever(stop_event: asyncio.Event) -> None:
    """Loop until `stop_event` is set. Designed to be cancelled cleanly."""
    if not settings.stt_base_url:
        logger.warning(
            "pipeline: OMILOG_STT_BASE_URL not set — runner is idle. "
            "Sessions will sit in pending_stt until configured."
        )
    else:
        logger.info(
            "pipeline: runner started, stt=%s lang=%s poll=%.1fs",
            settings.stt_base_url,
            settings.stt_language,
            settings.pipeline_poll_seconds,
        )

    while not stop_event.is_set():
        try:
            sess_id = _claim_next_pending_stt()
        except Exception:
            logger.exception("pipeline: claim error")
            sess_id = None

        if sess_id is None:
            try:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=settings.pipeline_poll_seconds
                )
            except asyncio.TimeoutError:
                pass
            continue

        if not settings.stt_base_url:
            # Don't burn through pending sessions when STT is disabled.
            await asyncio.sleep(settings.pipeline_poll_seconds)
            continue

        try:
            await process_one(sess_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("pipeline: unhandled error processing %s", sess_id)
            _mark_failed(sess_id, "runner: unhandled exception")


def _claim_next_pending_stt() -> UUID | None:
    """Pick the oldest pending session. Doesn't lock — single-runner assumption."""
    with Session(engine) as db:
        stmt = (
            select(AudioSession)
            .where(AudioSession.status == SessionStatus.pending_stt)
            .order_by(AudioSession.started_at)
            .limit(1)
        )
        row = db.exec(stmt).first()
        return row.id if row else None


async def process_one(session_id: UUID) -> None:
    logger.info("pipeline: processing session=%s", session_id)
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
        logger.error("pipeline: transcode failed session=%s err=%s", session_id, e)
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
        logger.error("pipeline: STT failed session=%s err=%s", session_id, e)
        _mark_failed(session_id, f"stt: {e}")
        return

    _save_transcript(
        session_id=session_id,
        text=result.text,
        segments=result.segments,
        language=result.language,
        model=settings.stt_model_name,
    )
    logger.info(
        "pipeline: session=%s transcribed (%d chars, lang=%s)",
        session_id,
        len(result.text),
        result.language,
    )


def _save_transcript(
    *,
    session_id: UUID,
    text: str,
    segments: list,
    language: str | None,
    model: str,
) -> None:
    with Session(engine) as db:
        db.add(
            Transcript(
                audio_session_id=session_id,
                text=text,
                segments_json=json.dumps(segments) if segments else None,
                language=language,
                model=model,
            )
        )
        sess = db.get(AudioSession, session_id)
        if sess is not None:
            # Phase 1 terminal state: STT done, LLM not built yet. Phase 2 will
            # add an LLM worker that consumes `pending_llm` and transitions
            # to `done`.
            sess.status = SessionStatus.pending_llm
            sess.error_msg = None
            db.add(sess)
        db.commit()


def _mark_failed(session_id: UUID, error_msg: str) -> None:
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None:
            return
        sess.status = SessionStatus.failed
        sess.error_msg = error_msg[:500]
        db.add(sess)
        db.commit()
