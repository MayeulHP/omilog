"""WebSocket receiver for the Chronicle app.

Wyoming-over-WebSocket framing isn't fully documented; we've reverse-engineered
enough from real traffic that:
  * JSON text frames carry control events (audio-start, audio-stop, etc.).
  * Each binary frame carries exactly one Opus packet.

We mux those packets into an Ogg container on the fly (see audio/ogg_opus.py),
so the resulting `storage/{session_id}.opus` is a real, playable file rather
than raw packets. If the per-frame=per-packet assumption ever breaks (e.g.
Chronicle bundles multiple packets per frame), Ogg page CRC errors will surface
in ffprobe and we'll know to add a deframer.
"""

import base64
import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from sqlmodel import Session

from ..audio.ogg_opus import OggOpusWriter
from ..auth import decode_token
from ..config import settings
from ..db import engine
from ..models import AudioSession, SessionStatus

router = APIRouter(tags=["audio"])
logger = logging.getLogger("omilog.audio_ws")

# Fallback when audio-start arrives without rate, or arrives after the first
# binary frame (defensive — shouldn't happen with Chronicle but might with
# future clients).
_DEFAULT_RATE = 16000
_DEFAULT_CHANNELS = 1


def _extract_token(ws: WebSocket, query_token: str | None) -> str | None:
    auth_header = ws.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return query_token


def _summarize_jwt_payload(token: str) -> object:
    """Decode the unverified payload section of a JWT for logging. Safe because
    the payload is just base64; we never log the signature."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return "?"
        pad = "=" * (-len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(parts[1] + pad))
    except Exception:
        return "?"


@router.websocket("/ws")
async def audio_ws(
    ws: WebSocket,
    codec: str = Query(default="opus"),
    token: str | None = Query(default=None),
):
    raw_token = _extract_token(ws, token)
    if not raw_token:
        logger.info("ws: rejected, no token in header or ?token= query")
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    try:
        user_id = decode_token(raw_token)
    except Exception as e:
        logger.info(
            "ws: rejected token (prefix=%s payload=%s) reason=%s",
            raw_token[:12],
            _summarize_jwt_payload(raw_token),
            e,
        )
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await ws.accept()
    session_id = uuid4()
    audio_path = settings.storage_dir / f"{session_id}.opus"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)

    sample_rate_hz: int | None = None
    channels: int = _DEFAULT_CHANNELS
    device_name: str | None = None
    client_id: str | None = None
    bytes_written = 0  # raw Opus payload bytes (file on disk includes Ogg overhead)
    packets_written = 0
    stop_event_seen = False

    with Session(engine) as db:
        db.add(
            AudioSession(
                id=session_id,
                user_id=user_id,
                codec=codec,
                audio_path=str(audio_path),
                started_at=started_at,
                status=SessionStatus.recording,
            )
        )
        db.commit()

    logger.info("ws: session %s opened user=%s codec=%s", session_id, user_id, codec)

    writer: OggOpusWriter | None = None

    def _ensure_writer(f) -> OggOpusWriter:
        nonlocal writer
        if writer is None:
            writer = OggOpusWriter(
                f,
                sample_rate=sample_rate_hz or _DEFAULT_RATE,
                channels=channels,
            )
            logger.info(
                "ws: opened OggOpusWriter session=%s rate=%s channels=%s",
                session_id,
                writer._sample_rate,  # noqa: SLF001 — log-only
                channels,
            )
        return writer

    try:
        with audio_path.open("wb") as f:
            while True:
                msg = await ws.receive()
                msg_type = msg.get("type")
                if msg_type == "websocket.disconnect":
                    break

                text = msg.get("text")
                if text is not None:
                    try:
                        evt = json.loads(text)
                    except json.JSONDecodeError:
                        logger.warning("ws: non-JSON text frame: %r", text[:200])
                        continue
                    etype = evt.get("type")
                    data = evt.get("data") or {}
                    if etype == "audio-start":
                        sample_rate_hz = data.get("rate") or sample_rate_hz
                        channels = data.get("channels") or channels
                        device_name = data.get("name") or device_name
                        client_id = data.get("client_id") or client_id
                        logger.info(
                            "ws: audio-start session=%s rate=%s channels=%s device=%s",
                            session_id,
                            sample_rate_hz,
                            channels,
                            device_name,
                        )
                    elif etype == "audio-chunk":
                        if "audio" in data:
                            logger.debug("ws: audio-chunk had inline payload field")
                    elif etype == "audio-stop":
                        stop_event_seen = True
                        logger.info("ws: audio-stop session=%s", session_id)
                        break
                    else:
                        logger.debug("ws: event %r data=%r", etype, data)
                    continue

                payload = msg.get("bytes")
                if payload:
                    _ensure_writer(f).write_packet(payload)
                    bytes_written += len(payload)
                    packets_written += 1
            # Loop ended cleanly (audio-stop or disconnect): finalize Ogg stream.
            if writer is not None:
                writer.close()
    except WebSocketDisconnect:
        logger.info("ws: session %s disconnected", session_id)
        # File handle is already closed; writer.close() may not have run, but
        # the resulting file is still readable (Ogg pages before EOS are valid
        # — players just won't know where the stream ends).
    except Exception:
        logger.exception("ws: session %s errored", session_id)
        _finalize(
            session_id, bytes_written, sample_rate_hz, device_name, client_id,
            started_at, status_=SessionStatus.failed,
            error_msg="ws handler raised",
        )
        try:
            await ws.close()
        except Exception:
            pass
        return

    # If VAD is on, parents land in pending_vad and the runner segments them.
    # If VAD is off, we fall back to pending_stt to keep the pipeline moving.
    if not packets_written:
        final_status = SessionStatus.silent
    elif settings.vad_enabled:
        final_status = SessionStatus.pending_vad
    else:
        final_status = SessionStatus.pending_stt
    _finalize(
        session_id, bytes_written, sample_rate_hz, device_name, client_id,
        started_at, status_=final_status,
    )
    logger.info(
        "ws: session %s closed packets=%d bytes=%d stop_event=%s -> %s",
        session_id,
        packets_written,
        bytes_written,
        stop_event_seen,
        audio_path,
    )
    try:
        await ws.close()
    except Exception:
        pass


def _finalize(
    session_id,
    bytes_written: int,
    sample_rate_hz: int | None,
    device_name: str | None,
    client_id: str | None,
    started_at: datetime,
    status_: SessionStatus,
    error_msg: str | None = None,
) -> None:
    ended_at = datetime.now(timezone.utc)
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None:
            return
        sess.ended_at = ended_at
        sess.duration_s = (ended_at - started_at).total_seconds()
        sess.sample_rate_hz = sample_rate_hz
        sess.device_name = device_name
        sess.client_id = client_id
        sess.bytes_written = bytes_written
        sess.status = status_
        sess.error_msg = error_msg
        db.add(sess)
        db.commit()
