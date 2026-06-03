"""WebSocket receiver for the Chronicle app.

The Chronicle app advertises Wyoming-over-WebSocket with `?codec=opus`. Wyoming
itself is a framed JSONL-plus-binary protocol designed for raw byte streams; how
exactly Chronicle maps it onto WS frames is **not yet verified against a live
capture** (the spec flags this as a Phase 0 mitmproxy task).

So this handler is intentionally permissive:
  * Any JSON text frame with `type == "audio-start"` records sample rate / codec.
  * Any JSON text frame with `type == "audio-stop"` (or socket close) finalizes.
  * Any binary frame's bytes are appended to `storage/{session_id}.opus`,
    regardless of whether they were preceded by an `audio-chunk` header.
  * Unknown event types are logged and ignored.

Once we have a real capture we'll tighten the parser and add a contract test.
"""

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from sqlmodel import Session

from ..auth import decode_token
from ..config import settings
from ..db import engine
from ..models import AudioSession, SessionStatus

router = APIRouter(tags=["audio"])
logger = logging.getLogger("omilog.audio_ws")


def _extract_token(ws: WebSocket, query_token: str | None) -> str | None:
    auth_header = ws.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return query_token


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
        # Surface *why* the token was rejected. We avoid logging the full
        # token (could be replayed) but log enough to diagnose: the first
        # 12 chars and the unverified payload (it's just base64, not secret).
        import base64
        import json as _json

        payload_summary = "?"
        try:
            parts = raw_token.split(".")
            if len(parts) >= 2:
                pad = "=" * (-len(parts[1]) % 4)
                payload_summary = _json.loads(
                    base64.urlsafe_b64decode(parts[1] + pad)
                )
        except Exception:
            pass
        logger.info(
            "ws: rejected token (prefix=%s payload=%s) reason=%s",
            raw_token[:12],
            payload_summary,
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
    device_name: str | None = None
    client_id: str | None = None
    bytes_written = 0
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
                        device_name = data.get("name") or device_name
                        client_id = data.get("client_id") or client_id
                        logger.info(
                            "ws: audio-start session=%s rate=%s device=%s",
                            session_id,
                            sample_rate_hz,
                            device_name,
                        )
                    elif etype == "audio-chunk":
                        # If the chunk's payload is inlined as base64 or hex in
                        # `data`, surface it for later analysis. Real binary
                        # payloads arrive in subsequent binary frames.
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
                    f.write(payload)
                    bytes_written += len(payload)
    except WebSocketDisconnect:
        logger.info("ws: session %s disconnected", session_id)
    except Exception:
        logger.exception("ws: session %s errored", session_id)
        _finalize(session_id, bytes_written, sample_rate_hz, device_name, client_id,
                  started_at, status_=SessionStatus.failed,
                  error_msg="ws handler raised")
        try:
            await ws.close()
        except Exception:
            pass
        return

    final_status = (
        SessionStatus.pending_stt if bytes_written else SessionStatus.silent
    )
    _finalize(
        session_id, bytes_written, sample_rate_hz, device_name, client_id,
        started_at, status_=final_status,
    )
    logger.info(
        "ws: session %s closed bytes=%d stop_event=%s -> %s",
        session_id,
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
