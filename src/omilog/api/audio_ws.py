"""WebSocket receiver for the Chronicle app.

Wyoming-over-WebSocket framing isn't fully documented; we've reverse-engineered
enough from real traffic that:
  * JSON text frames carry control events (audio-start, audio-stop, etc.).
  * Each binary frame carries exactly one Opus packet.

We mux those packets into an Ogg container on the fly (see audio/ogg_opus.py),
so each `storage/{session_id}.opus` is a real, playable file. Long-running
BLE captures get **rolled over** every `OMILOG_WS_ROLLOVER_SECONDS` so the
pipeline (VAD → STT → LLM) can chew through chunks throughout the day, rather
than waiting for the phone to disconnect at end-of-day.
"""

import asyncio
import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from uuid import UUID, uuid4

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status
from sqlmodel import Session

from ..audio.ogg_opus import OggOpusWriter
from ..auth import decode_token
from ..config import settings
from ..db import engine
from ..models import AudioSession, SessionStatus

router = APIRouter(tags=["audio"])
logger = logging.getLogger("omilog.audio_ws")

# Fallback when audio-start arrives without rate.
_DEFAULT_RATE = 16000
_DEFAULT_CHANNELS = 1


def _extract_token(ws: WebSocket, query_token: str | None) -> str | None:
    auth_header = ws.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return query_token


def _summarize_jwt_payload(token: str) -> object:
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return "?"
        pad = "=" * (-len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(parts[1] + pad))
    except Exception:
        return "?"


# ──────────────────────────────────────────────────────────────────────────────
# One segment of audio between BLE-open and rollover (or BLE-close).
# ──────────────────────────────────────────────────────────────────────────────

class WSSegment:
    """One file, one DB row, one Ogg writer. Cleanly closeable mid-WS so we
    can immediately open the next segment without dropping packets."""

    def __init__(
        self,
        *,
        user_id: str,
        codec: str,
        device_name: str | None,
        client_id: str | None,
        sample_rate: int,
        channels: int,
        started_at: datetime,
    ) -> None:
        self.session_id: UUID = uuid4()
        self.audio_path: Path = settings.storage_dir / f"{self.session_id}.opus"
        self.audio_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_id = user_id
        self.codec = codec
        self.device_name = device_name
        self.client_id = client_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.started_at = started_at

        self._file = self.audio_path.open("wb")
        self._writer: OggOpusWriter | None = None
        self.bytes_written = 0
        self.packets_written = 0
        self._closed = False
        self._insert_recording_row()
        logger.info(
            "ws: segment %s opened user=%s codec=%s",
            self.session_id,
            self.user_id,
            self.codec,
        )

    def _insert_recording_row(self) -> None:
        with Session(engine) as db:
            db.add(
                AudioSession(
                    id=self.session_id,
                    user_id=self.user_id,
                    codec=self.codec,
                    device_name=self.device_name,
                    client_id=self.client_id,
                    audio_path=str(self.audio_path),
                    started_at=self.started_at,
                    status=SessionStatus.recording,
                )
            )
            db.commit()

    def write_packet(self, payload: bytes) -> None:
        if self._closed:
            raise RuntimeError(f"WSSegment {self.session_id} already closed")
        if self._writer is None:
            self._writer = OggOpusWriter(
                self._file,
                sample_rate=self.sample_rate,
                channels=self.channels,
            )
        self._writer.write_packet(payload)
        self.bytes_written += len(payload)
        self.packets_written += 1

    def close(self, *, was_rollover: bool = False) -> None:
        if self._closed:
            return
        if self._writer is not None:
            self._writer.close()
        self._file.close()
        self._closed = True

        ended_at = datetime.now(timezone.utc)
        # Picked status: empty file → silent; otherwise hand off to the pipeline
        # (pending_vad if VAD is enabled, else straight to STT).
        if self.packets_written == 0:
            final_status = SessionStatus.silent
        elif settings.vad_enabled:
            final_status = SessionStatus.pending_vad
        else:
            final_status = SessionStatus.pending_stt

        with Session(engine) as db:
            sess = db.get(AudioSession, self.session_id)
            if sess is not None:
                sess.ended_at = ended_at
                sess.duration_s = (ended_at - self.started_at).total_seconds()
                sess.sample_rate_hz = self.sample_rate
                sess.bytes_written = self.bytes_written
                sess.status = final_status
                db.add(sess)
                db.commit()
        logger.info(
            "ws: segment %s closed (rollover=%s) packets=%d bytes=%d -> %s",
            self.session_id,
            was_rollover,
            self.packets_written,
            self.bytes_written,
            final_status.value,
        )

    def mark_failed(self, error_msg: str) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        try:
            self._file.close()
        except Exception:
            pass
        self._closed = True
        with Session(engine) as db:
            sess = db.get(AudioSession, self.session_id)
            if sess is not None:
                sess.status = SessionStatus.failed
                sess.error_msg = error_msg[:500]
                sess.ended_at = datetime.now(timezone.utc)
                db.add(sess)
                db.commit()


# ──────────────────────────────────────────────────────────────────────────────
# WS endpoint
# ──────────────────────────────────────────────────────────────────────────────

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

    # Connection-level state — survives rollovers within one BLE/WS session.
    sample_rate_hz: int = _DEFAULT_RATE
    channels: int = _DEFAULT_CHANNELS
    device_name: str | None = None
    client_id: str | None = None
    rollover_interval = settings.ws_rollover_seconds
    recv_timeout = settings.ws_receive_timeout_seconds

    def _new_segment() -> WSSegment:
        return WSSegment(
            user_id=user_id,
            codec=codec,
            device_name=device_name,
            client_id=client_id,
            sample_rate=sample_rate_hz,
            channels=channels,
            started_at=datetime.now(timezone.utc),
        )

    segment = _new_segment()
    segment_start = monotonic()
    stop_event_seen = False

    try:
        while True:
            # Time out on idle WS so the rollover check fires even when no
            # binary frames arrive. wait_for cancels the inner receive on
            # timeout, which is safe for Starlette's queue-backed WebSocket.
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=recv_timeout)
            except asyncio.TimeoutError:
                msg = None

            if msg is not None:
                if msg.get("type") == "websocket.disconnect":
                    break

                text = msg.get("text")
                if text is not None:
                    try:
                        evt = json.loads(text)
                    except json.JSONDecodeError:
                        logger.warning("ws: non-JSON text frame: %r", text[:200])
                    else:
                        etype = evt.get("type")
                        data = evt.get("data") or {}
                        if etype == "audio-start":
                            new_rate = data.get("rate")
                            new_channels = data.get("channels")
                            if new_rate and new_rate != sample_rate_hz:
                                sample_rate_hz = new_rate
                                # If the current segment hasn't written packets
                                # yet, retro-fit it; otherwise the existing
                                # Ogg stream keeps its original rate.
                                if segment.packets_written == 0:
                                    segment.sample_rate = sample_rate_hz
                            if new_channels:
                                channels = new_channels
                                if segment.packets_written == 0:
                                    segment.channels = channels
                            device_name = data.get("name") or device_name
                            client_id = data.get("client_id") or client_id
                            logger.info(
                                "ws: audio-start session=%s rate=%s channels=%s device=%s",
                                segment.session_id,
                                sample_rate_hz,
                                channels,
                                device_name,
                            )
                        elif etype == "audio-stop":
                            stop_event_seen = True
                            logger.info(
                                "ws: audio-stop session=%s", segment.session_id
                            )
                            break
                        elif etype == "audio-chunk":
                            if "audio" in data:
                                logger.debug(
                                    "ws: audio-chunk had inline payload field"
                                )
                        else:
                            logger.debug("ws: event %r data=%r", etype, data)
                else:
                    payload = msg.get("bytes")
                    if payload:
                        # Reset the rollover timer on the *first* packet of
                        # this segment so the rollover interval measures
                        # speech duration, not "time since segment object was
                        # constructed." Otherwise a 30-min silence at the
                        # start would cause the first incoming packet to
                        # instantly trigger a rollover into a tiny segment.
                        if segment.packets_written == 0:
                            segment_start = monotonic()
                        segment.write_packet(payload)

            # Rollover check — run on every loop iteration, not just on message
            # arrival, so a long silence still rolls over on schedule.
            if (
                rollover_interval > 0
                and segment.packets_written > 0
                and (monotonic() - segment_start) >= rollover_interval
            ):
                logger.info(
                    "ws: rollover after %.0fs session=%s",
                    monotonic() - segment_start,
                    segment.session_id,
                )
                segment.close(was_rollover=True)
                segment = _new_segment()
                segment_start = monotonic()

    except WebSocketDisconnect:
        logger.info("ws: disconnected session=%s", segment.session_id)
    except Exception as e:
        logger.exception("ws: errored session=%s", segment.session_id)
        segment.mark_failed(f"ws handler raised: {type(e).__name__}: {e}")
        try:
            await ws.close()
        except Exception:
            pass
        return

    segment.close(was_rollover=False)
    logger.info(
        "ws: session over (stop_event=%s)",
        stop_event_seen,
    )
    try:
        await ws.close()
    except Exception:
        pass
