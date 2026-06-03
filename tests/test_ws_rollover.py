"""WebSocket rollover tests.

Time-based: we set a short rollover interval, sleep between packets, and
assert that a single WS connection produced two AudioSession rows. The
test takes ~0.6s — slow for unit tests but fast enough that we run it on
every push.

If these become flaky on a loaded CI box, bump _ROLLOVER_S and _SLEEP_S
proportionally.
"""

import time

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from omilog.config import settings as cfg
from omilog.db import engine
from omilog.models import AudioSession, SessionStatus


_ROLLOVER_S = 0.3
_RECV_TIMEOUT_S = 0.05
_SLEEP_BETWEEN_PACKETS = 0.6


def _list_sessions(user: str = "test") -> list[AudioSession]:
    with Session(engine) as db:
        return list(
            db.exec(
                select(AudioSession)
                .where(AudioSession.user_id == user)
                .order_by(AudioSession.started_at.asc())
            ).all()
        )


def test_rollover_splits_one_ws_into_multiple_sessions(
    client: TestClient, auth_token: str, monkeypatch
):
    monkeypatch.setattr(cfg, "ws_rollover_seconds", _ROLLOVER_S, raising=False)
    monkeypatch.setattr(
        cfg, "ws_receive_timeout_seconds", _RECV_TIMEOUT_S, raising=False
    )

    p1 = b"\xaa" * 128
    p2 = b"\xbb" * 128

    with client.websocket_connect(f"/ws?codec=opus&token={auth_token}") as ws:
        ws.send_json({"type": "audio-start", "data": {"rate": 16000}})
        ws.send_bytes(p1)
        time.sleep(_SLEEP_BETWEEN_PACKETS)  # rollover should fire mid-sleep
        ws.send_bytes(p2)
        ws.send_json({"type": "audio-stop", "data": {}})

    sessions = _list_sessions()
    assert len(sessions) == 2, f"expected rollover into 2 sessions, got {len(sessions)}"
    # Default VAD enabled → pending_vad terminal state.
    for s in sessions:
        assert s.status == SessionStatus.pending_vad
        assert s.bytes_written == 128


def test_no_rollover_when_interval_zero(
    client: TestClient, auth_token: str, monkeypatch
):
    monkeypatch.setattr(cfg, "ws_rollover_seconds", 0.0, raising=False)
    monkeypatch.setattr(
        cfg, "ws_receive_timeout_seconds", _RECV_TIMEOUT_S, raising=False
    )

    p1 = b"\xcc" * 64
    p2 = b"\xdd" * 64

    with client.websocket_connect(f"/ws?codec=opus&token={auth_token}") as ws:
        ws.send_json({"type": "audio-start", "data": {"rate": 16000}})
        ws.send_bytes(p1)
        # Even with a long delay, no rollover when disabled.
        time.sleep(_SLEEP_BETWEEN_PACKETS)
        ws.send_bytes(p2)
        ws.send_json({"type": "audio-stop", "data": {}})

    sessions = _list_sessions()
    assert len(sessions) == 1
    assert sessions[0].bytes_written == 128  # both packets in one segment


def test_empty_segment_not_rolled_over(
    client: TestClient, auth_token: str, monkeypatch
):
    """If no packets have been written, an idle wait shouldn't produce empty
    segments — rollover only fires when there's something to flush."""
    monkeypatch.setattr(cfg, "ws_rollover_seconds", _ROLLOVER_S, raising=False)
    monkeypatch.setattr(
        cfg, "ws_receive_timeout_seconds", _RECV_TIMEOUT_S, raising=False
    )

    with client.websocket_connect(f"/ws?codec=opus&token={auth_token}") as ws:
        ws.send_json({"type": "audio-start", "data": {"rate": 16000}})
        time.sleep(_SLEEP_BETWEEN_PACKETS)  # idle, no rollover should occur
        ws.send_bytes(b"\xee" * 64)
        ws.send_json({"type": "audio-stop", "data": {}})

    sessions = _list_sessions()
    assert len(sessions) == 1
    assert sessions[0].bytes_written == 64
