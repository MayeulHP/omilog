"""Status dashboard + conversation deletion tests."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from omilog.config import settings
from omilog.db import engine
from omilog.models import (
    ActionItem,
    AudioSession,
    CalendarEvent,
    Conversation,
    PersonMention,
    SessionStatus,
    Transcript,
    WakeAction,
    WakeInvocation,
)


def _login(client: TestClient, password: str) -> None:
    client.post("/login", data={"username": "test", "password": password})


# ──────────────────────────────────────────────────────────────────────────────
# /status
# ──────────────────────────────────────────────────────────────────────────────

def test_status_page_requires_auth(client: TestClient):
    r = client.get("/status", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_status_renders_empty_state(client: TestClient, password: str, monkeypatch):
    # No backends configured → both show "disabled".
    monkeypatch.setattr(settings, "stt_base_url", "", raising=False)
    monkeypatch.setattr(settings, "llm_base_url", "", raising=False)
    _login(client, password)
    r = client.get("/status")
    assert r.status_code == 200
    assert "Backends" in r.text
    assert "Pipeline" in r.text
    assert "Storage" in r.text
    # Pipeline counts default to 0 when nothing pending.
    assert "pending_vad" in r.text


def test_status_reports_backend_up(client: TestClient, password: str, monkeypatch):
    monkeypatch.setattr(
        settings, "stt_base_url", "http://gpu.tailnet:8080", raising=False
    )
    monkeypatch.setattr(
        settings, "llm_base_url", "http://gpu.tailnet:8081/v1", raising=False
    )

    # Mock httpx so the page can reach our fake backends instantly.
    class _FakeResp:
        status_code = 200

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get(self, url):
            return _FakeResp()

    with patch("omilog.web.routes.httpx.AsyncClient", _FakeClient):
        _login(client, password)
        r = client.get("/status")
    assert r.status_code == 200
    assert "✓ up" in r.text


def test_status_reports_backend_down(client: TestClient, password: str, monkeypatch):
    import httpx

    monkeypatch.setattr(settings, "stt_base_url", "http://nope:8080", raising=False)
    monkeypatch.setattr(settings, "llm_base_url", "", raising=False)

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get(self, url):
            raise httpx.ConnectError("refused")

    with patch("omilog.web.routes.httpx.AsyncClient", _FakeClient):
        _login(client, password)
        r = client.get("/status")
    assert r.status_code == 200
    assert "ConnectError" in r.text


def test_status_pipeline_counts_reflect_db(client: TestClient, password: str, monkeypatch):
    now = datetime.now(timezone.utc)
    with Session(engine) as db:
        for _ in range(3):
            db.add(
                AudioSession(
                    user_id="test",
                    audio_path="/tmp/x.opus",
                    codec="opus",
                    started_at=now,
                    status=SessionStatus.pending_stt,
                )
            )
        db.add(
            AudioSession(
                user_id="test",
                audio_path="/tmp/y.opus",
                codec="opus",
                # Use real wall-clock so it lands within the "last 24h" recent-failures window.
                started_at=now - timedelta(minutes=10),
                status=SessionStatus.failed,
                error_msg="something blew up",
            )
        )
        db.commit()

    # Avoid hitting the network during the test.
    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def get(self, url):
            raise Exception("disabled")

    monkeypatch.setattr(settings, "stt_base_url", "", raising=False)
    monkeypatch.setattr(settings, "llm_base_url", "", raising=False)
    _login(client, password)
    with patch("omilog.web.routes.httpx.AsyncClient", _FakeClient):
        r = client.get("/status")
    assert r.status_code == 200
    # 3 pending_stt + 1 failed should both surface in the rendered HTML.
    assert ">3<" in r.text  # the pending_stt card
    assert "something blew up" in r.text  # the recent failures table


# ──────────────────────────────────────────────────────────────────────────────
# Conversation deletion
# ──────────────────────────────────────────────────────────────────────────────

def _seed_conversation_with_children(
    *, user: str = "test", audio_file: Path | None = None
) -> tuple[UUID, UUID]:
    """Return (audio_session_id, conversation_id) plus children rows."""
    sid = uuid4()
    cid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id=user,
                audio_path=str(audio_file) if audio_file else "/tmp/x.opus",
                codec="opus",
                started_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
                status=SessionStatus.done,
            )
        )
        db.flush()
        db.add(
            Conversation(
                id=cid,
                audio_session_id=sid,
                user_id=user,
                title="To delete",
                summary="...",
                started_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
            )
        )
        db.add(
            Transcript(
                audio_session_id=sid,
                text="...",
                language="fr",
                model="whisper",
            )
        )
        db.flush()
        db.add(
            CalendarEvent(
                conversation_id=cid,
                title="Some event",
                starts_at=datetime(2026, 6, 4, tzinfo=timezone.utc),
                confidence=0.9,
            )
        )
        db.add(
            ActionItem(
                conversation_id=cid, text="something to do", owner="user"
            )
        )
        db.add(PersonMention(conversation_id=cid, name="Marie"))
        db.commit()
    return sid, cid


def test_conversation_delete_removes_all_dependents(
    client: TestClient, password: str, tmp_path
):
    audio_file = Path(__import__("os").environ["OMILOG_STORAGE_DIR"]) / f"{uuid4()}.opus"
    audio_file.write_bytes(b"opus-bytes")

    sid, cid = _seed_conversation_with_children(audio_file=audio_file)
    _login(client, password)
    r = client.post(
        f"/conversations/{cid}/delete", follow_redirects=False
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/"

    with Session(engine) as db:
        assert db.get(Conversation, cid) is None
        assert db.get(AudioSession, sid) is None
        # All children gone:
        assert list(db.exec(select(CalendarEvent))) == []
        assert list(db.exec(select(ActionItem))) == []
        assert list(db.exec(select(PersonMention))) == []
        assert list(db.exec(select(Transcript))) == []
    assert not audio_file.exists(), "audio file should have been unlinked"


def test_conversation_delete_removes_wake_invocations(
    client: TestClient, password: str
):
    sid, cid = _seed_conversation_with_children()
    aid = uuid4()
    with Session(engine) as db:
        db.add(
            WakeAction(
                id=aid,
                user_id="test",
                name="x",
                phrases_json=json.dumps(["X"]),
                command="echo x",
            )
        )
        db.flush()
        db.add(
            WakeInvocation(
                wake_action_id=aid,
                conversation_id=cid,
                matched_phrase="X",
                input_text="hi",
                command_resolved="echo hi",
                exit_code=0,
                stdout="",
                stderr="",
                duration_ms=1,
            )
        )
        db.commit()

    _login(client, password)
    r = client.post(f"/conversations/{cid}/delete", follow_redirects=False)
    assert r.status_code == 303
    with Session(engine) as db:
        # Wake action stays, only its invocation for this conv is gone.
        assert db.get(WakeAction, aid) is not None
        assert list(db.exec(select(WakeInvocation))) == []


def test_conversation_delete_404_for_other_user(client: TestClient, password: str):
    _sid, cid = _seed_conversation_with_children(user="not-me")
    _login(client, password)
    r = client.post(f"/conversations/{cid}/delete", follow_redirects=False)
    assert r.status_code == 404


def test_conversation_delete_requires_auth(client: TestClient):
    cid = uuid4()
    r = client.post(f"/conversations/{cid}/delete", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"
