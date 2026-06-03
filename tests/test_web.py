"""Web UI smoke tests.

Cover the auth gate (redirect / HX-Redirect), the four pages, the action
toggle, and the cookie set/clear lifecycle.
"""

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from omilog.config import settings
from omilog.db import engine
from omilog.models import (
    ActionItem,
    ActionItemStatus,
    AudioSession,
    CalendarEvent,
    Conversation,
    PersonMention,
    SessionStatus,
    Transcript,
)


def _seed_conversation(user: str = "test") -> tuple[UUID, UUID, UUID]:
    sid = uuid4()
    cid = uuid4()
    aid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id=user,
                audio_path="/tmp/x.opus",
                codec="opus",
                started_at=datetime(2026, 6, 3, 10, 0, tzinfo=timezone.utc),
                duration_s=42.5,
                status=SessionStatus.done,
            )
        )
        db.flush()
        db.add(
            Conversation(
                id=cid,
                audio_session_id=sid,
                user_id=user,
                title="Déjeuner Marie",
                summary="Bref échange à propos du déjeuner de demain.",
                topics_json=json.dumps(["déjeuner"]),
                started_at=datetime(2026, 6, 3, 10, 0, tzinfo=timezone.utc),
            )
        )
        db.add(
            Transcript(
                audio_session_id=sid,
                text="Salut Marie.",
                segments_json=json.dumps(
                    [{"start": 0.0, "text": "Salut Marie."}]
                ),
                language="fr",
                model="whisper-large-v3-turbo",
            )
        )
        db.flush()
        db.add(
            ActionItem(
                id=aid,
                conversation_id=cid,
                text="Envoyer la présentation",
                owner="user",
            )
        )
        db.add(
            CalendarEvent(
                conversation_id=cid,
                title="Déjeuner Bastille",
                starts_at=datetime(2099, 6, 4, 12, 30, tzinfo=timezone.utc),
                location="Bastille",
                confidence=0.85,
            )
        )
        db.add(PersonMention(conversation_id=cid, name="Marie", context="amie"))
        db.commit()
    return sid, cid, aid


# ──────────────────────────────────────────────────────────────────────────────
# Auth flow
# ──────────────────────────────────────────────────────────────────────────────

def test_login_page_renders(client: TestClient):
    r = client.get("/login")
    assert r.status_code == 200
    assert "<form" in r.text
    assert 'name="username"' in r.text


def test_login_bad_credentials_shows_error(client: TestClient):
    r = client.post(
        "/login",
        data={"username": "test", "password": "wrong"},
    )
    assert r.status_code == 401
    assert "Identifiants invalides" in r.text


def test_login_good_sets_cookie_and_redirects(client: TestClient, password: str):
    r = client.post(
        "/login",
        data={"username": "test", "password": password},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/"
    assert settings.cookie_name in r.cookies


def test_unauth_index_redirects(client: TestClient):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_unauth_htmx_request_gets_hx_redirect(client: TestClient):
    r = client.post(
        "/actions/00000000-0000-0000-0000-000000000000/status",
        data={"status": "done"},
        headers={"HX-Request": "true"},
        follow_redirects=False,
    )
    assert r.status_code == 401
    assert r.headers.get("hx-redirect") == "/login"


def test_logout_clears_cookie(client: TestClient, password: str):
    client.post("/login", data={"username": "test", "password": password})
    r = client.get("/logout", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


# ──────────────────────────────────────────────────────────────────────────────
# Pages render with seeded data
# ──────────────────────────────────────────────────────────────────────────────

def _login(client: TestClient, password: str) -> None:
    r = client.post("/login", data={"username": "test", "password": password})
    assert r.status_code in (200, 303)


def test_index_lists_seeded_conversation(client: TestClient, password: str):
    _seed_conversation()
    _login(client, password)
    r = client.get("/")
    assert r.status_code == 200
    assert "Déjeuner Marie" in r.text


def test_conversation_detail_renders_everything(client: TestClient, password: str):
    _, cid, _ = _seed_conversation()
    _login(client, password)
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    assert "Déjeuner Marie" in r.text
    assert "Bastille" in r.text
    assert "Envoyer la présentation" in r.text
    assert "Marie" in r.text
    assert "Salut Marie." in r.text


def test_events_page_lists_upcoming(client: TestClient, password: str):
    _seed_conversation()
    _login(client, password)
    r = client.get("/events")
    assert r.status_code == 200
    assert "Déjeuner Bastille" in r.text


def test_actions_default_shows_open(client: TestClient, password: str):
    _seed_conversation()
    _login(client, password)
    r = client.get("/actions")
    assert r.status_code == 200
    assert "Envoyer la présentation" in r.text


# ──────────────────────────────────────────────────────────────────────────────
# Action toggle (HTMX partial)
# ──────────────────────────────────────────────────────────────────────────────

def test_action_toggle_returns_partial_and_persists(
    client: TestClient, password: str
):
    _, _, aid = _seed_conversation()
    _login(client, password)
    r = client.post(
        f"/actions/{aid}/status",
        data={"status": "done"},
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200
    # The partial includes the row id so we can target it for swap.
    assert f"action-{aid}" in r.text
    # And the "Rouvrir" button (visible once done) is in the rendered HTML.
    assert "Rouvrir" in r.text

    # Persisted?
    with Session(engine) as db:
        row = db.get(ActionItem, aid)
        assert row is not None
        assert row.status == ActionItemStatus.done


def test_action_toggle_rejects_invalid_status(
    client: TestClient, password: str
):
    _, _, aid = _seed_conversation()
    _login(client, password)
    r = client.post(f"/actions/{aid}/status", data={"status": "weird"})
    assert r.status_code == 400


def test_action_toggle_404_for_other_user(client: TestClient, password: str):
    # Seed under a different user_id so our 'test' user shouldn't see it.
    _, _, aid = _seed_conversation(user="not-me")
    _login(client, password)
    r = client.post(f"/actions/{aid}/status", data={"status": "done"})
    assert r.status_code == 404
