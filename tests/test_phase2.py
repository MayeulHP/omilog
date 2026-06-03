"""Phase 2: LLM extraction + endpoints.

LLM call is mocked everywhere — none of these tests touch a real model. The
runner LLM stage is exercised against a synthetic transcript fixture.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from omilog.db import engine
from omilog.models import (
    ActionItem,
    AudioSession,
    CalendarEvent,
    Conversation,
    PersonMention,
    SessionStatus,
    Transcript,
)
from omilog.pipeline import extract, llm, runner
from omilog.pipeline.llm import ChatResult


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_clean_json():
    raw = json.dumps(
        {
            "title": "Déjeuner avec Marie",
            "summary": "Discussion brève sur le travail.",
            "topics": ["déjeuner", "travail"],
            "calendar_events": [
                {
                    "title": "Déjeuner",
                    "starts_at": "2026-06-15T12:30:00+02:00",
                    "ends_at": None,
                    "location": "Bastille",
                    "attendees": ["Marie"],
                    "confidence": 0.8,
                }
            ],
            "action_items": [
                {"text": "Envoyer la présentation", "owner": "user", "due_at": None}
            ],
            "people_mentioned": [{"name": "Marie", "context": "collègue"}],
        }
    )
    e = extract.parse(raw)
    assert e.title == "Déjeuner avec Marie"
    assert e.topics == ["déjeuner", "travail"]
    assert e.calendar_events[0]["title"] == "Déjeuner"
    assert e.action_items[0]["owner"] == "user"
    assert e.people_mentioned[0]["name"] == "Marie"


def test_parse_strips_think_block():
    raw = (
        "<think>The user wants me to extract...</think>\n"
        '{"title": "A", "summary": "B", "topics": [], '
        '"calendar_events": [], "action_items": [], "people_mentioned": []}'
    )
    e = extract.parse(raw)
    assert e.title == "A"


def test_parse_strips_code_fence():
    raw = (
        "```json\n"
        '{"title": "X", "summary": "Y", "topics": [], '
        '"calendar_events": [], "action_items": [], "people_mentioned": []}\n'
        "```"
    )
    e = extract.parse(raw)
    assert e.title == "X"


def test_parse_recovers_from_prose_wrap():
    raw = (
        "Sure! Here's the extraction:\n"
        '{"title": "Wrap", "summary": "yes", "topics": [], '
        '"calendar_events": [], "action_items": [], "people_mentioned": []}\n'
        "Let me know if you need anything else."
    )
    e = extract.parse(raw)
    assert e.title == "Wrap"


def test_parse_raises_on_malformed():
    # Garbage is still garbage even with json_repair — the leading "{not" is
    # malformed enough that we can't recover a useful dict.
    with pytest.raises(ValueError):
        extract.parse("complete nonsense no braces at all")


def test_parse_recovers_truncated_summary():
    """Realistic failure: LLM hit max_tokens mid-summary. json_repair closes
    the string and any open structures so we still get title + partial summary
    + the empty arrays."""
    truncated = (
        '{\n'
        '  "title": "Test de session et projets personnels",\n'
        '  "summary": "L\'utilisateur effectue un test audio et évoque des intentions de '
    )
    e = extract.parse(truncated)
    assert e.title == "Test de session et projets personnels"
    assert e.summary is not None
    assert "L'utilisateur" in e.summary


def test_parse_was_repaired_flag_false_on_clean_input():
    raw = json.dumps(
        {
            "title": "OK",
            "summary": "Clean.",
            "topics": [],
            "calendar_events": [],
            "action_items": [],
            "people_mentioned": [],
        }
    )
    e = extract.parse(raw)
    assert e.was_repaired is False


def test_parse_was_repaired_flag_true_on_truncation():
    truncated = '{\n  "title": "T",\n  "summary": "Mid-sentence'
    e = extract.parse(truncated)
    assert e.was_repaired is True


def test_parse_recovers_partial_events_list():
    """Truncation in the middle of an events array — earlier complete events
    should survive, partial last entry is allowed to be empty/dropped."""
    partial = (
        '{\n'
        '  "title": "Meeting",\n'
        '  "summary": "Discussed Q4 plans.",\n'
        '  "calendar_events": [\n'
        '    {"title": "Follow-up", "starts_at": "2026-06-15T10:00:00Z",\n'
        '     "ends_at": null, "location": "HQ", "attendees": [], "confidence": 0.9},\n'
        '    {"title": "Standup", "starts_at": "2026-06-16'
    )
    e = extract.parse(partial)
    assert e.title == "Meeting"
    assert e.summary == "Discussed Q4 plans."
    # First event was complete — must survive.
    titles = [evt.get("title") for evt in e.calendar_events]
    assert "Follow-up" in titles


def test_parse_iso8601_handles_z_suffix():
    assert extract.parse_iso8601("2026-06-15T10:00:00Z") == datetime(
        2026, 6, 15, 10, 0, tzinfo=timezone.utc
    )


def test_parse_iso8601_returns_none_on_garbage():
    assert extract.parse_iso8601("not a date") is None
    assert extract.parse_iso8601(None) is None
    assert extract.parse_iso8601("") is None


def test_build_messages_includes_date_and_timestamps():
    segs = [
        {"start": 0.0, "text": "Salut Marie."},
        {"start": 5.4, "text": "On se voit demain à 19h ?"},
    ]
    msgs = extract.build_messages(
        transcript_text="ignored",
        transcript_segments=segs,
        now=datetime(2026, 6, 3, 10, 0),
        timezone_label="Europe/Paris",
    )
    assert msgs[0]["role"] == "system"
    assert "/no_think" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "Europe/Paris" in user
    assert "2026-06-03" in user
    assert "[00:00] Salut Marie." in user
    assert "[00:05] On se voit demain à 19h ?" in user


# ──────────────────────────────────────────────────────────────────────────────
# LLM client (mocked httpx)
# ──────────────────────────────────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, payload, status=200, text=None):
        self.status_code = status
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def post(self, url, json=None):  # noqa: A002 — matches httpx API
        _FakeAsyncClient.last_url = url
        _FakeAsyncClient.last_body = json
        return _FakeAsyncClient.next_response


async def test_chat_json_parses_openai_response():
    _FakeAsyncClient.next_response = _FakeResp(
        {
            "choices": [
                {"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}
            ]
        }
    )
    with patch.object(llm.httpx, "AsyncClient", _FakeAsyncClient):
        result = await llm.chat_json(
            base_url="http://gpu.tailnet:8081/v1",
            model="qwen3",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert _FakeAsyncClient.last_url == "http://gpu.tailnet:8081/v1/chat/completions"
    assert _FakeAsyncClient.last_body["response_format"] == {"type": "json_object"}
    assert result.text == '{"ok": true}'
    assert result.finish_reason == "stop"


async def test_chat_json_disabled_when_url_blank():
    with pytest.raises(llm.LLMError):
        await llm.chat_json(base_url="", model="x", messages=[])


async def test_chat_json_surfaces_http_error():
    _FakeAsyncClient.next_response = _FakeResp({}, status=500, text="oh no")
    with patch.object(llm.httpx, "AsyncClient", _FakeAsyncClient), pytest.raises(
        llm.LLMError
    ):
        await llm.chat_json(
            base_url="http://x", model="m", messages=[{"role": "user", "content": ""}]
        )


# ──────────────────────────────────────────────────────────────────────────────
# Runner LLM stage
# ──────────────────────────────────────────────────────────────────────────────

def _insert_session_with_transcript(text: str, segments: list[dict]) -> UUID:
    sid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="test",
                audio_path="/tmp/fake.opus",
                codec="opus",
                started_at=datetime(2026, 6, 3, 10, 0, tzinfo=timezone.utc),
                status=SessionStatus.pending_llm,
            )
        )
        db.add(
            Transcript(
                audio_session_id=sid,
                text=text,
                segments_json=json.dumps(segments),
                language="fr",
                model="whisper-large-v3-turbo",
            )
        )
        db.commit()
    return sid


def _get_session(sid: UUID) -> AudioSession:
    with Session(engine) as db:
        s = db.get(AudioSession, sid)
        assert s is not None
        return s


async def test_runner_llm_happy_path(monkeypatch):
    sid = _insert_session_with_transcript(
        "Salut Marie, on se voit demain à 19h à la Bastille.",
        [
            {"start": 0.0, "text": "Salut Marie."},
            {"start": 2.0, "text": "On se voit demain à 19h à la Bastille."},
        ],
    )
    monkeypatch.setattr(
        runner.settings, "llm_base_url", "http://gpu.tailnet:8081/v1", raising=False
    )

    fake_extraction = {
        "title": "Rendez-vous Bastille",
        "summary": "Marie et le porteur se voient demain à 19h.",
        "topics": ["rendez-vous"],
        "calendar_events": [
            {
                "title": "Marie — Bastille",
                "starts_at": "2026-06-04T19:00:00+02:00",
                "ends_at": None,
                "location": "Bastille",
                "attendees": ["Marie"],
                "confidence": 0.9,
            }
        ],
        "action_items": [],
        "people_mentioned": [{"name": "Marie", "context": "amie"}],
    }
    with patch.object(
        runner,
        "chat_json",
        new=AsyncMock(
            return_value=ChatResult(
                text=json.dumps(fake_extraction), finish_reason="stop", raw={}
            )
        ),
    ):
        await runner.process_llm(sid)

    sess = _get_session(sid)
    assert sess.status == SessionStatus.done
    assert sess.error_msg is None

    with Session(engine) as db:
        conv = db.exec(
            select(Conversation).where(Conversation.audio_session_id == sid)
        ).first()
        assert conv is not None
        assert conv.title == "Rendez-vous Bastille"
        assert json.loads(conv.topics_json) == ["rendez-vous"]

        events = db.exec(
            select(CalendarEvent).where(CalendarEvent.conversation_id == conv.id)
        ).all()
        assert len(events) == 1
        assert events[0].location == "Bastille"
        assert events[0].confidence == 0.9
        assert events[0].starts_at is not None

        people = db.exec(
            select(PersonMention).where(PersonMention.conversation_id == conv.id)
        ).all()
        assert [p.name for p in people] == ["Marie"]


async def test_runner_llm_failure_marks_failed(monkeypatch):
    sid = _insert_session_with_transcript("trivial", [{"start": 0, "text": "trivial"}])
    monkeypatch.setattr(
        runner.settings, "llm_base_url", "http://gpu.tailnet:8081/v1", raising=False
    )
    with patch.object(
        runner, "chat_json", new=AsyncMock(side_effect=runner.LLMError("503"))
    ):
        await runner.process_llm(sid)
    sess = _get_session(sid)
    assert sess.status == SessionStatus.failed
    assert "llm" in (sess.error_msg or "")


async def test_runner_llm_bad_json_marks_failed(monkeypatch):
    sid = _insert_session_with_transcript("trivial", [{"start": 0, "text": "x"}])
    monkeypatch.setattr(
        runner.settings, "llm_base_url", "http://gpu.tailnet:8081/v1", raising=False
    )
    with patch.object(
        runner,
        "chat_json",
        new=AsyncMock(
            return_value=ChatResult(text="not json", finish_reason="stop", raw={})
        ),
    ):
        await runner.process_llm(sid)
    sess = _get_session(sid)
    assert sess.status == SessionStatus.failed
    assert "llm-parse" in (sess.error_msg or "")


# ──────────────────────────────────────────────────────────────────────────────
# Query endpoints
# ──────────────────────────────────────────────────────────────────────────────

def _seed_one_conversation(user: str = "test") -> tuple[UUID, UUID]:
    """Return (session_id, conversation_id) with one event + one action item + one person.

    Inserts in two passes — SQLAlchemy doesn't know about the FK dependency
    between AudioSession→Conversation→{children} when we use manual UUIDs
    instead of relationship() declarations, so it can reorder INSERTs and the
    FK constraint fails. db.flush() after each level guarantees the parent
    rows exist before children reference them.
    """
    sid = uuid4()
    cid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id=user,
                audio_path="/tmp/fake.opus",
                codec="opus",
                started_at=datetime(2026, 6, 3, 10, 0, tzinfo=timezone.utc),
                status=SessionStatus.done,
            )
        )
        db.flush()
        db.add(
            Conversation(
                id=cid,
                audio_session_id=sid,
                user_id=user,
                title="Test conv",
                summary="A summary.",
                topics_json=json.dumps(["test", "phase2"]),
                started_at=datetime(2026, 6, 3, 10, 0, tzinfo=timezone.utc),
                ended_at=datetime(2026, 6, 3, 10, 5, tzinfo=timezone.utc),
            )
        )
        db.add(
            Transcript(
                audio_session_id=sid,
                text="Salut.",
                segments_json=json.dumps([{"start": 0, "text": "Salut."}]),
                language="fr",
                model="whisper-large-v3-turbo",
            )
        )
        db.flush()
        db.add(
            CalendarEvent(
                conversation_id=cid,
                title="Future thing",
                starts_at=datetime(2099, 1, 1, 12, 0, tzinfo=timezone.utc),
                location="HQ",
                attendees_json=json.dumps(["Alice"]),
                confidence=0.7,
            )
        )
        db.add(
            ActionItem(
                conversation_id=cid,
                text="Send the doc",
                owner="user",
            )
        )
        db.add(PersonMention(conversation_id=cid, name="Alice", context="collègue"))
        db.commit()
    return sid, cid


def test_conversations_list_returns_seeded_conv(client: TestClient, auth_token: str):
    _seed_one_conversation()
    r = client.get(
        "/api/conversations", headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert r.status_code == 200
    data = r.json()
    assert any(c["title"] == "Test conv" for c in data)


def test_conversations_detail_bundles_everything(client: TestClient, auth_token: str):
    _, cid = _seed_one_conversation()
    r = client.get(
        f"/api/conversations/{cid}",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "Test conv"
    assert body["topics"] == ["test", "phase2"]
    assert body["transcript"]["text"] == "Salut."
    assert body["calendar_events"][0]["title"] == "Future thing"
    assert body["action_items"][0]["text"] == "Send the doc"
    assert body["people_mentioned"][0]["name"] == "Alice"


def test_events_upcoming_filter(client: TestClient, auth_token: str):
    _seed_one_conversation()
    r = client.get(
        "/api/events?upcoming=true",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert any(e["title"] == "Future thing" for e in data)


def test_action_items_default_open(client: TestClient, auth_token: str):
    _seed_one_conversation()
    r = client.get(
        "/api/action-items",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert r.status_code == 200
    data = r.json()
    assert any(a["text"] == "Send the doc" and a["status"] == "open" for a in data)


def test_people_aggregates_mentions(client: TestClient, auth_token: str):
    _seed_one_conversation()
    r = client.get(
        "/api/people", headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert r.status_code == 200
    data = r.json()
    alice = next((p for p in data if p["name"] == "Alice"), None)
    assert alice is not None
    assert alice["mention_count"] >= 1
    assert alice["latest_context"] == "collègue"


def test_events_requires_auth(client: TestClient):
    assert client.get("/api/events").status_code == 401


def test_action_items_requires_auth(client: TestClient):
    assert client.get("/api/action-items").status_code == 401


def test_people_requires_auth(client: TestClient):
    assert client.get("/api/people").status_code == 401
