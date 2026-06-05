"""Decisions extraction + per-category toggle tests.

Two new behaviors:
- The LLM is asked for a new ``decisions`` array alongside calendar_events,
  action_items, and people_mentioned. Parser pulls them; runner stores them
  as Decision rows; UI renders them on the conversation detail page.
- Each category (calendar, actions, decisions, people, topics) can be
  toggled off via settings.extract_*. When off, the prompt schema drops the
  section AND the runner ignores anything the LLM returns for it anyway.
  Both halves of the toggle are tested.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
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
    Decision,
    PersonMention,
    SessionStatus,
)
from omilog.pipeline import extract, runner


# ──────────────────────────────────────────────────────────────────────────────
# Prompt: schema is built dynamically from enabled set
# ──────────────────────────────────────────────────────────────────────────────


# Schema-field assertions use the quoted form ("calendar_events", etc.)
# because the prose paragraphs naturally talk about these concepts in
# English — "for action items: prefer owner=..." — even when the JSON
# schema doesn't include them. The quoted form is unambiguous.

def test_default_prompt_includes_all_category_fields():
    p = extract.render_default_system_prompt()
    assert '"calendar_events"' in p
    assert '"action_items"' in p
    assert '"decisions"' in p
    assert '"people_mentioned"' in p
    assert '"topics"' in p


def test_prompt_omits_disabled_categories():
    """When a category is off, its schema fragment is dropped — model never
    sees the field name in the schema, can't waste tokens generating an
    empty array for it."""
    p = extract.render_default_system_prompt(
        enabled={
            "calendar_events": False,
            "action_items": True,
            "decisions": True,
            "people_mentioned": True,
            "topics": True,
        }
    )
    assert '"calendar_events"' not in p
    assert '"action_items"' in p
    assert '"decisions"' in p


def test_prompt_all_off_still_has_title_summary_quality():
    """The core fields (title, summary, quality_score, quality_reasoning) are
    NEVER toggle-able — without them the conversation list and quality
    filter would all break. Confirms they survive an everything-off pass.
    """
    p = extract.render_default_system_prompt(enabled={k: False for k in extract.DEFAULT_ENABLED})
    assert '"title"' in p
    assert '"summary"' in p
    assert '"quality_score"' in p
    assert '"quality_reasoning"' in p
    # No category fields in the schema. The prose still mentions categories
    # by name (e.g. "For action items: prefer owner=...") because that's
    # general modeling guidance — checking the quoted JSON-field form is
    # what actually proves the schema is clean.
    assert '"calendar_events"' not in p
    assert '"action_items"' not in p
    assert '"decisions"' not in p
    assert '"people_mentioned"' not in p
    assert '"topics"' not in p


def test_prompt_decisions_guidance_distinguishes_from_actions():
    """The prompt has language explaining decisions-vs-action-items to avoid
    double-counting. Regression guard against accidentally dropping it."""
    p = extract.render_default_system_prompt()
    assert "DECISION" in p or "decision" in p.lower()
    # The 'prefer action_items' guidance is the actual disambiguation rule.
    assert "action_item" in p


# ──────────────────────────────────────────────────────────────────────────────
# Parser: decisions field round-trips
# ──────────────────────────────────────────────────────────────────────────────


def test_parser_extracts_decisions():
    payload = json.dumps({
        "title": "x",
        "summary": "y",
        "quality_score": 0.7,
        "quality_reasoning": "ok",
        "calendar_events": [],
        "action_items": [],
        "decisions": [
            {"text": "We'll use Postgres", "made_by": "user", "confidence": 0.9},
            {"text": "Skip the meeting", "made_by": "Marie", "confidence": 0.6},
        ],
        "people_mentioned": [],
        "topics": [],
    })
    e = extract.parse(payload)
    assert len(e.decisions) == 2
    assert e.decisions[0]["text"] == "We'll use Postgres"
    assert e.decisions[1]["made_by"] == "Marie"


def test_parser_missing_decisions_field_defaults_to_empty_list():
    """Pre-decisions transcripts won't have the field — parse must not crash
    or default to None."""
    payload = json.dumps({
        "title": "x",
        "summary": "y",
        "quality_score": 0.5,
        "calendar_events": [],
        "action_items": [],
        "people_mentioned": [],
        "topics": [],
    })
    e = extract.parse(payload)
    assert e.decisions == []


def test_parser_rejects_non_dict_decisions():
    """Bad shapes (string instead of dict) silently dropped — same as the
    other extraction lists."""
    payload = json.dumps({
        "title": "x",
        "summary": "y",
        "decisions": [
            {"text": "good"},
            "not a dict",
            123,
            None,
        ],
    })
    e = extract.parse(payload)
    assert len(e.decisions) == 1
    assert e.decisions[0]["text"] == "good"


# ──────────────────────────────────────────────────────────────────────────────
# Runner: stores decisions, respects toggles for every category
# ──────────────────────────────────────────────────────────────────────────────


def _make_session(user: str = "test") -> UUID:
    sid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id=user,
                audio_path="/tmp/x.opus",
                codec="opus",
                started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                status=SessionStatus.done,
            )
        )
        db.commit()
    return sid


def _full_extraction() -> extract.Extraction:
    """Build an extraction that has one item in every category, so toggle
    tests can check what got stored vs dropped."""
    return extract.Extraction(
        title="t",
        summary="s",
        topics=["t1", "t2"],
        calendar_events=[{
            "title": "Lunch", "starts_at": None, "ends_at": None,
            "location": None, "attendees": [], "confidence": 0.8,
        }],
        action_items=[{"text": "send the deck", "owner": "user", "due_at": None}],
        decisions=[{"text": "Use Postgres", "made_by": "user", "confidence": 0.9}],
        people_mentioned=[{"name": "Marie", "context": "lead"}],
        quality_score=0.7,
        quality_reasoning="ok",
    )


def test_runner_stores_decisions(monkeypatch):
    monkeypatch.setattr(settings, "extract_calendar_events", True)
    monkeypatch.setattr(settings, "extract_action_items", True)
    monkeypatch.setattr(settings, "extract_decisions", True)
    monkeypatch.setattr(settings, "extract_people_mentioned", True)
    monkeypatch.setattr(settings, "extract_topics", True)
    sid = _make_session()
    conv_id = runner._save_extraction(
        session_id=sid,
        user_id="test",
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ended_at=None,
        extraction=_full_extraction(),
    )
    with Session(engine) as db:
        rows = list(
            db.exec(select(Decision).where(Decision.conversation_id == conv_id)).all()
        )
    assert len(rows) == 1
    assert rows[0].text == "Use Postgres"
    assert rows[0].made_by == "user"
    assert rows[0].confidence == 0.9


def test_runner_skips_decisions_when_toggle_off(monkeypatch):
    """Even if the LLM returned decisions (e.g. it ignored a prompt that
    omitted the field), the runner doesn't store them when the toggle is
    off."""
    monkeypatch.setattr(settings, "extract_decisions", False)
    sid = _make_session()
    conv_id = runner._save_extraction(
        session_id=sid,
        user_id="test",
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ended_at=None,
        extraction=_full_extraction(),
    )
    with Session(engine) as db:
        rows = list(
            db.exec(select(Decision).where(Decision.conversation_id == conv_id)).all()
        )
    assert rows == []


def test_runner_skips_calendar_when_toggle_off(monkeypatch):
    monkeypatch.setattr(settings, "extract_calendar_events", False)
    sid = _make_session()
    conv_id = runner._save_extraction(
        session_id=sid,
        user_id="test",
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ended_at=None,
        extraction=_full_extraction(),
    )
    with Session(engine) as db:
        rows = list(
            db.exec(select(CalendarEvent).where(CalendarEvent.conversation_id == conv_id)).all()
        )
    assert rows == []


def test_runner_skips_actions_when_toggle_off(monkeypatch):
    monkeypatch.setattr(settings, "extract_action_items", False)
    sid = _make_session()
    conv_id = runner._save_extraction(
        session_id=sid,
        user_id="test",
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ended_at=None,
        extraction=_full_extraction(),
    )
    with Session(engine) as db:
        rows = list(
            db.exec(select(ActionItem).where(ActionItem.conversation_id == conv_id)).all()
        )
    assert rows == []


def test_runner_skips_people_when_toggle_off(monkeypatch):
    monkeypatch.setattr(settings, "extract_people_mentioned", False)
    sid = _make_session()
    conv_id = runner._save_extraction(
        session_id=sid,
        user_id="test",
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ended_at=None,
        extraction=_full_extraction(),
    )
    with Session(engine) as db:
        rows = list(
            db.exec(select(PersonMention).where(PersonMention.conversation_id == conv_id)).all()
        )
    assert rows == []


def test_runner_skips_topics_when_toggle_off(monkeypatch):
    monkeypatch.setattr(settings, "extract_topics", False)
    sid = _make_session()
    conv_id = runner._save_extraction(
        session_id=sid,
        user_id="test",
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ended_at=None,
        extraction=_full_extraction(),
    )
    with Session(engine) as db:
        conv = db.get(Conversation, conv_id)
    assert conv.topics_json is None


def test_runner_storing_with_all_toggles_off_still_creates_conversation(monkeypatch):
    """The Conversation row itself (title, summary, quality) is always
    persisted — only the optional sections respect toggles. Otherwise a
    box with everything off would silently drop every capture."""
    for flag in (
        "extract_calendar_events",
        "extract_action_items",
        "extract_decisions",
        "extract_people_mentioned",
        "extract_topics",
    ):
        monkeypatch.setattr(settings, flag, False)
    sid = _make_session()
    conv_id = runner._save_extraction(
        session_id=sid,
        user_id="test",
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ended_at=None,
        extraction=_full_extraction(),
    )
    with Session(engine) as db:
        conv = db.get(Conversation, conv_id)
    assert conv is not None
    assert conv.title == "t"
    assert conv.quality_score == 0.7


def test_extraction_flags_helper_reads_settings(monkeypatch):
    monkeypatch.setattr(settings, "extract_calendar_events", False)
    monkeypatch.setattr(settings, "extract_decisions", True)
    flags = runner._extraction_flags()
    assert flags["calendar_events"] is False
    assert flags["decisions"] is True
    # Keys match what extract.build_messages expects.
    assert set(flags) == set(extract.DEFAULT_ENABLED)


# ──────────────────────────────────────────────────────────────────────────────
# UI: conversation page shows Decisions section when there are any
# ──────────────────────────────────────────────────────────────────────────────


def _login(client: TestClient, password: str) -> None:
    client.post("/login", data={"username": "test", "password": password})


def _seed_conversation_with_decision(*, decision_text: str = "Use Postgres") -> UUID:
    sid = uuid4()
    cid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="test",
                audio_path="/tmp/x.opus",
                codec="opus",
                started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                status=SessionStatus.done,
            )
        )
        db.flush()
        db.add(
            Conversation(
                id=cid,
                audio_session_id=sid,
                user_id="test",
                title="t",
                started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            )
        )
        db.flush()
        db.add(
            Decision(
                conversation_id=cid,
                text=decision_text,
                made_by="user",
                confidence=0.85,
            )
        )
        db.commit()
    return cid


def test_conversation_page_renders_decisions_section(
    client: TestClient, password: str
):
    _login(client, password)
    cid = _seed_conversation_with_decision(decision_text="Switch to Postgres")
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    assert "Decisions" in r.text
    assert "Switch to Postgres" in r.text
    assert "by user" in r.text
    assert "85%" in r.text  # confidence


def test_conversation_page_skips_decisions_section_when_none(
    client: TestClient, password: str
):
    """No decisions on this conversation → no section header. Avoids an
    empty 'Decisions (0)' showing up on conversations that don't have any
    (e.g. pre-Decision-feature captures)."""
    _login(client, password)
    sid = uuid4()
    cid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="test",
                audio_path="/tmp/x.opus",
                codec="opus",
                started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                status=SessionStatus.done,
            )
        )
        db.flush()
        db.add(
            Conversation(
                id=cid,
                audio_session_id=sid,
                user_id="test",
                title="t",
                started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            )
        )
        db.commit()
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    # The Decisions section header should be absent.
    assert "<h2>Decisions" not in r.text


def test_config_page_lists_extraction_toggles(client: TestClient, password: str):
    """The five toggle checkboxes show up in /config so the user can flip
    them without editing .env."""
    _login(client, password)
    r = client.get("/config")
    assert r.status_code == 200
    assert "Extraction categories" in r.text
    for key in (
        "OMILOG_EXTRACT_CALENDAR_EVENTS",
        "OMILOG_EXTRACT_ACTION_ITEMS",
        "OMILOG_EXTRACT_DECISIONS",
        "OMILOG_EXTRACT_PEOPLE_MENTIONED",
        "OMILOG_EXTRACT_TOPICS",
    ):
        assert key in r.text
