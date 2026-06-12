"""Daily-summary pipeline + UI tests.

The actual LLM call is patched in via monkeypatch on
``omilog.pipeline.daily.chat_json`` so we never hit a real backend.
Everything else (date-bounds math, quality filtering, conversation
formatting, caching, UI rendering) is exercised against the real
SQLite + FastAPI test client.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from omilog.config import settings
from omilog.db import engine
from omilog.models import (
    ActionItem,
    AudioSession,
    CalendarEvent,
    Conversation,
    DailySummary,
    PersonMention,
    SessionStatus,
)
from omilog.pipeline import daily as daily_mod


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────────


def _seed_conv(
    *,
    user: str = "test",
    started_at: datetime,
    title: str,
    summary: str = "",
    quality: float = 0.5,
    override: float | None = None,
) -> UUID:
    """Insert a Conversation with the given quality.

    SQLite stores tz-aware datetimes as ISO strings, so range-WHERE clauses
    end up doing LEXICOGRAPHIC comparison rather than true datetime maths.
    To match production (where AudioSession.started_at and
    Conversation.started_at are both UTC defaults), we normalise to UTC
    here even if the test author handed us Paris time.
    """
    if started_at.tzinfo is not None:
        started_at = started_at.astimezone(ZoneInfo("UTC"))
    sid = uuid4()
    cid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id=user,
                audio_path="/tmp/x.opus",
                codec="opus",
                started_at=started_at,
                status=SessionStatus.done,
            )
        )
        db.flush()
        db.add(
            Conversation(
                id=cid,
                audio_session_id=sid,
                user_id=user,
                title=title,
                summary=summary,
                started_at=started_at,
                quality_score=quality,
                quality_override=override,
            )
        )
        db.commit()
    return cid


@dataclass
class _FakeChat:
    text: str


def _patch_chat(monkeypatch, response_text: str) -> dict[str, list]:
    """Replace daily.chat_json with a stub that records its call args and
    returns ``response_text``. Returns a 'recorder' dict the test can read."""
    calls: dict[str, list] = {"args": []}

    async def fake(**kwargs):
        calls["args"].append(kwargs)
        return _FakeChat(text=response_text)

    monkeypatch.setattr(daily_mod, "chat_json", fake)
    return calls


def _login(client: TestClient, password: str) -> None:
    client.post("/login", data={"username": "test", "password": password})


# ──────────────────────────────────────────────────────────────────────────────
# Date-bounds math (the bit that's easy to get wrong with timezones)
# ──────────────────────────────────────────────────────────────────────────────


def test_utc_day_bounds_paris_summer():
    """June Paris: UTC+2. Local 'June 15' = 22:00 UTC June 14 → 22:00 UTC June 15."""
    start, end = daily_mod._utc_day_bounds(date(2026, 6, 15), "Europe/Paris")
    assert start == datetime(2026, 6, 14, 22, 0, tzinfo=ZoneInfo("UTC"))
    assert end == datetime(2026, 6, 15, 22, 0, tzinfo=ZoneInfo("UTC"))


def test_utc_day_bounds_paris_winter():
    """January Paris: UTC+1."""
    start, end = daily_mod._utc_day_bounds(date(2026, 1, 15), "Europe/Paris")
    assert start == datetime(2026, 1, 14, 23, 0, tzinfo=ZoneInfo("UTC"))
    assert end == datetime(2026, 1, 15, 23, 0, tzinfo=ZoneInfo("UTC"))


def test_utc_day_bounds_invalid_tz_falls_back_to_utc():
    start, end = daily_mod._utc_day_bounds(date(2026, 6, 15), "Not/AReal_Zone")
    assert start == datetime(2026, 6, 15, 0, 0, tzinfo=ZoneInfo("UTC"))
    assert end == datetime(2026, 6, 16, 0, 0, tzinfo=ZoneInfo("UTC"))


# ──────────────────────────────────────────────────────────────────────────────
# Quality filtering — eligibility for the summary
# ──────────────────────────────────────────────────────────────────────────────


def test_fetch_eligible_includes_above_threshold():
    """Two conversations on the same day, one above the threshold, one below."""
    d = date(2026, 6, 15)
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    _seed_conv(started_at=when, title="substantive", quality=0.8)
    _seed_conv(
        started_at=when + timedelta(hours=1), title="noise", quality=0.1
    )
    eligible = daily_mod._fetch_eligible("test", d, threshold=0.3)
    titles = [c.title for c in eligible]
    assert "substantive" in titles
    assert "noise" not in titles


def test_fetch_eligible_respects_override_over_score():
    """User override wins over LLM score for eligibility filtering."""
    d = date(2026, 6, 15)
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    # LLM said substantive, user overrode to noise → excluded.
    _seed_conv(
        started_at=when,
        title="user-hid",
        quality=0.9,
        override=0.0,
    )
    # LLM said noise, user overrode to substantive → included.
    _seed_conv(
        started_at=when + timedelta(minutes=10),
        title="user-rescued",
        quality=0.1,
        override=1.0,
    )
    eligible = daily_mod._fetch_eligible("test", d, threshold=0.3)
    titles = [c.title for c in eligible]
    assert "user-hid" not in titles
    assert "user-rescued" in titles


def test_fetch_eligible_excludes_other_days():
    """A late-night Paris conversation must not leak into the wrong day."""
    d = date(2026, 6, 15)
    # 23:30 local on June 14 — that's June 14, not 15.
    yesterday_late = datetime(
        2026, 6, 14, 23, 30, tzinfo=ZoneInfo("Europe/Paris")
    )
    _seed_conv(started_at=yesterday_late, title="late-yesterday", quality=0.8)
    eligible = daily_mod._fetch_eligible("test", d, threshold=0.3)
    assert eligible == []


def test_fetch_eligible_is_per_user():
    d = date(2026, 6, 15)
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    _seed_conv(user="alice", started_at=when, title="alice-conv", quality=0.8)
    _seed_conv(user="bob", started_at=when, title="bob-conv", quality=0.8)
    assert len(daily_mod._fetch_eligible("alice", d, threshold=0.3)) == 1
    assert len(daily_mod._fetch_eligible("bob", d, threshold=0.3)) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Context rendering — what the LLM sees
# ──────────────────────────────────────────────────────────────────────────────


def test_build_context_includes_titles_and_summaries():
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    cid = _seed_conv(
        started_at=when, title="Lunch with Marie", summary="Discussed Friday plans.",
        quality=0.8,
    )
    with Session(engine) as db:
        conv = db.get(Conversation, cid)
    rendered = daily_mod._build_context([conv], date(2026, 6, 15))
    assert "Lunch with Marie" in rendered
    assert "Discussed Friday plans" in rendered
    assert "14:00" in rendered
    assert "2026-06-15" in rendered


def test_build_context_includes_events_and_actions():
    """The LLM gets enough hooks to write a richer narrative — not just
    titles + summaries but the extracted highlights too."""
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    cid = _seed_conv(
        started_at=when, title="Project review", summary="...", quality=0.8,
    )
    with Session(engine) as db:
        db.add(
            CalendarEvent(
                conversation_id=cid,
                title="Sync with Paul",
                starts_at=datetime(2026, 6, 17, 14, 0, tzinfo=timezone.utc),
            )
        )
        db.add(
            ActionItem(
                conversation_id=cid,
                text="send the deck",
                owner="user",
            )
        )
        db.add(
            PersonMention(
                conversation_id=cid,
                name="Marie",
                context="brought up the deadline",
            )
        )
        db.commit()
        conv = db.get(Conversation, cid)

    rendered = daily_mod._build_context([conv], date(2026, 6, 15))
    assert "Sync with Paul" in rendered
    assert "send the deck" in rendered
    assert "Marie" in rendered


# ──────────────────────────────────────────────────────────────────────────────
# generate() end-to-end with a patched LLM
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_calls_llm_and_returns_narrative(monkeypatch):
    """Eligible conversations + valid LLM JSON → narrative populated."""
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    _seed_conv(started_at=when, title="x", summary="...", quality=0.8)
    monkeypatch.setattr(settings, "llm_base_url", "http://fake-llm.test:1234/v1")
    _patch_chat(monkeypatch, '{"narrative": "Lunch with Marie about Friday plans."}')

    result = await daily_mod.generate("test", date(2026, 6, 15))
    assert result.narrative == "Lunch with Marie about Friday plans."
    assert len(result.conversation_ids) == 1


@pytest.mark.asyncio
async def test_generate_returns_none_when_no_eligible(monkeypatch):
    """Sparse day with only noise conversations — no LLM call, narrative=None."""
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    _seed_conv(started_at=when, title="just-noise", quality=0.1)
    monkeypatch.setattr(settings, "llm_base_url", "http://fake-llm.test:1234/v1")
    recorder = _patch_chat(monkeypatch, '{"narrative": "should not be called"}')

    result = await daily_mod.generate("test", date(2026, 6, 15))
    assert result.narrative is None
    assert result.conversation_ids == []
    assert recorder["args"] == []  # no LLM call


@pytest.mark.asyncio
async def test_generate_threshold_changes_what_feeds_summary(monkeypatch):
    """A 0.5 conversation should be included at threshold=0.3 but excluded
    at threshold=0.7. This is how the UI's threshold input affects output."""
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    _seed_conv(started_at=when, title="mid", quality=0.5)
    monkeypatch.setattr(settings, "llm_base_url", "http://fake-llm.test:1234/v1")
    recorder = _patch_chat(monkeypatch, '{"narrative": "ok"}')

    low = await daily_mod.generate("test", date(2026, 6, 15), quality_threshold=0.3)
    assert low.narrative == "ok"
    assert len(low.conversation_ids) == 1

    recorder["args"].clear()
    high = await daily_mod.generate("test", date(2026, 6, 15), quality_threshold=0.7)
    assert high.narrative is None  # no eligible at this threshold
    assert recorder["args"] == []


@pytest.mark.asyncio
async def test_generate_handles_code_fenced_response(monkeypatch):
    """Models love to wrap JSON in ```json fences despite instructions. The
    parser strips them."""
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    _seed_conv(started_at=when, title="x", quality=0.8)
    monkeypatch.setattr(settings, "llm_base_url", "http://fake-llm.test:1234/v1")
    _patch_chat(
        monkeypatch,
        '```json\n{"narrative": "a day"}\n```',
    )
    result = await daily_mod.generate("test", date(2026, 6, 15))
    assert result.narrative == "a day"


@pytest.mark.asyncio
async def test_generate_handles_think_block(monkeypatch):
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    _seed_conv(started_at=when, title="x", quality=0.8)
    monkeypatch.setattr(settings, "llm_base_url", "http://fake-llm.test:1234/v1")
    _patch_chat(
        monkeypatch,
        "<think>let me see</think>{\"narrative\": \"clean output\"}",
    )
    result = await daily_mod.generate("test", date(2026, 6, 15))
    assert result.narrative == "clean output"


@pytest.mark.asyncio
async def test_generate_rejects_unclosed_think_block(monkeypatch):
    """Truncation mid-reasoning (no </think> ever emitted): the output is all
    scratchpad, including a draft narrative that must NOT be stored."""
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    _seed_conv(started_at=when, title="x", quality=0.8)
    monkeypatch.setattr(settings, "llm_base_url", "http://fake-llm.test:1234/v1")
    _patch_chat(
        monkeypatch,
        '<think>Drafting: {"narrative": "DRAFT scratchpad"} hmm, but actually',
    )
    with pytest.raises(ValueError):
        await daily_mod.generate("test", date(2026, 6, 15))


@pytest.mark.asyncio
async def test_generate_passes_disable_thinking_from_settings(monkeypatch):
    """generate() must plumb settings.llm_disable_thinking through to
    chat_json so the per-request thinking override actually fires."""
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    _seed_conv(started_at=when, title="x", quality=0.8)
    monkeypatch.setattr(settings, "llm_base_url", "http://fake-llm.test:1234/v1")
    monkeypatch.setattr(settings, "llm_disable_thinking", True)
    recorder = _patch_chat(monkeypatch, '{"narrative": "ok"}')
    await daily_mod.generate("test", date(2026, 6, 15))
    assert recorder["args"][0]["disable_thinking"] is True


@pytest.mark.asyncio
async def test_generate_raises_on_malformed_llm_output(monkeypatch):
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    _seed_conv(started_at=when, title="x", quality=0.8)
    monkeypatch.setattr(settings, "llm_base_url", "http://fake-llm.test:1234/v1")
    _patch_chat(monkeypatch, "not json at all, just prose.")
    with pytest.raises(ValueError, match="LLM output|narrative"):
        await daily_mod.generate("test", date(2026, 6, 15))


# ──────────────────────────────────────────────────────────────────────────────
# Store / get_cached / drift detection
# ──────────────────────────────────────────────────────────────────────────────


def test_store_persists_summary():
    cid = uuid4()
    result = daily_mod.DailyResult(
        narrative="a day", conversation_ids=[cid], quality_threshold=0.3
    )
    row = daily_mod.store("test", date(2026, 6, 15), result)
    assert row.narrative == "a day"
    assert row.conversation_count == 1
    assert json.loads(row.conversation_ids_json) == [str(cid)]


def test_store_replaces_existing():
    """Regenerate overwrites — no accidental double rows for the same day."""
    cid1 = uuid4()
    cid2 = uuid4()
    daily_mod.store(
        "test",
        date(2026, 6, 15),
        daily_mod.DailyResult(
            narrative="v1", conversation_ids=[cid1], quality_threshold=0.3
        ),
    )
    daily_mod.store(
        "test",
        date(2026, 6, 15),
        daily_mod.DailyResult(
            narrative="v2", conversation_ids=[cid2], quality_threshold=0.5
        ),
    )
    cached = daily_mod.get_cached("test", date(2026, 6, 15))
    assert cached.narrative == "v2"
    assert json.loads(cached.conversation_ids_json) == [str(cid2)]


def test_get_cached_returns_none_when_absent():
    assert daily_mod.get_cached("test", date(2099, 12, 31)) is None


def test_conversation_ids_for_decodes():
    cid = uuid4()
    row = DailySummary(
        user_id="test",
        date="2026-06-15",
        narrative="",
        conversation_ids_json=json.dumps([str(cid)]),
        conversation_count=1,
        quality_threshold=0.3,
    )
    assert daily_mod.conversation_ids_for(row) == [cid]


def test_conversation_ids_for_handles_malformed():
    """Tolerant: malformed JSON or wrong types don't crash the page."""
    row = DailySummary(
        user_id="test",
        date="2026-06-15",
        narrative="",
        conversation_ids_json="not valid json",
        conversation_count=0,
        quality_threshold=0.3,
    )
    assert daily_mod.conversation_ids_for(row) == []


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────


def test_daily_root_redirects_to_today(client: TestClient, password: str):
    _login(client, password)
    r = client.get("/daily", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"].startswith("/daily/")


def test_daily_show_empty_day(client: TestClient, password: str):
    _login(client, password)
    r = client.get("/daily/2026-06-15")
    assert r.status_code == 200
    # No summary AND no eligible conversations: empty-state copy.
    assert "No summary yet" in r.text or "no substantive conversations" in r.text.lower()


def test_daily_show_invalid_date_404(client: TestClient, password: str):
    _login(client, password)
    r = client.get("/daily/not-a-date")
    assert r.status_code == 404


def test_daily_show_renders_cached_summary(client: TestClient, password: str):
    _login(client, password)
    cid = uuid4()
    daily_mod.store(
        "test",
        date(2026, 6, 15),
        daily_mod.DailyResult(
            narrative="A productive afternoon with Marie.",
            conversation_ids=[cid],
            quality_threshold=0.3,
        ),
    )
    r = client.get("/daily/2026-06-15")
    assert r.status_code == 200
    assert "A productive afternoon with Marie." in r.text


def test_daily_show_lists_eligible_conversations(
    client: TestClient, password: str
):
    _login(client, password)
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    _seed_conv(
        started_at=when, title="meeting-with-marie", summary="x", quality=0.8
    )
    r = client.get("/daily/2026-06-15")
    assert r.status_code == 200
    assert "meeting-with-marie" in r.text


def test_daily_show_flags_drift_when_eligible_set_changed(
    client: TestClient, password: str
):
    """Cache has 1 conv; threshold or new captures pulled in another → UI
    should hint that regenerating would change the summary."""
    _login(client, password)
    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    # Cache was built from a single conv.
    cached_cid = uuid4()
    daily_mod.store(
        "test",
        date(2026, 6, 15),
        daily_mod.DailyResult(
            narrative="just one",
            conversation_ids=[cached_cid],
            quality_threshold=0.3,
        ),
    )
    # But the DB has a different, real conv now.
    _seed_conv(started_at=when, title="fresh-conv", quality=0.8)
    r = client.get("/daily/2026-06-15")
    assert r.status_code == 200
    assert "changed" in r.text.lower() or "drift" in r.text.lower() or "regenerate" in r.text.lower()


@pytest.mark.asyncio
async def test_daily_generate_endpoint_stores_summary(monkeypatch):
    """Full UI roundtrip: POST /daily/.../generate → cache row exists with
    the LLM's narrative."""
    # Need to use TestClient here, not pytest-asyncio, because the FastAPI
    # client wraps the async route in a sync interface.
    from fastapi.testclient import TestClient as TC

    from omilog.main import app

    monkeypatch.setattr(settings, "llm_base_url", "http://fake-llm.test:1234/v1")
    _patch_chat(monkeypatch, '{"narrative": "Quiet productive day."}')

    when = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    _seed_conv(started_at=when, title="x", quality=0.8)

    # Need a fresh password fixture-style. We'll log in via the same helper.
    from tests.conftest import TEST_PASSWORD

    with TC(app) as c:
        c.post("/login", data={"username": "test", "password": TEST_PASSWORD})
        r = c.post("/daily/2026-06-15/generate", follow_redirects=False)
        assert r.status_code == 303

    cached = daily_mod.get_cached("test", date(2026, 6, 15))
    assert cached is not None
    assert cached.narrative == "Quiet productive day."


def test_daily_generate_invalid_date_404(client: TestClient, password: str):
    _login(client, password)
    r = client.post("/daily/not-a-date/generate")
    assert r.status_code == 404
