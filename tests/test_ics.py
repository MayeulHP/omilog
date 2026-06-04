"""ICS generation + endpoint tests."""

from datetime import datetime, timezone
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from omilog import ics
from omilog.config import settings
from omilog.db import engine
from omilog.models import (
    AudioSession,
    CalendarEvent,
    Conversation,
    SessionStatus,
)


# ──────────────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────────────

def test_escape_text_handles_specials():
    assert ics._escape_text("hello") == "hello"
    assert ics._escape_text("a;b") == "a\\;b"
    assert ics._escape_text("a,b") == "a\\,b"
    assert ics._escape_text("a\\b") == "a\\\\b"
    assert ics._escape_text("a\nb") == "a\\nb"
    assert ics._escape_text("a\r\nb") == "a\\nb"
    assert ics._escape_text(None) == ""


def test_fold_short_line_passthrough():
    assert ics._fold("short line") == "short line"


def test_fold_long_line_splits_at_75_octets():
    line = "X" * 200
    folded = ics._fold(line)
    # First line is 75 chars exactly, rest start with a single leading space.
    parts = folded.split("\r\n")
    assert len(parts[0]) == 75
    for cont in parts[1:]:
        assert cont.startswith(" ")
    # And the whole thing reassembles back to the original (drop folding bytes).
    rebuilt = parts[0] + "".join(p[1:] for p in parts[1:])
    assert rebuilt == line


def test_fold_respects_utf8_boundaries():
    # 'é' is 2 bytes in UTF-8. Build a long string of them so the folding
    # boundary would land in the middle of one if naive.
    line = "é" * 100
    folded = ics._fold(line)
    # Reassembly should reproduce the original — proves we didn't truncate a
    # multibyte char.
    parts = folded.split("\r\n")
    rebuilt = parts[0] + "".join(p[1:] for p in parts[1:])
    assert rebuilt == line


def test_fmt_utc_converts_aware_datetime():
    dt_paris = datetime(2026, 6, 15, 14, 0, tzinfo=ZoneInfo("Europe/Paris"))
    # June: Paris is UTC+2, so 14:00 → 12:00 UTC.
    assert ics._fmt_utc(dt_paris) == "20260615T120000Z"


def test_fmt_utc_treats_naive_as_utc():
    dt = datetime(2026, 6, 15, 14, 0)
    assert ics._fmt_utc(dt) == "20260615T140000Z"


# ──────────────────────────────────────────────────────────────────────────────
# Event / calendar builders
# ──────────────────────────────────────────────────────────────────────────────

def _make_event(**overrides) -> CalendarEvent:
    defaults = dict(
        id=uuid4(),
        conversation_id=uuid4(),
        title="Déjeuner avec Marie",
        starts_at=datetime(2026, 6, 15, 12, 30, tzinfo=timezone.utc),
        ends_at=datetime(2026, 6, 15, 14, 0, tzinfo=timezone.utc),
        location="Bastille",
        confidence=0.85,
        exported_to_ics=False,
    )
    defaults.update(overrides)
    return CalendarEvent(**defaults)


def test_build_vevent_minimum_shape():
    evt = _make_event()
    lines = ics.build_vevent(
        evt, conversation_id="abc", conversation_title="Brunch chat"
    )
    text = "\r\n".join(lines)
    assert lines[0] == "BEGIN:VEVENT"
    assert lines[-1] == "END:VEVENT"
    assert f"UID:{evt.id}@omilog" in text
    assert "DTSTART:20260615T123000Z" in text
    assert "DTEND:20260615T140000Z" in text
    assert "SUMMARY:Déjeuner avec Marie" in text
    assert "LOCATION:Bastille" in text
    assert "confidence: 85%" in text


def test_build_vevent_defaults_end_to_one_hour():
    evt = _make_event(ends_at=None)
    lines = ics.build_vevent(evt)
    text = "\r\n".join(lines)
    assert "DTSTART:20260615T123000Z" in text
    # +1h from 12:30
    assert "DTEND:20260615T133000Z" in text


def test_build_vevent_raises_without_start():
    evt = _make_event(starts_at=None)
    with pytest.raises(ValueError, match="no starts_at"):
        ics.build_vevent(evt)


def test_build_vevent_escapes_summary():
    evt = _make_event(title="Lunch; with Marie, the boss")
    text = "\r\n".join(ics.build_vevent(evt))
    assert "SUMMARY:Lunch\\; with Marie\\, the boss" in text


def test_build_vcalendar_wraps_and_uses_crlf():
    evt = _make_event()
    out = ics.build_vcalendar([(evt, "c1", "Conv")])
    assert out.startswith("BEGIN:VCALENDAR\r\n")
    assert out.endswith("END:VCALENDAR\r\n")
    assert "VERSION:2.0" in out
    assert "PRODID:-//omilog//EN" in out
    assert "CALSCALE:GREGORIAN" in out


def test_build_vcalendar_skips_eventless_starts():
    skipped = _make_event(starts_at=None)
    kept = _make_event(title="With start")
    out = ics.build_vcalendar(
        [(skipped, None, None), (kept, None, None)]
    )
    assert "SUMMARY:With start" in out
    # Only one VEVENT block expected.
    assert out.count("BEGIN:VEVENT") == 1


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint: /calendar.ics
# ──────────────────────────────────────────────────────────────────────────────

def _seed_conversation_with_event(
    *,
    user: str = "test",
    confidence: float = 0.85,
    starts_at: datetime | None = None,
) -> tuple[UUID, UUID]:
    """Returns (conversation_id, event_id)."""
    sid = uuid4()
    cid = uuid4()
    eid = uuid4()
    starts_at = starts_at or datetime(2099, 6, 15, 12, 30, tzinfo=timezone.utc)
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id=user,
                audio_path="/tmp/x.opus",
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
                title="Test conv",
                summary="Y",
                started_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
            )
        )
        db.flush()
        db.add(
            CalendarEvent(
                id=eid,
                conversation_id=cid,
                title="Test event",
                starts_at=starts_at,
                ends_at=starts_at,
                location="Somewhere",
                confidence=confidence,
            )
        )
        db.commit()
    return cid, eid


def test_calendar_feed_403_when_token_missing(client: TestClient):
    r = client.get("/calendar.ics")
    assert r.status_code == 403


def test_calendar_feed_403_when_disabled(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "ics_feed_token", "", raising=False)
    r = client.get("/calendar.ics?token=anything")
    assert r.status_code == 403


def test_calendar_feed_403_on_wrong_token(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "ics_feed_token", "correct-token", raising=False)
    r = client.get("/calendar.ics?token=wrong")
    assert r.status_code == 403


def test_calendar_feed_returns_events(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "ics_feed_token", "good-token", raising=False)
    _seed_conversation_with_event()
    r = client.get("/calendar.ics?token=good-token")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/calendar")
    assert "BEGIN:VCALENDAR" in r.text
    assert "SUMMARY:Test event" in r.text


def test_calendar_feed_min_confidence_filter(client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "ics_feed_token", "tok", raising=False)
    _seed_conversation_with_event(confidence=0.3)  # below default 0.5
    r = client.get("/calendar.ics?token=tok")
    assert r.status_code == 200
    assert "SUMMARY:Test event" not in r.text
    # …but show up when explicitly lowering threshold
    r2 = client.get("/calendar.ics?token=tok&min_confidence=0.1")
    assert "SUMMARY:Test event" in r2.text


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint: /events/{id}/download.ics
# ──────────────────────────────────────────────────────────────────────────────

def test_event_download_returns_calendar_and_marks_exported(
    client: TestClient, auth_token: str
):
    _, eid = _seed_conversation_with_event()
    r = client.get(
        f"/events/{eid}/download.ics",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/calendar")
    assert "attachment" in r.headers["content-disposition"]
    assert "BEGIN:VCALENDAR" in r.text
    assert "SUMMARY:Test event" in r.text

    with Session(engine) as db:
        evt = db.get(CalendarEvent, eid)
        assert evt is not None
        assert evt.exported_to_ics is True


def test_event_download_requires_auth(client: TestClient):
    _, eid = _seed_conversation_with_event()
    assert client.get(f"/events/{eid}/download.ics").status_code == 401


def test_event_download_404_for_other_user(client: TestClient, auth_token: str):
    _, eid = _seed_conversation_with_event(user="not-me")
    r = client.get(
        f"/events/{eid}/download.ics",
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert r.status_code == 404
