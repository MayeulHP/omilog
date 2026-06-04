"""Conversation quality scoring: LLM-judged score + user override + UI filter.

The score itself is produced by the LLM during extraction (the prompt
asks for it with anchored ranges). We test:
- The parser pulls the score out and clamps it.
- The runner stores it, with a small penalty when extraction was repaired.
- The list filter buckets conversations correctly by effective_quality.
- The override endpoint changes which bucket a conversation falls into.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session

from omilog.db import engine
from omilog.models import (
    AudioSession,
    Conversation,
    SessionStatus,
)
from omilog.pipeline import extract
from omilog.web import routes


# ──────────────────────────────────────────────────────────────────────────────
# Parser pulls the score out
# ──────────────────────────────────────────────────────────────────────────────


def _llm_json(**overrides) -> str:
    """Build a synthetic LLM response with the new fields filled in."""
    body = {
        "title": "Test",
        "summary": "A short conversation.",
        "quality_score": 0.6,
        "quality_reasoning": "Brief but had a concrete plan.",
        "calendar_events": [],
        "action_items": [],
        "people_mentioned": [],
        "topics": [],
    }
    body.update(overrides)
    return json.dumps(body)


def test_parse_extracts_quality_score():
    e = extract.parse(_llm_json(quality_score=0.42))
    assert e.quality_score == 0.42


def test_parse_clamps_out_of_range_quality_score():
    """LLMs sometimes return 1.2 or -0.1. Clamp rather than reject."""
    assert extract.parse(_llm_json(quality_score=1.5)).quality_score == 1.0
    assert extract.parse(_llm_json(quality_score=-0.5)).quality_score == 0.0


def test_parse_accepts_string_quality_score():
    """Some models stringify numbers in JSON even when asked not to."""
    e = extract.parse(_llm_json(quality_score="0.75"))
    assert e.quality_score == 0.75


def test_parse_missing_quality_score_returns_none():
    """An older prompt override that doesn't ask for the score leaves the
    field missing — caller must fall back to a default."""
    e = extract.parse(_llm_json(quality_score=None))
    assert e.quality_score is None


def test_parse_garbage_quality_score_returns_none():
    """Defensive: a non-numeric value shouldn't crash the whole parse."""
    e = extract.parse(_llm_json(quality_score="not a number"))
    assert e.quality_score is None


def test_parse_extracts_quality_reasoning():
    e = extract.parse(_llm_json(quality_reasoning="Single-speaker rambling, likely TV."))
    assert e.quality_reasoning == "Single-speaker rambling, likely TV."


def test_parse_empty_quality_reasoning_becomes_none():
    e = extract.parse(_llm_json(quality_reasoning=""))
    assert e.quality_reasoning is None


def test_prompt_includes_quality_anchors():
    """The LLM needs the anchor language to score consistently. If someone
    edits the prompt and removes the anchors, this test catches it."""
    prompt = extract.render_default_system_prompt("")
    assert "quality_score" in prompt
    assert "0.0:" in prompt or "0.0 " in prompt  # noise anchor
    assert "1.0:" in prompt or "1.0 " in prompt  # substantive anchor
    assert "Be conservative" in prompt


# ──────────────────────────────────────────────────────────────────────────────
# Runner storage: score is persisted, repair is penalised
# ──────────────────────────────────────────────────────────────────────────────


def _make_session(user: str = "test") -> UUID:
    """A done AudioSession suitable for hanging a Conversation off."""
    sid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id=user,
                audio_path="/tmp/test.opus",
                codec="opus",
                started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                status=SessionStatus.done,
            )
        )
        db.commit()
    return sid


def test_save_extraction_stores_quality_score():
    sid = _make_session()
    extraction = extract.Extraction(
        title="t",
        summary="s",
        quality_score=0.8,
        quality_reasoning="Decisions made.",
    )
    from omilog.pipeline import runner

    conv_id = runner._save_extraction(
        session_id=sid,
        user_id="test",
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ended_at=None,
        extraction=extraction,
    )
    with Session(engine) as db:
        conv = db.get(Conversation, conv_id)
    assert conv.quality_score == 0.8
    assert conv.quality_reasoning == "Decisions made."


def test_save_extraction_defaults_to_half_when_score_missing():
    """An old prompt override that doesn't return quality_score shouldn't
    silently bury the conversation as noise — fall back to mid-range."""
    sid = _make_session()
    extraction = extract.Extraction(
        title="t", summary="s", quality_score=None
    )
    from omilog.pipeline import runner

    conv_id = runner._save_extraction(
        session_id=sid,
        user_id="test",
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ended_at=None,
        extraction=extraction,
    )
    with Session(engine) as db:
        conv = db.get(Conversation, conv_id)
    assert conv.quality_score == 0.5


def test_save_extraction_penalises_repaired():
    """Truncated extractions are partial — discount the score a bit since
    we don't fully trust the self-assessment."""
    sid = _make_session()
    extraction = extract.Extraction(
        title="t",
        summary="s",
        quality_score=0.7,
        was_repaired=True,
    )
    from omilog.pipeline import runner

    conv_id = runner._save_extraction(
        session_id=sid,
        user_id="test",
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ended_at=None,
        extraction=extraction,
    )
    with Session(engine) as db:
        conv = db.get(Conversation, conv_id)
    # 0.7 - 0.1 penalty
    assert abs(conv.quality_score - 0.6) < 1e-9


def test_save_extraction_penalty_does_not_go_below_zero():
    sid = _make_session()
    extraction = extract.Extraction(
        title="t", summary="s", quality_score=0.05, was_repaired=True
    )
    from omilog.pipeline import runner

    conv_id = runner._save_extraction(
        session_id=sid,
        user_id="test",
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ended_at=None,
        extraction=extraction,
    )
    with Session(engine) as db:
        conv = db.get(Conversation, conv_id)
    assert conv.quality_score == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Bucket classifier
# ──────────────────────────────────────────────────────────────────────────────


def test_bucket_classifier_anchors():
    """Boundaries must match the prompt's anchor anchors so the LLM and
    the UI agree on what 'noise' / 'substantive' mean."""
    assert routes._quality_bucket(0.0) == "noise"
    assert routes._quality_bucket(0.29) == "noise"
    assert routes._quality_bucket(0.3) == "normal"
    assert routes._quality_bucket(0.5) == "normal"
    assert routes._quality_bucket(0.69) == "normal"
    assert routes._quality_bucket(0.7) == "substantive"
    assert routes._quality_bucket(1.0) == "substantive"


def test_effective_quality_prefers_override():
    """When the user has explicitly set a score, that wins over the LLM's."""
    conv = Conversation(
        audio_session_id=uuid4(),
        user_id="test",
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        quality_score=0.9,        # LLM said substantive
        quality_override=0.0,     # user overrode to noise
    )
    assert routes._effective_quality(conv) == 0.0


def test_effective_quality_falls_back_to_llm_score_when_no_override():
    conv = Conversation(
        audio_session_id=uuid4(),
        user_id="test",
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        quality_score=0.42,
        quality_override=None,
    )
    assert routes._effective_quality(conv) == 0.42


# ──────────────────────────────────────────────────────────────────────────────
# List filter + override endpoint
# ──────────────────────────────────────────────────────────────────────────────


def _login(client: TestClient, password: str) -> None:
    client.post("/login", data={"username": "test", "password": password})


def _seed_conv(*, quality: float, override: float | None = None, title: str = "x") -> UUID:
    """Insert a minimal Conversation row with a known quality score."""
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
                title=title,
                summary="...",
                started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                quality_score=quality,
                quality_override=override,
            )
        )
        db.commit()
    return cid


def test_index_default_filter_hides_noise(client: TestClient, password: str):
    _login(client, password)
    _seed_conv(quality=0.1, title="noise-row")
    _seed_conv(quality=0.5, title="normal-row")
    r = client.get("/")
    assert r.status_code == 200
    assert "normal-row" in r.text
    assert "noise-row" not in r.text
    # The hidden-count hint should mention the buried row.
    assert "hidden" in r.text


def test_index_substantive_filter_hides_normal(client: TestClient, password: str):
    _login(client, password)
    _seed_conv(quality=0.5, title="normal-row")
    _seed_conv(quality=0.8, title="substantive-row")
    r = client.get("/?q=substantive")
    assert r.status_code == 200
    assert "substantive-row" in r.text
    assert "normal-row" not in r.text


def test_index_all_filter_shows_everything(client: TestClient, password: str):
    _login(client, password)
    _seed_conv(quality=0.05, title="noise-row")
    _seed_conv(quality=0.5, title="normal-row")
    _seed_conv(quality=0.9, title="substantive-row")
    r = client.get("/?q=all")
    assert r.status_code == 200
    assert "noise-row" in r.text
    assert "normal-row" in r.text
    assert "substantive-row" in r.text


def test_index_show_hidden_overrides_filter(client: TestClient, password: str):
    """When the user clicks 'show anyway' the hidden noise comes back even
    if the quality filter is still 'normal'."""
    _login(client, password)
    _seed_conv(quality=0.1, title="hidden-noise")
    r = client.get("/?q=normal&show_hidden=1")
    assert "hidden-noise" in r.text


def test_index_respects_override(client: TestClient, password: str):
    """A low-LLM-score conversation that the user marked substantive should
    appear at the substantive filter, NOT be hidden as noise."""
    _login(client, password)
    _seed_conv(quality=0.05, override=1.0, title="user-rescued")
    r = client.get("/?q=substantive")
    assert r.status_code == 200
    assert "user-rescued" in r.text


def test_index_respects_override_to_hide(client: TestClient, password: str):
    """And the inverse: a substantive-rated conversation the user marked
    noise should disappear from the default filter."""
    _login(client, password)
    _seed_conv(quality=0.9, override=0.0, title="user-buried")
    r = client.get("/")  # default = normal+
    assert r.status_code == 200
    assert "user-buried" not in r.text


def test_rate_sets_override_to_noise(client: TestClient, password: str):
    _login(client, password)
    cid = _seed_conv(quality=0.6)
    r = client.post(f"/conversations/{cid}/rate", data={"rating": "noise"})
    assert r.status_code == 200
    assert r.headers.get("HX-Refresh") == "true"
    with Session(engine) as db:
        conv = db.get(Conversation, cid)
    assert conv.quality_override == 0.0


def test_rate_sets_override_to_substantive(client: TestClient, password: str):
    _login(client, password)
    cid = _seed_conv(quality=0.4)
    client.post(f"/conversations/{cid}/rate", data={"rating": "substantive"})
    with Session(engine) as db:
        conv = db.get(Conversation, cid)
    assert conv.quality_override == 1.0


def test_rate_clear_removes_override(client: TestClient, password: str):
    _login(client, password)
    cid = _seed_conv(quality=0.4, override=1.0)
    client.post(f"/conversations/{cid}/rate", data={"rating": "clear"})
    with Session(engine) as db:
        conv = db.get(Conversation, cid)
    assert conv.quality_override is None


def test_rate_rejects_invalid_value(client: TestClient, password: str):
    _login(client, password)
    cid = _seed_conv(quality=0.5)
    r = client.post(f"/conversations/{cid}/rate", data={"rating": "garbage"})
    assert r.status_code == 400


def test_rate_404_for_wrong_user(client: TestClient, password: str):
    _login(client, password)
    # Manually create a conversation under another user.
    sid = uuid4()
    cid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="not-test",
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
                user_id="not-test",
                title="t",
                started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                quality_score=0.5,
            )
        )
        db.commit()
    r = client.post(f"/conversations/{cid}/rate", data={"rating": "noise"})
    assert r.status_code == 404


def test_conversation_page_shows_score_and_reasoning(
    client: TestClient, password: str
):
    _login(client, password)
    cid = _seed_conv(quality=0.8, title="x")
    with Session(engine) as db:
        conv = db.get(Conversation, cid)
        conv.quality_reasoning = "Concrete decisions about Friday's plan."
        db.add(conv)
        db.commit()
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    assert "Substantive" in r.text
    assert "80%" in r.text
    assert "Concrete decisions" in r.text


def test_conversation_page_shows_override_marker(
    client: TestClient, password: str
):
    _login(client, password)
    cid = _seed_conv(quality=0.4, override=1.0)
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    # Manual-override marker (the ✋ emoji is the visual cue)
    assert "✋" in r.text or "Manual override" in r.text
