"""Cross-conversation speaker linking tests.

Two layers:
- Pure-data unit tests for the matching / running-average / cosine logic.
  These don't touch the DB or any backend.
- DB-integration tests for ``_link_speakers_to_segments`` and the UI
  endpoints. The DB uses the shared test SQLite from conftest.

We don't hit sherpa-onnx here — embeddings are passed in directly as
``list[float]``. The runner only invokes ``compute_speaker_embeddings``
when diarization is available, but the linking step itself takes
already-computed embeddings, so it's testable without any of that.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from omilog.config import settings
from omilog.db import engine
from omilog.models import (
    AudioSession,
    Conversation,
    SessionStatus,
    Speaker,
    Transcript,
)
from omilog.pipeline import runner


# ──────────────────────────────────────────────────────────────────────────────
# Pure data: cosine similarity + serialization + running average
# ──────────────────────────────────────────────────────────────────────────────


def test_cosine_identical_vectors_is_one():
    v = [1.0, 2.0, 3.0]
    assert runner._cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal_vectors_is_zero():
    assert runner._cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)


def test_cosine_zero_norm_returns_zero():
    """Degenerate input shouldn't NaN — the matching loop already filters
    these out, but defense in depth."""
    assert runner._cosine_similarity([0, 0, 0], [1, 1, 1]) == 0.0


def test_cosine_known_value():
    # 30 degrees: cos = 0.866...
    a = [1.0, 0.0]
    b = [math.cos(math.radians(30)), math.sin(math.radians(30))]
    assert runner._cosine_similarity(a, b) == pytest.approx(0.866, abs=1e-3)


def test_cosine_mismatched_length_returns_zero():
    """We never expect this in practice (all embeddings come from the same
    extractor) but the function shouldn't crash if someone hands it a bad
    vector — better a missed match than a 500."""
    assert runner._cosine_similarity([1, 2], [1, 2, 3]) == 0.0


def test_emb_bytes_roundtrip_preserves_values():
    original = [0.1, -0.2, 1e-3, 0.999, -1.5]
    b = runner._emb_to_bytes(original)
    roundtripped = runner._emb_from_bytes(b)
    assert len(roundtripped) == len(original)
    for o, r in zip(original, roundtripped):
        # float32 has ~7 digits of precision; the test values are within that.
        assert r == pytest.approx(o, abs=1e-6)


def test_emb_bytes_uses_four_bytes_per_float():
    """192-D TitaNet → 768 bytes. Keeps storage cost predictable."""
    assert len(runner._emb_to_bytes([0.0] * 192)) == 768


def test_running_average_first_sample():
    # avg of (existing, new) with n=1 = (existing + new) / 2
    out = runner._running_average([0.0], [1.0], n=1)
    assert out == [0.5]


def test_running_average_weights_existing_by_n():
    # 9 prior samples averaging to 1.0, plus a new 0.0 → new avg = 0.9
    out = runner._running_average([1.0], [0.0], n=9)
    assert out == [pytest.approx(0.9)]


def test_running_average_vector():
    out = runner._running_average([1.0, 2.0, 3.0], [4.0, 6.0, 8.0], n=1)
    assert out == [pytest.approx(2.5), pytest.approx(4.0), pytest.approx(5.5)]


# ──────────────────────────────────────────────────────────────────────────────
# Linker DB integration
# ──────────────────────────────────────────────────────────────────────────────

# Hand-tuned 4-D embeddings that test the threshold behavior without depending
# on a real model. The default threshold is 0.6.
USER_VOICE = [1.0, 0.1, 0.0, 0.0]   # roughly the "user" direction
MARIE_VOICE = [0.0, 1.0, 0.1, 0.0]  # orthogonal-ish to USER, "Marie"
PAUL_VOICE = [0.0, 0.0, 1.0, 0.1]   # orthogonal to both above, "Paul"

# Slightly noisy versions for next-conversation re-detection (should match):
USER_VOICE_NOISY = [0.95, 0.15, 0.05, 0.05]
MARIE_VOICE_NOISY = [0.05, 0.97, 0.12, 0.02]


def _user(name: str = "test") -> str:
    return name


def _segments_with_labels(labels: list[str]) -> list[dict]:
    """Build a fake segments list with one entry per label, no audio."""
    out = []
    for i, label in enumerate(labels):
        out.append(
            {
                "start": float(i),
                "end": float(i + 1),
                "text": f"line {i}",
                "speaker": label,
            }
        )
    return out


def test_link_creates_new_speakers_when_none_exist():
    segments = _segments_with_labels(["USER", "S1"])
    embeddings = {"USER": USER_VOICE, "S1": MARIE_VOICE}

    out = runner._link_speakers_to_segments(
        user_id=_user(),
        segments=segments,
        embeddings_by_label=embeddings,
    )

    # Both segments got speaker_id annotations.
    assert out[0]["speaker_id"] is not None
    assert out[1]["speaker_id"] is not None
    assert out[0]["speaker_id"] != out[1]["speaker_id"]

    with Session(engine) as db:
        speakers = list(db.exec(select(Speaker).where(Speaker.user_id == "test")).all())
    assert len(speakers) == 2
    # USER label flipped is_user=True; the other didn't.
    user_row = next(s for s in speakers if s.is_user)
    other_row = next(s for s in speakers if not s.is_user)
    assert user_row.mention_count == 1
    assert other_row.mention_count == 1


def test_link_matches_existing_speaker_above_threshold():
    """Second conversation with a noisy version of the same voice should
    REUSE the existing Speaker row, not create a new one."""
    # First conversation seeds the speaker.
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=_segments_with_labels(["USER"]),
        embeddings_by_label={"USER": USER_VOICE},
    )

    # Second conversation: noisy version of the same voice.
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=_segments_with_labels(["USER"]),
        embeddings_by_label={"USER": USER_VOICE_NOISY},
    )

    with Session(engine) as db:
        speakers = list(db.exec(select(Speaker).where(Speaker.user_id == "test")).all())
    assert len(speakers) == 1
    assert speakers[0].mention_count == 2
    assert speakers[0].is_user is True


def test_link_creates_new_speaker_below_threshold():
    """A genuinely different voice should NOT match the existing row."""
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=_segments_with_labels(["USER"]),
        embeddings_by_label={"USER": USER_VOICE},
    )
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=_segments_with_labels(["S1"]),
        embeddings_by_label={"S1": PAUL_VOICE},  # orthogonal to USER_VOICE
    )

    with Session(engine) as db:
        speakers = list(db.exec(select(Speaker).where(Speaker.user_id == "test")).all())
    assert len(speakers) == 2


def test_link_is_user_is_sticky():
    """Once a voice has been labeled USER in any conversation, future
    matches don't un-flag it even if it shows up as S1."""
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=_segments_with_labels(["USER"]),
        embeddings_by_label={"USER": USER_VOICE},
    )
    # Same embedding, now labeled S1 (maybe in a meeting they were quiet).
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=_segments_with_labels(["S1"]),
        embeddings_by_label={"S1": USER_VOICE_NOISY},
    )

    with Session(engine) as db:
        speakers = list(db.exec(select(Speaker).where(Speaker.user_id == "test")).all())
    assert len(speakers) == 1
    assert speakers[0].is_user is True  # sticky despite the S1 label


def test_link_user_id_isolation():
    """Speaker rows are scoped per user_id; identical voices for two users
    should NOT collide."""
    runner._link_speakers_to_segments(
        user_id="alice",
        segments=_segments_with_labels(["USER"]),
        embeddings_by_label={"USER": USER_VOICE},
    )
    runner._link_speakers_to_segments(
        user_id="bob",
        segments=_segments_with_labels(["USER"]),
        embeddings_by_label={"USER": USER_VOICE},
    )

    with Session(engine) as db:
        alice_rows = list(db.exec(select(Speaker).where(Speaker.user_id == "alice")).all())
        bob_rows = list(db.exec(select(Speaker).where(Speaker.user_id == "bob")).all())
    assert len(alice_rows) == 1
    assert len(bob_rows) == 1
    assert alice_rows[0].id != bob_rows[0].id


def test_link_skips_degenerate_zero_embedding():
    """A zero-norm embedding can't be cosined against anything — quietly
    skip rather than NaN-out the matching."""
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=_segments_with_labels(["USER"]),
        embeddings_by_label={"USER": [0.0, 0.0, 0.0, 0.0]},
    )
    with Session(engine) as db:
        speakers = list(db.exec(select(Speaker).where(Speaker.user_id == "test")).all())
    assert len(speakers) == 0


def test_link_running_average_with_truly_similar_inputs():
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=_segments_with_labels(["USER"]),
        embeddings_by_label={"USER": [1.0, 0.0, 0.0, 0.0]},
    )
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=_segments_with_labels(["USER"]),
        embeddings_by_label={"USER": [0.9, 0.1, 0.0, 0.0]},  # close enough to match
    )

    with Session(engine) as db:
        speakers = list(db.exec(select(Speaker).where(Speaker.user_id == "test")).all())
    assert len(speakers) == 1
    stored = runner._emb_from_bytes(speakers[0].embedding)
    # (1.0 + 0.9) / 2 = 0.95, (0.0 + 0.1) / 2 = 0.05
    assert stored[0] == pytest.approx(0.95, abs=1e-5)
    assert stored[1] == pytest.approx(0.05, abs=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# UI endpoints
# ──────────────────────────────────────────────────────────────────────────────


def _login(client: TestClient, password: str) -> None:
    client.post("/login", data={"username": "test", "password": password})


def _make_speaker(
    *,
    user_id: str = "test",
    name: str | None = None,
    is_user: bool = False,
    mention_count: int = 1,
) -> UUID:
    sid = uuid4()
    with Session(engine) as db:
        db.add(
            Speaker(
                id=sid,
                user_id=user_id,
                name=name,
                embedding=runner._emb_to_bytes(USER_VOICE),
                is_user=is_user,
                mention_count=mention_count,
            )
        )
        db.commit()
    return sid


def test_speakers_index_requires_login(client: TestClient):
    r = client.get("/speakers", follow_redirects=False)
    # Unauthenticated browser hit → 303 redirect to /login.
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_speakers_index_empty(client: TestClient, password: str):
    _login(client, password)
    r = client.get("/speakers")
    assert r.status_code == 200
    assert "No speakers yet" in r.text


def test_speakers_index_shows_known_speakers(client: TestClient, password: str):
    _login(client, password)
    _make_speaker(name="Marie", mention_count=3)
    _make_speaker(name=None, is_user=True, mention_count=5)
    r = client.get("/speakers")
    assert r.status_code == 200
    assert "Marie" in r.text
    # is_user marker visible
    assert "you" in r.text.lower()


def test_speaker_rename_sets_name(client: TestClient, password: str):
    _login(client, password)
    sid = _make_speaker(name=None)
    r = client.post(f"/speakers/{sid}/rename", data={"name": "Marie"})
    assert r.status_code == 200
    assert r.headers.get("HX-Refresh") == "true"
    with Session(engine) as db:
        sp = db.get(Speaker, sid)
    assert sp.name == "Marie"


def test_speaker_rename_clears_name_on_empty(client: TestClient, password: str):
    _login(client, password)
    sid = _make_speaker(name="Marie")
    r = client.post(f"/speakers/{sid}/rename", data={"name": "   "})
    assert r.status_code == 200
    with Session(engine) as db:
        sp = db.get(Speaker, sid)
    assert sp.name is None


def test_speaker_rename_404_on_wrong_user(client: TestClient, password: str):
    _login(client, password)
    sid = _make_speaker(user_id="someone-else", name="x")
    r = client.post(f"/speakers/{sid}/rename", data={"name": "y"})
    assert r.status_code == 404


def test_speaker_toggle_user_flips_flag(client: TestClient, password: str):
    _login(client, password)
    sid = _make_speaker(is_user=False)
    client.post(f"/speakers/{sid}/toggle-user")
    with Session(engine) as db:
        sp = db.get(Speaker, sid)
    assert sp.is_user is True
    # And flip back.
    client.post(f"/speakers/{sid}/toggle-user")
    with Session(engine) as db:
        sp = db.get(Speaker, sid)
    assert sp.is_user is False


def test_speaker_delete_removes_row(client: TestClient, password: str):
    _login(client, password)
    sid = _make_speaker(name="To Delete")
    r = client.post(f"/speakers/{sid}/delete")
    assert r.status_code == 200
    with Session(engine) as db:
        assert db.get(Speaker, sid) is None


def test_speaker_delete_404_on_wrong_user(client: TestClient, password: str):
    _login(client, password)
    sid = _make_speaker(user_id="someone-else", name="x")
    r = client.post(f"/speakers/{sid}/delete")
    assert r.status_code == 404


# ──────────────────────────────────────────────────────────────────────────────
# Conversation detail page surfaces speaker names + handles unlinked segments
# ──────────────────────────────────────────────────────────────────────────────


def _seed_conversation_with_segments(
    *,
    user: str = "test",
    segments: list[dict],
) -> UUID:
    """Insert a conversation + its audio session + transcript with the given
    segments. Returns conv.id for the test to GET /conversations/{id}."""
    sid = uuid4()
    cid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id=user,
                audio_path="/tmp/none.opus",
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
                user_id=user,
                title="Test conv",
                summary="...",
                started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            )
        )
        db.add(
            Transcript(
                audio_session_id=sid,
                text=" ".join(s.get("text", "") for s in segments),
                segments_json=json.dumps(segments),
                language="fr",
            )
        )
        db.commit()
    return cid


def test_conversation_page_shows_speaker_name_when_linked(
    client: TestClient, password: str
):
    _login(client, password)
    sid = _make_speaker(name="Marie")
    cid = _seed_conversation_with_segments(
        segments=[
            {"start": 0, "end": 5, "text": "Salut", "speaker": "S1", "speaker_id": str(sid)},
        ]
    )
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    # The label and the name should both appear (label in the speakers section
    # header / class, the name as the display label in the transcript line).
    assert "Marie" in r.text


def test_conversation_page_falls_back_to_label_for_unnamed_speaker(
    client: TestClient, password: str
):
    _login(client, password)
    sid = _make_speaker(name=None)  # known voice, not labeled yet
    cid = _seed_conversation_with_segments(
        segments=[
            {"start": 0, "end": 5, "text": "Salut", "speaker": "S1", "speaker_id": str(sid)},
        ]
    )
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    # No name set → display the per-conversation label.
    assert "[S1]" in r.text


def test_conversation_page_handles_pre_phase5_transcripts(
    client: TestClient, password: str
):
    """Segments from before linking shipped don't have speaker_id. The page
    should render them with the per-conversation label and not crash."""
    _login(client, password)
    cid = _seed_conversation_with_segments(
        segments=[
            {"start": 0, "end": 5, "text": "Salut", "speaker": "USER"},
        ]
    )
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    assert "[USER]" in r.text


def test_link_marks_segments_with_speaker_id():
    """End-to-end of the linker: pass segments + embeddings, get back
    segments with speaker_id keys filled in."""
    segments = [
        {"start": 0, "end": 5, "text": "a", "speaker": "USER"},
        {"start": 5, "end": 10, "text": "b", "speaker": "USER"},
        {"start": 10, "end": 15, "text": "c", "speaker": "S1"},
    ]
    out = runner._link_speakers_to_segments(
        user_id=_user(),
        segments=segments,
        embeddings_by_label={
            "USER": USER_VOICE,
            "S1": MARIE_VOICE,
        },
    )
    # Both USER segments share the same id; S1 gets a different one.
    assert out[0]["speaker_id"] == out[1]["speaker_id"]
    assert out[0]["speaker_id"] != out[2]["speaker_id"]


def test_link_threshold_respects_settings(monkeypatch):
    """If the user dials up the threshold, identical-voice re-detection
    should stop matching (over-strict). Confirms the setting is actually
    consulted at link-time, not frozen at import."""
    # Seed at default threshold (0.6).
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=_segments_with_labels(["USER"]),
        embeddings_by_label={"USER": USER_VOICE},
    )
    # Crank threshold to 0.999 — even noisy-self shouldn't match.
    monkeypatch.setattr(settings, "speaker_match_threshold", 0.999)
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=_segments_with_labels(["USER"]),
        embeddings_by_label={"USER": USER_VOICE_NOISY},
    )
    with Session(engine) as db:
        speakers = list(db.exec(select(Speaker).where(Speaker.user_id == "test")).all())
    # Should be TWO rows now — the high threshold treated them as different.
    assert len(speakers) == 2
