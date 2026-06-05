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
# Merge — what the user wanted when they renamed two rows to the same name
# ──────────────────────────────────────────────────────────────────────────────


def _make_speaker_with_embedding(
    *,
    user_id: str = "test",
    name: str | None = None,
    is_user: bool = False,
    mention_count: int = 1,
    embedding: list[float] | None = None,
) -> UUID:
    """Like _make_speaker but lets the test specify the embedding so we can
    assert the post-merge centroid mathematically."""
    sid = uuid4()
    with Session(engine) as db:
        db.add(
            Speaker(
                id=sid,
                user_id=user_id,
                name=name,
                embedding=runner._emb_to_bytes(embedding if embedding is not None else USER_VOICE),
                is_user=is_user,
                mention_count=mention_count,
            )
        )
        db.commit()
    return sid


def test_merge_combines_two_speakers_into_one(client: TestClient, password: str):
    _login(client, password)
    a = _make_speaker_with_embedding(
        name=None, mention_count=3, embedding=[1.0, 0.0, 0.0, 0.0]
    )
    b = _make_speaker_with_embedding(
        name="Marie", mention_count=5, embedding=[0.0, 1.0, 0.0, 0.0]
    )
    r = client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(a), str(b)]},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/speakers"
    with Session(engine) as db:
        speakers = list(db.exec(select(Speaker).where(Speaker.user_id == "test")).all())
    assert len(speakers) == 1
    survivor = speakers[0]
    # Named survivor wins as primary
    assert survivor.name == "Marie"
    # mention_count sums
    assert survivor.mention_count == 8
    # Embedding is weighted average: (3 * a + 5 * b) / 8
    stored = runner._emb_from_bytes(survivor.embedding)
    assert stored[0] == pytest.approx(3 / 8, abs=1e-5)
    assert stored[1] == pytest.approx(5 / 8, abs=1e-5)


def test_merge_picks_named_primary_over_unnamed(client: TestClient, password: str):
    """Even if the unnamed has more mentions, the named one wins as primary
    because keeping the name is the user's intent more often than not."""
    _login(client, password)
    unnamed_big = _make_speaker_with_embedding(name=None, mention_count=100)
    named_small = _make_speaker_with_embedding(name="Marie", mention_count=2)
    client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(unnamed_big), str(named_small)]},
    )
    with Session(engine) as db:
        speakers = list(db.exec(select(Speaker).where(Speaker.user_id == "test")).all())
    assert len(speakers) == 1
    assert speakers[0].name == "Marie"


def test_merge_picks_most_mentioned_when_both_named(client: TestClient, password: str):
    """When both candidates have names, mention count breaks the tie."""
    _login(client, password)
    little = _make_speaker_with_embedding(name="Old typo", mention_count=2)
    big = _make_speaker_with_embedding(name="Marie", mention_count=20)
    client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(little), str(big)]},
    )
    with Session(engine) as db:
        speakers = list(db.exec(select(Speaker).where(Speaker.user_id == "test")).all())
    assert len(speakers) == 1
    assert speakers[0].name == "Marie"


def test_merge_is_user_flag_is_or_of_constituents(
    client: TestClient, password: str
):
    """If ANY of the merged speakers was marked is_user, the survivor is."""
    _login(client, password)
    a = _make_speaker_with_embedding(is_user=False)
    b = _make_speaker_with_embedding(is_user=True)
    client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(a), str(b)]},
    )
    with Session(engine) as db:
        speakers = list(db.exec(select(Speaker).where(Speaker.user_id == "test")).all())
    assert len(speakers) == 1
    assert speakers[0].is_user is True


def test_merge_unnamed_primary_inherits_first_named_secondary(
    client: TestClient, password: str
):
    """Edge case: NO row is named yet but they're being merged anyway. The
    inheritance path covers a corner where ranking has the named candidate
    last (e.g. heavy unnamed beats a small named one — but that case picks
    the named as primary). This test covers the all-unnamed case."""
    _login(client, password)
    a = _make_speaker_with_embedding(name=None, mention_count=5)
    b = _make_speaker_with_embedding(name=None, mention_count=3)
    client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(a), str(b)]},
    )
    with Session(engine) as db:
        speakers = list(db.exec(select(Speaker).where(Speaker.user_id == "test")).all())
    assert len(speakers) == 1
    assert speakers[0].name is None  # nothing to inherit
    assert speakers[0].mention_count == 8


def test_merge_three_or_more_speakers(client: TestClient, password: str):
    _login(client, password)
    ids = [_make_speaker_with_embedding(name=None, mention_count=1) for _ in range(4)]
    r = client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(s) for s in ids]},
        follow_redirects=False,
    )
    assert r.status_code == 303
    with Session(engine) as db:
        speakers = list(db.exec(select(Speaker).where(Speaker.user_id == "test")).all())
    assert len(speakers) == 1
    assert speakers[0].mention_count == 4


def test_merge_rewrites_transcript_segments_to_primary(
    client: TestClient, password: str
):
    """The real test of merge usefulness: past conversation pages must
    stop showing the merged voices as separate labels. This is the bit
    that distinguishes 'merge' from 'rename to the same string'."""
    _login(client, password)
    primary = _make_speaker_with_embedding(name="Marie", mention_count=5)
    dupe = _make_speaker_with_embedding(name=None, mention_count=2)
    cid = _seed_conversation_with_segments(
        segments=[
            {"start": 0, "end": 5, "text": "Salut", "speaker": "S1",
             "speaker_id": str(dupe)},
            {"start": 5, "end": 10, "text": "Ça va", "speaker": "S2",
             "speaker_id": str(primary)},
        ]
    )
    client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(primary), str(dupe)]},
    )
    # Inspect the persisted transcript directly.
    with Session(engine) as db:
        t = db.exec(
            select(Transcript)
            .join(AudioSession, Transcript.audio_session_id == AudioSession.id)
            .where(Conversation.id == cid)
            .join(Conversation, Conversation.audio_session_id == AudioSession.id)
        ).first()
        # Fallback selector in case the join above is sensitive to driver:
        if t is None:
            convs = list(db.exec(select(Conversation).where(Conversation.id == cid)).all())
            assert convs
            t = db.exec(
                select(Transcript).where(Transcript.audio_session_id == convs[0].audio_session_id)
            ).first()
    assert t is not None
    segs = json.loads(t.segments_json)
    # BOTH segments now point at the primary; the dupe's id appears nowhere.
    for seg in segs:
        assert seg["speaker_id"] == str(primary)


def test_merge_rejects_fewer_than_two_ids(client: TestClient, password: str):
    _login(client, password)
    a = _make_speaker(name="x")
    r = client.post("/speakers/merge", data={"speaker_ids": [str(a)]})
    assert r.status_code == 400


def test_merge_rejects_unknown_id(client: TestClient, password: str):
    _login(client, password)
    a = _make_speaker(name="x")
    bogus = uuid4()
    r = client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(a), str(bogus)]},
    )
    assert r.status_code == 404


def test_merge_rejects_cross_user(client: TestClient, password: str):
    """A merge form that includes another user's Speaker id must fail
    rather than quietly merging across accounts."""
    _login(client, password)
    mine = _make_speaker(name="mine")
    theirs = _make_speaker(user_id="someone-else", name="theirs")
    r = client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(mine), str(theirs)]},
    )
    # Either 404 (filter found N-1 rows) — we don't reveal "exists but yours not".
    assert r.status_code == 404
    # Both rows still exist; the cross-user one was untouched.
    with Session(engine) as db:
        assert db.get(Speaker, mine) is not None
        assert db.get(Speaker, theirs) is not None


def test_merge_rejects_duplicate_ids_in_selection(
    client: TestClient, password: str
):
    _login(client, password)
    a = _make_speaker(name="x")
    r = client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(a), str(a)]},
    )
    assert r.status_code == 400


def test_merge_rejects_invalid_uuid(client: TestClient, password: str):
    _login(client, password)
    a = _make_speaker(name="x")
    r = client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(a), "not-a-uuid"]},
    )
    assert r.status_code == 400


def test_speakers_page_has_merge_form(client: TestClient, password: str):
    """Regression guard: the merge UI must appear on /speakers, otherwise
    the feature is unreachable from the UI."""
    _login(client, password)
    _make_speaker(name="a", mention_count=1)
    _make_speaker(name="b", mention_count=1)
    r = client.get("/speakers")
    assert r.status_code == 200
    assert 'action="/speakers/merge"' in r.text
    assert "Merge selected" in r.text
    # Explanation about rename-vs-merge is also there.
    assert "NOT a merge" in r.text


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


def test_conversation_page_shows_toggle_user_button_for_each_speaker(
    client: TestClient, password: str
):
    """The 'this is me / not me' affordance only existed on /speakers, where
    the user can't hear the voices to ground their decision. Mirror it on
    the conversation page so they can listen and tag without switching
    tabs."""
    _login(client, password)
    sid_a = _make_speaker(name=None, is_user=False)
    sid_b = _make_speaker(name="Marie", is_user=True)
    cid = _seed_conversation_with_segments(
        segments=[
            {"start": 0, "end": 5, "text": "Salut", "speaker": "S1",
             "speaker_id": str(sid_a)},
            {"start": 5, "end": 10, "text": "Bonjour", "speaker": "USER",
             "speaker_id": str(sid_b)},
        ]
    )
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    # The unmarked one shows the "mark as me" affordance. The compact
    # conversation-page layout uses "↑ me" as the button text (vs
    # /speakers' longer "↑ this is me") to fit a single-line row; the
    # full phrase still lives in the title=... tooltip.
    assert "↑ me" in r.text or "this is me" in r.text
    # The user-marked one shows the inverse.
    assert "not me" in r.text
    # Both forms point at the toggle endpoint.
    assert f"/speakers/{sid_a}/toggle-user" in r.text
    assert f"/speakers/{sid_b}/toggle-user" in r.text


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


# ──────────────────────────────────────────────────────────────────────────────
# Preview pointer: the audio snippet shown on /speakers per voice
# ──────────────────────────────────────────────────────────────────────────────


def test_link_populates_preview_pointer_when_session_given():
    """Linker should pick the longest segment per label and store it as
    the preview pointer on the matched/created Speaker."""
    session_id = uuid4()
    segments = [
        {"start": 0.0, "end": 2.5, "text": "a", "speaker": "USER"},
        {"start": 3.0, "end": 9.5, "text": "b", "speaker": "USER"},   # longest USER
        {"start": 10.0, "end": 12.0, "text": "c", "speaker": "S1"},
    ]
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=segments,
        embeddings_by_label={"USER": USER_VOICE, "S1": MARIE_VOICE},
        audio_session_id=session_id,
    )
    with Session(engine) as db:
        speakers = list(db.exec(select(Speaker).where(Speaker.user_id == "test")).all())
    by_user = next(s for s in speakers if s.is_user)
    by_other = next(s for s in speakers if not s.is_user)
    assert by_user.preview_audio_session_id == session_id
    assert by_user.preview_start_s == 3.0
    assert by_user.preview_end_s == 9.5  # well under the 15s cap
    assert by_other.preview_audio_session_id == session_id
    assert by_other.preview_start_s == 10.0
    assert by_other.preview_end_s == 12.0


def test_link_caps_preview_at_max_seconds():
    """A 60-second segment shouldn't produce a 60-second preview clip —
    capped to runner._PREVIEW_MAX_SECONDS so the audio player on /speakers
    never has to stream a long slice."""
    session_id = uuid4()
    segments = [
        {"start": 5.0, "end": 65.0, "text": "very long", "speaker": "USER"},
    ]
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=segments,
        embeddings_by_label={"USER": USER_VOICE},
        audio_session_id=session_id,
    )
    with Session(engine) as db:
        sp = db.exec(select(Speaker).where(Speaker.user_id == "test")).first()
    assert sp.preview_start_s == 5.0
    # 15s cap from segment start, NOT from segment end.
    assert sp.preview_end_s == 5.0 + runner._PREVIEW_MAX_SECONDS


def test_link_does_not_overwrite_longer_preview():
    """Later conversations shouldn't downgrade a Speaker's preview clip —
    only replace it when the new candidate is strictly longer."""
    session_a = uuid4()
    session_b = uuid4()
    # First conversation: long USER segment seeds a 12s preview.
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=[{"start": 0.0, "end": 12.0, "speaker": "USER"}],
        embeddings_by_label={"USER": USER_VOICE},
        audio_session_id=session_a,
    )
    # Second conversation: short USER segment shouldn't downgrade.
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=[{"start": 0.0, "end": 2.0, "speaker": "USER"}],
        embeddings_by_label={"USER": USER_VOICE_NOISY},
        audio_session_id=session_b,
    )
    with Session(engine) as db:
        sp = db.exec(select(Speaker).where(Speaker.user_id == "test")).first()
    # Preview still points at the 12s clip from session_a.
    assert sp.preview_audio_session_id == session_a
    assert sp.preview_end_s == 12.0


def test_link_replaces_preview_when_new_is_longer():
    """The inverse: a longer clip in a later conversation DOES replace
    the existing shorter one. Lets the preview improve over time as the
    speaker is heard in more / better recordings."""
    session_a = uuid4()
    session_b = uuid4()
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=[{"start": 0.0, "end": 2.0, "speaker": "USER"}],
        embeddings_by_label={"USER": USER_VOICE},
        audio_session_id=session_a,
    )
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=[{"start": 5.0, "end": 11.0, "speaker": "USER"}],
        embeddings_by_label={"USER": USER_VOICE_NOISY},
        audio_session_id=session_b,
    )
    with Session(engine) as db:
        sp = db.exec(select(Speaker).where(Speaker.user_id == "test")).first()
    assert sp.preview_audio_session_id == session_b
    assert sp.preview_start_s == 5.0
    assert sp.preview_end_s == 11.0


def test_link_no_session_id_leaves_preview_null():
    """Backward compat: callers that don't pass audio_session_id (e.g. unit
    tests, manual /tune replays) shouldn't break — they just skip the
    preview-update step."""
    runner._link_speakers_to_segments(
        user_id=_user(),
        segments=[{"start": 0, "end": 5, "speaker": "USER"}],
        embeddings_by_label={"USER": USER_VOICE},
        # no audio_session_id
    )
    with Session(engine) as db:
        sp = db.exec(select(Speaker).where(Speaker.user_id == "test")).first()
    assert sp.preview_audio_session_id is None
    assert sp.preview_start_s is None
    assert sp.preview_end_s is None


# ──────────────────────────────────────────────────────────────────────────────
# UI: preview audio element renders on /speakers when pointer is set
# ──────────────────────────────────────────────────────────────────────────────


def test_speakers_page_renders_audio_preview(client: TestClient, password: str):
    """When a speaker has a preview pointer, /speakers includes an <audio>
    element targeting the right session with the right time fragment."""
    _login(client, password)
    sid_audio = uuid4()
    # Seed the AudioSession the preview points at — otherwise FK fails.
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid_audio,
                user_id="test",
                audio_path="/tmp/x.opus",
                codec="opus",
                started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                status=SessionStatus.done,
            )
        )
        db.flush()
        speaker = Speaker(
            user_id="test",
            name="Marie",
            embedding=runner._emb_to_bytes(USER_VOICE),
            preview_audio_session_id=sid_audio,
            preview_start_s=3.5,
            preview_end_s=11.25,
        )
        db.add(speaker)
        db.commit()
    r = client.get("/speakers")
    assert r.status_code == 200
    # The audio element points at the right session with the
    # media-fragment range covering the preview window.
    assert f"/sessions/{sid_audio}/audio#t=3.50,11.25" in r.text
    # And the human-readable clip-duration hint shows the actual length.
    assert "7.8s clip" in r.text


def test_speakers_page_shows_no_preview_for_old_speakers(
    client: TestClient, password: str
):
    """Speakers seeded before the preview feature existed have null
    pointer fields; the UI should say so cleanly rather than rendering
    a broken audio element."""
    _login(client, password)
    _make_speaker(name="Old voice")  # preview fields default to None
    r = client.get("/speakers")
    assert r.status_code == 200
    assert "no preview yet" in r.text


# ──────────────────────────────────────────────────────────────────────────────
# Merge: return_to + same-origin guard + conversation-page merge form
# ──────────────────────────────────────────────────────────────────────────────


def test_merge_redirects_to_speakers_by_default(
    client: TestClient, password: str
):
    """Without return_to, merge lands the user on /speakers (current
    behavior — don't break existing /speakers form)."""
    _login(client, password)
    a = _make_speaker(name="dup1")
    b = _make_speaker(name="dup2")
    r = client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(a), str(b)]},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/speakers"


def test_merge_respects_return_to(client: TestClient, password: str):
    """Conversation page's merge form passes return_to so the user stays
    on the conversation after merging."""
    _login(client, password)
    a = _make_speaker(name="dup1")
    b = _make_speaker(name="dup2")
    target = f"/conversations/{uuid4()}"
    r = client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(a), str(b)], "return_to": target},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == target


def test_merge_rejects_offsite_return_to(client: TestClient, password: str):
    """Hostile form value can't bounce the user to another origin —
    same-origin guard kicks in and falls back to /speakers."""
    _login(client, password)
    a = _make_speaker(name="dup1")
    b = _make_speaker(name="dup2")
    for bad in (
        "https://evil.example.com/",
        "//evil.example.com/page",
        "javascript:alert(1)",
        "ftp://elsewhere/",
    ):
        r = client.post(
            "/speakers/merge",
            data={"speaker_ids": [str(a), str(b)], "return_to": bad},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert r.headers["location"] == "/speakers", (
            f"expected fallback to /speakers for {bad!r}, "
            f"got {r.headers['location']!r}"
        )
        # Re-seed for next iteration (the merge deleted the secondary).
        a = _make_speaker(name="dup1")
        b = _make_speaker(name="dup2")


def test_conversation_page_renders_merge_form_when_2plus_speakers(
    client: TestClient, password: str
):
    """Conversation page's speakers section gets multi-select + merge
    button when 2+ speakers are linked. Lets the user fold over-split
    voices without bouncing to /speakers."""
    _login(client, password)
    sp_user = _make_speaker(name=None, is_user=True)
    sp_marie = _make_speaker(name="Marie")
    cid = _seed_conversation_with_segments(
        segments=[
            {"start": 0, "end": 5, "text": "hello", "speaker": "USER",
             "speaker_id": str(sp_user)},
            {"start": 5, "end": 10, "text": "bonjour", "speaker": "S1",
             "speaker_id": str(sp_marie)},
        ]
    )
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    # Checkboxes for both speakers.
    assert f'value="{sp_user}"' in r.text
    assert f'value="{sp_marie}"' in r.text
    # Merge form points at /speakers/merge with return_to set back here.
    assert 'action="/speakers/merge"' in r.text
    assert f'value="/conversations/{cid}"' in r.text
    # The "Merge selected" button text is present.
    assert "Merge selected" in r.text


def test_conversation_page_dedupes_speakers_by_id(
    client: TestClient, password: str
):
    """When multiple in-conversation labels point at the SAME Speaker
    (because sherpa-onnx over-split a person whose embeddings the
    cross-conv linker correctly re-merged), the speakers section must
    render ONE row per distinct Speaker with the labels grouped — not
    one row per label. Otherwise ticking the duplicates would post the
    same id twice and the merge endpoint would reject it."""
    _login(client, password)
    marie = _make_speaker(name="Marie")
    paul = _make_speaker(name="Paul")
    cid = _seed_conversation_with_segments(
        segments=[
            # Two different labels for Marie (over-split that re-merged).
            {"start": 0, "end": 5, "text": "a", "speaker": "S1",
             "speaker_id": str(marie)},
            {"start": 5, "end": 9, "text": "b", "speaker": "S3",
             "speaker_id": str(marie)},
            # And one for Paul.
            {"start": 10, "end": 14, "text": "c", "speaker": "S2",
             "speaker_id": str(paul)},
        ]
    )
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    # Each Speaker.id appears as a checkbox VALUE exactly ONCE — that's
    # the property that makes the merge POST safe. (The label spans
    # themselves can repeat in other parts of the HTML.)
    body = r.text
    assert body.count(f'value="{marie}"') == 1, (
        "Marie's id should appear in exactly one checkbox value"
    )
    assert body.count(f'value="{paul}"') == 1
    # Both of Marie's labels show up grouped in the row header so the
    # user can see which sherpa-onnx clusters got folded into her.
    assert "S1" in body and "S3" in body
    # And the linked-count visible in the summary is the DISTINCT count
    # (2 Speakers), not the per-label count (3 entries).
    assert "2 linked" in body


def test_merge_silently_dedupes_duplicate_ids(client: TestClient, password: str):
    """Defensive: a stale tab or hand-crafted form posting the same id
    twice used to 400 with 'duplicate speaker ids in selection'. The
    UI-side fix dedupes by speaker_id, but the backend now also dedupes
    so a stale POST doesn't drop the user on a useless error page."""
    _login(client, password)
    a = _make_speaker(name="dup1")
    b = _make_speaker(name="dup2")
    # Post a's id twice + b's id once. Backend should treat it as
    # "merge a + b" rather than rejecting the dupe.
    r = client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(a), str(a), str(b)]},
        follow_redirects=False,
    )
    assert r.status_code == 303
    # b was merged into a (or vice versa); only one row remains.
    with Session(engine) as db:
        rows = list(
            db.exec(select(Speaker).where(Speaker.user_id == "test")).all()
        )
    assert len(rows) == 1


def test_merge_rejects_post_with_only_one_distinct_id(
    client: TestClient, password: str
):
    """Merging a speaker with itself is meaningless. After dedupe, if
    fewer than 2 distinct ids remain, we 400 with a clearer message
    than the old 'duplicate speaker ids' (which was misleading — the
    duplicates weren't the bug, the lack of a second target was)."""
    _login(client, password)
    a = _make_speaker(name="solo")
    r = client.post(
        "/speakers/merge",
        data={"speaker_ids": [str(a), str(a), str(a)]},
    )
    assert r.status_code == 400


def test_conversation_page_hides_merge_button_with_one_speaker(
    client: TestClient, password: str
):
    """Single-speaker conversations get no merge button (nothing to
    merge with). Still shows the rename/toggle controls."""
    _login(client, password)
    sp = _make_speaker(name=None, is_user=True)
    cid = _seed_conversation_with_segments(
        segments=[
            {"start": 0, "end": 5, "text": "hello", "speaker": "USER",
             "speaker_id": str(sp)},
        ]
    )
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    # No visible merge button when there's only one linked speaker. The
    # phrase "Merge selected" still appears in the checkbox `title=` tooltip
    # — check for the button-specific marker (🔗 prefix) which only
    # appears inside the `{% if linked_speaker_count >= 2 %}` block.
    assert "🔗 Merge selected" not in r.text
