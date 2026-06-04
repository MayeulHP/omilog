"""Audio-synced transcript UI tests.

The actual click + highlight behaviour is vanilla browser JS — we don't
run a headless browser here. What we DO test is the server-side contract
the JS depends on: the audio element has the right id, each transcript
paragraph carries data-start / data-end attributes the JS can parse,
the follow-playback checkbox only appears when there's audio to sync to.

If any of these break (someone refactors the template and drops an
attribute), the JS silently no-ops in the browser. Tests here are the
regression guard.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session

from omilog.db import engine
from omilog.models import (
    AudioSession,
    Conversation,
    SessionStatus,
    Transcript,
)


def _login(client: TestClient, password: str) -> None:
    client.post("/login", data={"username": "test", "password": password})


def _seed_conversation_with_audio_and_segments(
    *,
    user: str = "test",
    segments: list[dict] | None = None,
    has_audio: bool = True,
) -> tuple[str, str]:
    """Returns (audio_session_id, conversation_id) for the test to GET."""
    sid = uuid4()
    cid = uuid4()
    segments = segments if segments is not None else [
        {"start": 0.0, "end": 5.0, "text": "Salut, ça va?", "speaker": "USER"},
        {"start": 5.0, "end": 9.5, "text": "Très bien.", "speaker": "S1"},
    ]
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id=user,
                audio_path="/tmp/x.opus" if has_audio else None,
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
                title="t",
                started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            )
        )
        db.add(
            Transcript(
                audio_session_id=sid,
                text=" ".join(s["text"] for s in segments),
                segments_json=json.dumps(segments),
                language="fr",
            )
        )
        db.commit()
    return str(sid), str(cid)


def test_audio_element_has_known_id(client: TestClient, password: str):
    """The JS hooks <audio id="conv-audio">. If someone renames it, the
    sync silently dies."""
    _login(client, password)
    _, cid = _seed_conversation_with_audio_and_segments()
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    assert 'id="conv-audio"' in r.text


def test_each_segment_carries_data_start_and_data_end(
    client: TestClient, password: str
):
    """Both attributes must be present; JS parses them with parseFloat
    and falls back gracefully if missing, but we want to surface real data
    not silently-disabled paragraphs."""
    _login(client, password)
    _, cid = _seed_conversation_with_audio_and_segments(
        segments=[
            {"start": 0.0, "end": 5.0, "text": "first", "speaker": "USER"},
            {"start": 12.5, "end": 18.25, "text": "second", "speaker": "S1"},
        ]
    )
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    # Three-decimal precision in the template format string preserves
    # sub-millisecond timing without going overboard.
    assert 'data-start="0.000"' in r.text
    assert 'data-end="5.000"' in r.text
    assert 'data-start="12.500"' in r.text
    assert 'data-end="18.250"' in r.text


def test_follow_checkbox_renders_when_audio_present(
    client: TestClient, password: str
):
    _login(client, password)
    _, cid = _seed_conversation_with_audio_and_segments()
    r = client.get(f"/conversations/{cid}")
    assert 'id="transcript-follow"' in r.text


def test_no_follow_checkbox_when_audio_missing(
    client: TestClient, password: str
):
    """If the source audio is gone (deleted, archived), no checkbox: the
    sync JS no-ops anyway, but rendering a dead UI control is noisy."""
    _login(client, password)
    _, cid = _seed_conversation_with_audio_and_segments(has_audio=False)
    r = client.get(f"/conversations/{cid}")
    assert 'id="transcript-follow"' not in r.text
    # And the <script> block that wires up the sync should also be absent.
    assert 'conv-audio' not in r.text or 'addEventListener' not in r.text


def test_no_sync_script_when_no_transcript_segments(
    client: TestClient, password: str
):
    """A transcript that's only plain text (no segments_json) can't be
    synced; the script tag should be skipped entirely."""
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
        db.add(
            Transcript(
                audio_session_id=sid,
                text="plain text only, no segment timings",
                segments_json=None,  # the key bit
                language="fr",
            )
        )
        db.commit()
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    # Audio still renders, transcript renders as bulk text, no sync.
    assert 'id="conv-audio"' in r.text
    assert 'data-start' not in r.text


def test_segment_titles_advertise_click_to_jump(
    client: TestClient, password: str
):
    """Hover hint so the click affordance isn't a mystery. Catches the
    case where someone refactors and drops the title attribute."""
    _login(client, password)
    _, cid = _seed_conversation_with_audio_and_segments(
        segments=[
            {"start": 65.0, "end": 70.0, "text": "x", "speaker": "USER"},
        ]
    )
    r = client.get(f"/conversations/{cid}")
    assert "Click to jump to 01:05" in r.text


def test_segments_without_start_get_zero(client: TestClient, password: str):
    """Defensive: a malformed segment with no 'start' field still renders
    a clickable line, just one that seeks to t=0. Better than crashing
    the whole transcript over a single bad row."""
    _login(client, password)
    _, cid = _seed_conversation_with_audio_and_segments(
        segments=[
            {"text": "no timing info here", "speaker": "USER"},
        ]
    )
    r = client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    assert 'data-start="0.000"' in r.text
