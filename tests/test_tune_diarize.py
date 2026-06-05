"""/tune/{id}/diarize endpoint tests.

The real diarize call goes through sherpa-onnx, which isn't available in
CI. We monkeypatch `diarize_uncached` so the route can be exercised end to
end without the diarization extra installed. Real diarization is covered
by the runner-integration tests in test_diarize.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session

from omilog.config import settings
from omilog.db import engine
from omilog.models import (
    AudioSession,
    Conversation,
    SessionStatus,
    Transcript,
)


def _login(client: TestClient, password: str) -> None:
    client.post("/login", data={"username": "test", "password": password})


def _seed_session_with_audio_and_transcript(tmp_path: Path) -> UUID:
    """Build a session with a real on-disk audio file and a transcript whose
    segments we can re-label. Audio bytes are placeholder — the diarize
    function is mocked so ffmpeg never sees them."""
    sid = uuid4()
    cid = uuid4()
    audio = tmp_path / f"{sid}.opus"
    audio.write_bytes(b"fake-opus")
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="test",
                audio_path=str(audio),
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
                text="bonjour ça va",
                segments_json=json.dumps([
                    {"start": 0.0, "end": 5.0, "text": "Bonjour", "speaker": "USER"},
                    {"start": 5.0, "end": 10.0, "text": "Ça va", "speaker": "S1"},
                ]),
                language="fr",
            )
        )
        db.commit()
    return sid


def test_tune_session_page_shows_diarize_section(
    client: TestClient, password: str, tmp_path, monkeypatch
):
    """When diarization is enabled, the /tune/<id> page must include the
    Tune diarization heading and the four knobs."""
    monkeypatch.setattr(settings, "diarization_enabled", True)
    # Pretend sherpa-onnx is importable for the page-render check.
    from omilog.pipeline import diarize as diarize_mod
    monkeypatch.setattr(diarize_mod, "DIARIZATION_AVAILABLE", True, raising=False)
    sid = _seed_session_with_audio_and_transcript(tmp_path)

    _login(client, password)
    r = client.get(f"/tune/{sid}")
    assert r.status_code == 200
    assert "Tune diarization" in r.text
    for key in ("num_clusters", "cluster_threshold", "min_speech_s", "min_silence_s"):
        assert f'name="{key}"' in r.text


def test_tune_session_page_omits_diarize_when_disabled(
    client: TestClient, password: str, tmp_path, monkeypatch
):
    """When diarization is off, the heading is there but the form is not —
    a soft 'enable it first' message replaces the controls."""
    monkeypatch.setattr(settings, "diarization_enabled", False)
    sid = _seed_session_with_audio_and_transcript(tmp_path)

    _login(client, password)
    r = client.get(f"/tune/{sid}")
    assert r.status_code == 200
    # Heading still renders so the user knows the section exists.
    assert "Tune diarization" in r.text
    # But the form fields don't.
    assert 'name="num_clusters"' not in r.text


def test_tune_diarize_rejects_when_disabled(
    client: TestClient, password: str, tmp_path, monkeypatch
):
    """Direct POST while diarization is off → 400."""
    monkeypatch.setattr(settings, "diarization_enabled", False)
    sid = _seed_session_with_audio_and_transcript(tmp_path)

    _login(client, password)
    r = client.post(
        f"/tune/{sid}/diarize",
        data={
            "num_clusters": 2,
            "cluster_threshold": 0.4,
            "min_speech_s": 0.5,
            "min_silence_s": 0.5,
        },
    )
    assert r.status_code == 400


def test_tune_diarize_404_for_wrong_user(
    client: TestClient, password: str, tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "diarization_enabled", True)
    from omilog.pipeline import diarize as diarize_mod
    monkeypatch.setattr(diarize_mod, "DIARIZATION_AVAILABLE", True, raising=False)

    sid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="someone-else",
                audio_path="/tmp/x.opus",
                codec="opus",
                started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                status=SessionStatus.done,
            )
        )
        db.commit()

    _login(client, password)
    r = client.post(
        f"/tune/{sid}/diarize",
        data={
            "num_clusters": 2,
            "cluster_threshold": 0.4,
            "min_speech_s": 0.5,
            "min_silence_s": 0.5,
        },
    )
    assert r.status_code == 404


def test_tune_diarize_runs_and_renders_results(
    client: TestClient, password: str, tmp_path, monkeypatch
):
    """Happy path: mocked diarize returns 2 clusters across 4 turns; the
    fragment shows cluster talk times, the relabeled segment count, and
    the apply-defaults form."""
    monkeypatch.setattr(settings, "diarization_enabled", True)
    from omilog.pipeline import diarize as diarize_mod
    from omilog.web import routes as web_routes
    monkeypatch.setattr(diarize_mod, "DIARIZATION_AVAILABLE", True, raising=False)
    sid = _seed_session_with_audio_and_transcript(tmp_path)

    fake_turns = [
        {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
        {"start": 4.5, "end": 6.5, "speaker": "SPEAKER_01"},
        {"start": 7.0, "end": 9.0, "speaker": "SPEAKER_00"},
        {"start": 9.5, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    # Patch the network/CPU heavy bits. transcode is mocked so ffmpeg
    # doesn't need to actually run on the placeholder audio.
    with patch.object(
        web_routes, "transcode_to_wav_bytes",
        new=AsyncMock(return_value=b"riff-wav-fake"),
    ), patch.object(
        diarize_mod, "diarize_uncached",
        new=AsyncMock(return_value=fake_turns),
    ):
        _login(client, password)
        r = client.post(
            f"/tune/{sid}/diarize",
            data={
                "num_clusters": 2,
                "cluster_threshold": 0.4,
                "min_speech_s": 0.5,
                "min_silence_s": 0.5,
            },
        )
    assert r.status_code == 200
    assert "2 cluster(s)" in r.text
    assert "4 turn(s)" in r.text
    assert "SPEAKER_00" in r.text
    assert "SPEAKER_01" in r.text
    # After USER/S1 relabel:
    assert ">USER<" in r.text or "speaker-USER" in r.text
    # The apply-defaults form must be there so the user can persist.
    assert 'action="/tune/apply-diarize-defaults"' in r.text or '/tune/apply-diarize-defaults' in r.text


def test_tune_diarize_clamps_out_of_range_params(
    client: TestClient, password: str, tmp_path, monkeypatch
):
    """Form values outside the declared ranges (like num_clusters=999)
    get clamped server-side rather than crashing sherpa-onnx with a
    weird configuration."""
    monkeypatch.setattr(settings, "diarization_enabled", True)
    from omilog.pipeline import diarize as diarize_mod
    from omilog.web import routes as web_routes
    monkeypatch.setattr(diarize_mod, "DIARIZATION_AVAILABLE", True, raising=False)
    sid = _seed_session_with_audio_and_transcript(tmp_path)

    captured = {}

    async def fake_diarize(*args, **kwargs):
        # The route calls diarize_uncached(wav_bytes, **kwargs). Stash the
        # kwargs only — the positional wav arg is just bytes.
        captured.update(kwargs)
        return []  # empty turns is fine for this test

    with patch.object(
        web_routes, "transcode_to_wav_bytes",
        new=AsyncMock(return_value=b"wav"),
    ), patch.object(
        diarize_mod, "diarize_uncached", new=fake_diarize,
    ):
        _login(client, password)
        client.post(
            f"/tune/{sid}/diarize",
            data={
                "num_clusters": 999,         # clamped to 32
                "cluster_threshold": 5.0,    # clamped to 0.95
                "min_speech_s": 0.01,        # clamped to 0.1
                "min_silence_s": 100,        # clamped to 3.0
            },
        )
    assert captured["num_clusters"] == 32
    assert captured["cluster_threshold"] == 0.95
    assert captured["min_speech_s"] == 0.1
    assert captured["min_silence_s"] == 3.0


def test_tune_diarize_handles_diarize_error(
    client: TestClient, password: str, tmp_path, monkeypatch
):
    """sherpa-onnx blowups (model can't load, broadcast crash, etc.) get
    surfaced as a 400 with the error visible in the fragment."""
    monkeypatch.setattr(settings, "diarization_enabled", True)
    from omilog.pipeline import diarize as diarize_mod
    from omilog.web import routes as web_routes
    monkeypatch.setattr(diarize_mod, "DIARIZATION_AVAILABLE", True, raising=False)
    sid = _seed_session_with_audio_and_transcript(tmp_path)

    async def fake_diarize_raises(*_args, **_kwargs):
        raise diarize_mod.DiarizationError("model file is corrupted")

    with patch.object(
        web_routes, "transcode_to_wav_bytes",
        new=AsyncMock(return_value=b"wav"),
    ), patch.object(
        diarize_mod, "diarize_uncached", new=fake_diarize_raises,
    ):
        _login(client, password)
        r = client.post(
            f"/tune/{sid}/diarize",
            data={
                "num_clusters": 2,
                "cluster_threshold": 0.4,
                "min_speech_s": 0.5,
                "min_silence_s": 0.5,
            },
        )
    assert r.status_code == 400
    assert "model file is corrupted" in r.text


def test_tune_apply_diarize_defaults_writes_env(
    client: TestClient, password: str, tmp_path, monkeypatch
):
    """The Save button writes the four diarize keys to .env and tells the
    user a restart is required."""
    env = tmp_path / ".env"
    env.write_text(
        "# existing comment\n"
        "OMILOG_USERNAME=test\n"
        "OMILOG_DIARIZATION_NUM_CLUSTERS=-1\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    _login(client, password)
    r = client.post(
        "/tune/apply-diarize-defaults",
        data={
            "num_clusters": 3,
            "cluster_threshold": 0.45,
            "min_speech_s": 0.7,
            "min_silence_s": 0.4,
        },
    )
    assert r.status_code == 200
    assert "Saved" in r.text
    assert "restart" in r.text.lower()

    written = env.read_text(encoding="utf-8")
    # Existing keys overwritten, others preserved.
    assert "OMILOG_USERNAME=test" in written
    assert "# existing comment" in written
    assert "OMILOG_DIARIZATION_NUM_CLUSTERS=3" in written
    assert "OMILOG_DIARIZATION_CLUSTER_THRESHOLD=0.45" in written
    assert "OMILOG_DIARIZATION_MIN_SPEECH_SECONDS=0.7" in written
    assert "OMILOG_DIARIZATION_MIN_SILENCE_SECONDS=0.4" in written
