"""Pipeline runner / STT client tests.

We mock both ffmpeg (via the transcode_to_wav_bytes function) and the whisper
HTTP call so these tests are self-contained — no GPU, no ffmpeg-on-PATH required.

A separate test exercises the real /api/audio/upload + runner path against an
in-process FastAPI app to confirm the end-to-end wiring.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from omilog.db import engine
from omilog.models import AudioSession, SessionStatus, Transcript
from omilog.pipeline import runner, stt
from omilog.pipeline.stt import STTResult


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _insert_pending_session(audio_path: Path) -> UUID:
    sid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="test",
                audio_path=str(audio_path),
                codec="opus",
                status=SessionStatus.pending_stt,
            )
        )
        db.commit()
    return sid


def _get_session(session_id: UUID) -> AudioSession:
    with Session(engine) as db:
        row = db.get(AudioSession, session_id)
        assert row is not None
        return row


def _get_transcripts_for(session_id: UUID) -> list[Transcript]:
    with Session(engine) as db:
        return list(
            db.exec(
                select(Transcript).where(Transcript.audio_session_id == session_id)
            ).all()
        )


# ──────────────────────────────────────────────────────────────────────────────
# Runner: happy path
# ──────────────────────────────────────────────────────────────────────────────

async def test_process_one_happy_path(tmp_path: Path, monkeypatch):
    audio = tmp_path / "session.opus"
    audio.write_bytes(b"fake-ogg-opus")
    sid = _insert_pending_session(audio)

    monkeypatch.setattr(
        runner.settings, "stt_base_url", "http://gpu.tailnet:8080", raising=False
    )

    with patch.object(
        runner, "transcode_to_wav_bytes", new=AsyncMock(return_value=b"RIFF...")
    ) as mock_transcode, patch.object(
        runner,
        "transcribe_wav",
        new=AsyncMock(
            return_value=STTResult(
                text="Bonjour, ceci est un test.",
                segments=[{"start": 0.0, "end": 2.0, "text": "Bonjour, ceci est un test."}],
                language="fr",
                raw={},
            )
        ),
    ) as mock_stt:
        await runner.process_stt(sid)

    mock_transcode.assert_awaited_once_with(audio)
    mock_stt.assert_awaited_once()

    sess = _get_session(sid)
    assert sess.status == SessionStatus.pending_llm
    assert sess.error_msg is None

    transcripts = _get_transcripts_for(sid)
    assert len(transcripts) == 1
    t = transcripts[0]
    assert t.text == "Bonjour, ceci est un test."
    assert t.language == "fr"
    segments = json.loads(t.segments_json)
    assert segments[0]["text"] == "Bonjour, ceci est un test."


# ──────────────────────────────────────────────────────────────────────────────
# Runner: failure modes
# ──────────────────────────────────────────────────────────────────────────────

async def test_process_one_missing_audio_file(tmp_path: Path):
    sid = _insert_pending_session(tmp_path / "does-not-exist.opus")
    await runner.process_stt(sid)
    sess = _get_session(sid)
    assert sess.status == SessionStatus.failed
    assert "audio file missing" in (sess.error_msg or "")


async def test_process_one_transcode_failure(tmp_path: Path, monkeypatch):
    audio = tmp_path / "broken.opus"
    audio.write_bytes(b"not really audio")
    sid = _insert_pending_session(audio)
    monkeypatch.setattr(
        runner.settings, "stt_base_url", "http://gpu.tailnet:8080", raising=False
    )

    async def boom(*_a, **_kw):
        raise runner.TranscodeError("ffmpeg exit=1: invalid data")

    with patch.object(runner, "transcode_to_wav_bytes", new=boom):
        await runner.process_stt(sid)

    sess = _get_session(sid)
    assert sess.status == SessionStatus.failed
    assert "ffmpeg" in (sess.error_msg or "")


async def test_process_one_stt_failure(tmp_path: Path, monkeypatch):
    audio = tmp_path / "ok.opus"
    audio.write_bytes(b"fake")
    sid = _insert_pending_session(audio)
    monkeypatch.setattr(
        runner.settings, "stt_base_url", "http://gpu.tailnet:8080", raising=False
    )

    with patch.object(
        runner, "transcode_to_wav_bytes", new=AsyncMock(return_value=b"WAV")
    ), patch.object(
        runner, "transcribe_wav", new=AsyncMock(side_effect=runner.STTError("502 bad gw"))
    ):
        await runner.process_stt(sid)

    sess = _get_session(sid)
    assert sess.status == SessionStatus.failed
    assert "stt" in (sess.error_msg or "")


# ──────────────────────────────────────────────────────────────────────────────
# STT client unit tests (mocks httpx, no network)
# ──────────────────────────────────────────────────────────────────────────────

async def test_transcribe_wav_parses_verbose_json():
    payload = {
        "text": "  Hello world. ",
        "segments": [{"start": 0, "end": 1, "text": "Hello world."}],
        "language": "en",
    }
    captured: dict = {}

    class _DummyResp:
        status_code = 200
        text = json.dumps(payload)

        def json(self):
            return payload

    class _DummyClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def post(self, url, files, data):
            captured["url"] = url
            captured["files"] = files
            captured["data"] = data
            return _DummyResp()

    with patch.object(stt.httpx, "AsyncClient", _DummyClient):
        result = await stt.transcribe_wav(
            b"RIFF",
            base_url="http://gpu.tailnet:8080",
            inference_path="/inference",
            language="fr",
        )

    assert captured["url"] == "http://gpu.tailnet:8080/inference"
    assert captured["data"]["language"] == "fr"
    assert captured["data"]["response_format"] == "verbose_json"
    assert result.text == "Hello world."
    assert result.language == "en"


async def test_transcribe_wav_empty_text_raises():
    payload = {"text": "", "segments": [], "language": "en"}

    class _DummyResp:
        status_code = 200
        text = "{}"

        def json(self):
            return payload

    class _DummyClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def post(self, *a, **kw):
            return _DummyResp()

    with patch.object(stt.httpx, "AsyncClient", _DummyClient), pytest.raises(
        stt.STTError
    ):
        await stt.transcribe_wav(b"RIFF", base_url="http://x")


async def test_transcribe_wav_disabled_when_base_url_blank():
    with pytest.raises(stt.STTError):
        await stt.transcribe_wav(b"RIFF", base_url="")


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end via /api/audio/upload
# ──────────────────────────────────────────────────────────────────────────────

def test_audio_upload_creates_pending_session(client: TestClient, auth_token: str, tmp_path):
    body = b"\x00" * 1024
    r = client.post(
        "/api/audio/upload",
        files={"file": ("clip.wav", body, "audio/wav")},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert r.status_code == 200, r.text
    out = r.json()
    sid = UUID(out["session_id"])
    # VAD is on by default, so upload lands in pending_vad now.
    assert out["status"] == "pending_vad"
    assert out["bytes"] == 1024

    sess = _get_session(sid)
    assert sess.codec == "wav"
    assert Path(sess.audio_path).exists()
    assert Path(sess.audio_path).read_bytes() == body


def test_audio_upload_skip_vad_goes_to_pending_stt(
    client: TestClient, auth_token: str
):
    r = client.post(
        "/api/audio/upload?skip_vad=true",
        files={"file": ("clip.wav", b"X" * 256, "audio/wav")},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "pending_stt"


def test_audio_upload_requires_auth(client: TestClient):
    r = client.post(
        "/api/audio/upload",
        files={"file": ("clip.wav", b"hi", "audio/wav")},
    )
    assert r.status_code == 401


def test_audio_upload_rejects_empty(client: TestClient, auth_token: str):
    r = client.post(
        "/api/audio/upload",
        files={"file": ("clip.wav", b"", "audio/wav")},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert r.status_code == 400
