"""Diarization merge logic + runner integration (pyannote mocked).

The pyannote-audio dep is intentionally not in our test env, so the runner
path is exercised against a mocked diarize_mod.diarize call.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlmodel import Session, select

from omilog.db import engine
from omilog.models import AudioSession, SessionStatus, Transcript
from omilog.pipeline import diarize as diarize_mod
from omilog.pipeline import runner
from omilog.pipeline.diarize import DiarizationError
from omilog.pipeline.stt import STTResult


# ──────────────────────────────────────────────────────────────────────────────
# Merge primitives — pure data, no pyannote needed
# ──────────────────────────────────────────────────────────────────────────────

def test_overlap_basic():
    assert diarize_mod._overlap(0, 10, 5, 15) == 5
    assert diarize_mod._overlap(0, 5, 10, 15) == 0  # disjoint
    assert diarize_mod._overlap(0, 10, 2, 8) == 6   # b inside a
    assert diarize_mod._overlap(5, 15, 0, 20) == 10 # a inside b


def test_assign_speakers_simple_two_speakers():
    whisper = [
        {"start": 0.0, "end": 5.0, "text": "Hi"},
        {"start": 5.5, "end": 10.0, "text": "Hello"},
    ]
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ]
    out = diarize_mod.assign_speakers_to_segments(whisper, turns)
    assert out[0]["speaker"] == "SPEAKER_00"
    assert out[1]["speaker"] == "SPEAKER_01"


def test_assign_speakers_picks_max_overlap():
    # Whisper segment spans two pyannote turns; the one with more overlap wins.
    whisper = [{"start": 0.0, "end": 10.0, "text": "long"}]
    turns = [
        {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},   # 3s
        {"start": 3.0, "end": 10.0, "speaker": "SPEAKER_01"},  # 7s
    ]
    out = diarize_mod.assign_speakers_to_segments(whisper, turns)
    assert out[0]["speaker"] == "SPEAKER_01"


def test_assign_speakers_leaves_unmatched_alone():
    whisper = [{"start": 0.0, "end": 5.0, "text": "x"}]
    turns = [{"start": 100.0, "end": 110.0, "speaker": "SPEAKER_00"}]
    out = diarize_mod.assign_speakers_to_segments(whisper, turns)
    assert "speaker" not in out[0]


def test_relabel_user_is_longest_talker():
    segments = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},   # 5s
        {"start": 5.0, "end": 8.0, "speaker": "SPEAKER_01"},   # 3s
        {"start": 8.0, "end": 10.0, "speaker": "SPEAKER_00"},  # 2s
        {"start": 10.0, "end": 11.0, "speaker": "SPEAKER_02"}, # 1s
    ]
    out = diarize_mod.relabel_user_and_others(segments)
    # SPEAKER_00: 5+2 = 7s (longest) → USER
    # SPEAKER_01: 3s → S1
    # SPEAKER_02: 1s → S2
    assert out[0]["speaker"] == "USER"
    assert out[1]["speaker"] == "S1"
    assert out[2]["speaker"] == "USER"
    assert out[3]["speaker"] == "S2"


def test_relabel_handles_no_diarization():
    segments = [{"start": 0.0, "end": 5.0, "text": "x"}]
    out = diarize_mod.relabel_user_and_others(segments)
    # Nothing added, nothing breaks.
    assert "speaker" not in out[0]


def test_relabel_single_speaker_becomes_user():
    segments = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_03"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_03"},
    ]
    out = diarize_mod.relabel_user_and_others(segments)
    assert all(s["speaker"] == "USER" for s in out)


# ──────────────────────────────────────────────────────────────────────────────
# Get-pipeline / diarize errors when pyannote isn't there or config is missing
# ──────────────────────────────────────────────────────────────────────────────

async def test_get_pipeline_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(diarize_mod, "PYANNOTE_AVAILABLE", False, raising=False)
    with pytest.raises(DiarizationError, match="not installed"):
        await diarize_mod.get_pipeline("any-token", "any/model")


async def test_get_pipeline_raises_without_token(monkeypatch):
    monkeypatch.setattr(diarize_mod, "PYANNOTE_AVAILABLE", True, raising=False)
    with pytest.raises(DiarizationError, match="HF_TOKEN"):
        await diarize_mod.get_pipeline("", "any/model")


# ──────────────────────────────────────────────────────────────────────────────
# Runner: STT happy path with diarization mocked in
# ──────────────────────────────────────────────────────────────────────────────

def _insert_pending_stt(audio_path: Path) -> UUID:
    sid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="test",
                audio_path=str(audio_path),
                codec="opus",
                started_at=datetime(2026, 6, 3, 10, 0, tzinfo=timezone.utc),
                status=SessionStatus.pending_stt,
            )
        )
        db.commit()
    return sid


async def test_runner_stt_with_diarization_enabled(tmp_path: Path, monkeypatch):
    audio = tmp_path / "session.opus"
    audio.write_bytes(b"fake")
    sid = _insert_pending_stt(audio)

    monkeypatch.setattr(
        runner.settings, "stt_base_url", "http://stt", raising=False
    )
    monkeypatch.setattr(
        runner.settings, "diarization_enabled", True, raising=False
    )

    whisper_segments = [
        {"start": 0.0, "end": 4.0, "text": "Salut Marie."},
        {"start": 4.0, "end": 8.0, "text": "On se voit demain ?"},
        {"start": 8.0, "end": 12.0, "text": "Oui, à la Bastille."},
    ]
    fake_turns = [
        {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},  # USER candidate
        {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_01"},
        {"start": 8.0, "end": 12.0, "speaker": "SPEAKER_00"},  # USER wins talk-time
    ]
    with patch.object(
        runner, "transcode_to_wav_bytes", new=AsyncMock(return_value=b"WAV")
    ), patch.object(
        runner,
        "transcribe_wav",
        new=AsyncMock(
            return_value=STTResult(
                text="Salut Marie. On se voit demain ? Oui, à la Bastille.",
                segments=whisper_segments,
                language="fr",
                raw={},
            )
        ),
    ), patch.object(
        runner.diarize_mod, "diarize", new=AsyncMock(return_value=fake_turns)
    ):
        await runner.process_stt(sid)

    with Session(engine) as db:
        t = db.exec(
            select(Transcript).where(Transcript.audio_session_id == sid)
        ).first()
        assert t is not None
        segments = json.loads(t.segments_json)
    speakers = [s["speaker"] for s in segments]
    # SPEAKER_00 (8s total) became USER, SPEAKER_01 (4s) became S1.
    assert speakers == ["USER", "S1", "USER"]


async def test_runner_stt_continues_when_diarization_fails(
    tmp_path: Path, monkeypatch
):
    audio = tmp_path / "session.opus"
    audio.write_bytes(b"fake")
    sid = _insert_pending_stt(audio)

    monkeypatch.setattr(
        runner.settings, "stt_base_url", "http://stt", raising=False
    )
    monkeypatch.setattr(
        runner.settings, "diarization_enabled", True, raising=False
    )

    whisper_segments = [{"start": 0.0, "end": 5.0, "text": "Hi"}]
    with patch.object(
        runner, "transcode_to_wav_bytes", new=AsyncMock(return_value=b"WAV")
    ), patch.object(
        runner,
        "transcribe_wav",
        new=AsyncMock(
            return_value=STTResult(
                text="Hi", segments=whisper_segments, language="en", raw={}
            )
        ),
    ), patch.object(
        runner.diarize_mod,
        "diarize",
        new=AsyncMock(side_effect=DiarizationError("boom")),
    ):
        await runner.process_stt(sid)

    # Pipeline still succeeded: transcript written, status moved on.
    with Session(engine) as db:
        sess = db.get(AudioSession, sid)
        t = db.exec(
            select(Transcript).where(Transcript.audio_session_id == sid)
        ).first()
    assert sess.status == SessionStatus.pending_llm
    assert t is not None
    segments = json.loads(t.segments_json)
    # No speaker labels added when diarization fails.
    assert "speaker" not in segments[0]


async def test_runner_stt_skips_diarization_when_disabled(
    tmp_path: Path, monkeypatch
):
    audio = tmp_path / "session.opus"
    audio.write_bytes(b"fake")
    sid = _insert_pending_stt(audio)

    monkeypatch.setattr(
        runner.settings, "stt_base_url", "http://stt", raising=False
    )
    monkeypatch.setattr(
        runner.settings, "diarization_enabled", False, raising=False
    )

    whisper_segments = [{"start": 0.0, "end": 5.0, "text": "Hi"}]
    diarize_mock = AsyncMock(return_value=[])

    with patch.object(
        runner, "transcode_to_wav_bytes", new=AsyncMock(return_value=b"WAV")
    ), patch.object(
        runner,
        "transcribe_wav",
        new=AsyncMock(
            return_value=STTResult(
                text="Hi", segments=whisper_segments, language="en", raw={}
            )
        ),
    ), patch.object(runner.diarize_mod, "diarize", new=diarize_mock):
        await runner.process_stt(sid)

    diarize_mock.assert_not_awaited()


# ──────────────────────────────────────────────────────────────────────────────
# Extract: format and prompt include speaker labels
# ──────────────────────────────────────────────────────────────────────────────

def test_format_segments_includes_speaker_when_present():
    from omilog.pipeline import extract

    segments = [
        {"start": 0.0, "text": "Salut", "speaker": "USER"},
        {"start": 5.0, "text": "Bonjour", "speaker": "S1"},
        {"start": 10.0, "text": "(unlabeled)"},  # no speaker key
    ]
    out = extract._format_segments(segments)
    assert "[00:00] [USER] Salut" in out
    assert "[00:05] [S1] Bonjour" in out
    assert "[00:10] (unlabeled)" in out  # no bracket when missing


def test_system_prompt_references_speaker_labels():
    from omilog.pipeline.extract import SYSTEM_PROMPT

    assert "[USER]" in SYSTEM_PROMPT
    assert "[S1]" in SYSTEM_PROMPT
