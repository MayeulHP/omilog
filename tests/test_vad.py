"""Tests for the VAD / segmentation stage.

ffmpeg + filesystem effects are mocked everywhere except the
output-parser tests, which are pure-data.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlmodel import Session, select

from omilog.db import engine
from omilog.models import AudioSession, SessionStatus
from omilog.pipeline import runner, vad


# ──────────────────────────────────────────────────────────────────────────────
# silencedetect output parser
# ──────────────────────────────────────────────────────────────────────────────

_SAMPLE_FFMPEG_STDERR = """\
Input #0, ogg, from 'session.opus':
  Duration: 00:04:30.50, start: 0.000000, bitrate: 35 kb/s
[silencedetect @ 0x6000017a4140] silence_start: 0.000000
[silencedetect @ 0x6000017a4140] silence_end: 1.250000 | silence_duration: 1.250000
[silencedetect @ 0x6000017a4140] silence_start: 45.300000
[silencedetect @ 0x6000017a4140] silence_end: 130.420000 | silence_duration: 85.120000
[silencedetect @ 0x6000017a4140] silence_start: 230.000000
[silencedetect @ 0x6000017a4140] silence_end: 270.500000 | silence_duration: 40.500000
[silencedetect @ 0x6000017a4140] silence_start: 268.000000
"""


def test_parse_duration():
    assert vad.parse_duration(_SAMPLE_FFMPEG_STDERR) == 270.5


def test_parse_silencedetect_pairs():
    regions = vad.parse_silencedetect_output(_SAMPLE_FFMPEG_STDERR)
    # We get the 3 paired silences; the dangling final silence_start is ignored.
    assert regions == [
        (0.0, 1.25),
        (45.3, 130.42),
        (230.0, 270.5),
    ]


def test_parse_handles_empty_input():
    assert vad.parse_silencedetect_output("") == []
    assert vad.parse_duration("") is None


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation logic — deterministic, no I/O
# ──────────────────────────────────────────────────────────────────────────────

def test_segment_pure_speech_one_conversation():
    convs = vad.segment_by_silence_gaps(120.0, [], gap_threshold_s=60.0)
    assert convs == [(0.0, 120.0)]


def test_segment_pure_silence_empty():
    convs = vad.segment_by_silence_gaps(
        120.0, [(0.0, 120.0)], gap_threshold_s=60.0
    )
    assert convs == []


def test_segment_short_pauses_stay_in_one_conversation():
    # Three internal silences all shorter than the gap threshold.
    silences = [(10.0, 11.0), (40.0, 42.0), (80.0, 82.0)]
    convs = vad.segment_by_silence_gaps(120.0, silences, gap_threshold_s=60.0)
    assert convs == [(0.0, 120.0)]


def test_segment_long_silence_splits():
    # 85s silence between two speech blocks → two conversations.
    silences = [(45.0, 130.0)]
    convs = vad.segment_by_silence_gaps(200.0, silences, gap_threshold_s=60.0)
    assert convs == [(0.0, 45.0), (130.0, 200.0)]


def test_segment_trims_leading_and_trailing_silence():
    # Leading 90s silence + trailing 70s silence + one interior split.
    silences = [(0.0, 90.0), (200.0, 280.0), (400.0, 500.0)]
    convs = vad.segment_by_silence_gaps(500.0, silences, gap_threshold_s=60.0)
    # Speech-window is [90, 400]; the (200,280) gap is 80s → split.
    assert convs == [(90.0, 200.0), (280.0, 400.0)]


def test_segment_applies_pad():
    convs = vad.segment_by_silence_gaps(
        200.0, [(45.0, 130.0)], gap_threshold_s=60.0, pad_s=0.5
    )
    # Padded but clamped to [0, duration_s].
    assert convs == [(0.0, 45.5), (129.5, 200.0)]


def test_segment_multiple_long_silences():
    # Three speech blocks separated by two long gaps.
    silences = [(50.0, 130.0), (250.0, 320.0)]
    convs = vad.segment_by_silence_gaps(400.0, silences, gap_threshold_s=60.0)
    assert convs == [(0.0, 50.0), (130.0, 250.0), (320.0, 400.0)]


def test_segment_keeps_short_trailing_silence_inside_conversation():
    # Regression: a real-world capture had a brief trailing silence (a few
    # seconds while VAD missed quiet speech). The old logic trimmed the entire
    # tail off as 'trailing silence', losing the second utterance. The new
    # logic only trims a trailing silence that's itself >= gap_threshold.
    duration = 150.0
    silences = [(20.0, 25.0), (130.0, 149.0)]  # trailing silence is 19s, < gap
    convs = vad.segment_by_silence_gaps(duration, silences, gap_threshold_s=60.0)
    assert convs == [(0.0, 150.0)]


def test_segment_keeps_short_leading_silence_inside_conversation():
    # Same regression on the leading edge.
    duration = 150.0
    silences = [(0.0, 10.0)]  # leading silence 10s, well below gap
    convs = vad.segment_by_silence_gaps(duration, silences, gap_threshold_s=60.0)
    assert convs == [(0.0, 150.0)]


# ──────────────────────────────────────────────────────────────────────────────
# Runner: process_vad — mock vad.analyse + vad.extract_segment_to_opus
# ──────────────────────────────────────────────────────────────────────────────

def _insert_parent(audio_path: Path) -> UUID:
    sid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="test",
                audio_path=str(audio_path),
                codec="opus",
                started_at=datetime(2026, 6, 3, 10, 0, tzinfo=timezone.utc),
                status=SessionStatus.pending_vad,
            )
        )
        db.commit()
    return sid


def _get(sid: UUID) -> AudioSession:
    with Session(engine) as db:
        s = db.get(AudioSession, sid)
        assert s is not None
        return s


def _children_of(parent_id: UUID) -> list[AudioSession]:
    with Session(engine) as db:
        return list(
            db.exec(
                select(AudioSession).where(AudioSession.parent_id == parent_id)
            ).all()
        )


async def test_process_vad_all_silence_marks_silent_and_deletes(tmp_path):
    src = tmp_path / "parent.opus"
    src.write_bytes(b"fake")
    sid = _insert_parent(src)

    with patch.object(
        runner.vad,
        "analyse",
        new=AsyncMock(return_value=(120.0, [(0.0, 120.0)])),
    ), patch.object(runner.vad, "extract_segment_to_opus", new=AsyncMock()) as extract:
        await runner.process_vad(sid)

    assert _get(sid).status == SessionStatus.silent
    assert not src.exists()
    extract.assert_not_awaited()
    assert _children_of(sid) == []


async def test_process_vad_single_conversation_spawns_one_child(tmp_path):
    src = tmp_path / "parent.opus"
    src.write_bytes(b"fake")
    sid = _insert_parent(src)

    async def fake_extract(src_, dst_, **kwargs):
        # Pretend ffmpeg produced a child file.
        Path(dst_).write_bytes(b"opus-child-bytes")

    with patch.object(
        runner.vad,
        "analyse",
        new=AsyncMock(return_value=(120.0, [])),  # no silence → all speech
    ), patch.object(runner.vad, "extract_segment_to_opus", new=AsyncMock(side_effect=fake_extract)):
        await runner.process_vad(sid)

    parent = _get(sid)
    assert parent.status == SessionStatus.segmented
    assert not src.exists()  # parent freed

    children = _children_of(sid)
    assert len(children) == 1
    c = children[0]
    assert c.status == SessionStatus.pending_stt
    assert c.codec == "opus"
    assert c.parent_id == sid
    assert c.duration_s == pytest.approx(120.0)
    assert Path(c.audio_path).read_bytes() == b"opus-child-bytes"


async def test_process_vad_long_silence_splits_into_two_children(tmp_path):
    src = tmp_path / "parent.opus"
    src.write_bytes(b"fake")
    sid = _insert_parent(src)

    async def fake_extract(src_, dst_, **kwargs):
        Path(dst_).write_bytes(b"X")

    with patch.object(
        runner.vad,
        "analyse",
        new=AsyncMock(
            # 200s capture with a 90s silence in the middle → split.
            return_value=(200.0, [(50.0, 140.0)])
        ),
    ), patch.object(runner.vad, "extract_segment_to_opus", new=AsyncMock(side_effect=fake_extract)) as extract:
        await runner.process_vad(sid)

    parent = _get(sid)
    assert parent.status == SessionStatus.segmented
    assert not src.exists()

    children = _children_of(sid)
    assert len(children) == 2
    assert extract.await_count == 2
    # Children should appear in chronological order.
    children.sort(key=lambda c: c.started_at)
    # First child covers [0..50] (plus pad), second covers [140..200] (plus pad).
    assert children[0].duration_s == pytest.approx(50.0 + 0.4, abs=0.05)
    assert children[1].duration_s == pytest.approx(60.0 + 0.4, abs=0.05)
    assert all(c.status == SessionStatus.pending_stt for c in children)


async def test_process_vad_failure_keeps_parent(tmp_path):
    src = tmp_path / "parent.opus"
    src.write_bytes(b"fake")
    sid = _insert_parent(src)

    with patch.object(
        runner.vad, "analyse", new=AsyncMock(side_effect=runner.VADError("boom"))
    ):
        await runner.process_vad(sid)

    parent = _get(sid)
    assert parent.status == SessionStatus.failed
    assert "vad" in (parent.error_msg or "")
    # We do NOT delete the parent file on failure — it's needed for replay.
    assert src.exists()


async def test_process_vad_extract_failure_keeps_parent(tmp_path):
    src = tmp_path / "parent.opus"
    src.write_bytes(b"fake")
    sid = _insert_parent(src)

    with patch.object(
        runner.vad,
        "analyse",
        new=AsyncMock(return_value=(120.0, [])),
    ), patch.object(
        runner.vad,
        "extract_segment_to_opus",
        new=AsyncMock(side_effect=runner.VADError("ffmpeg failed")),
    ):
        await runner.process_vad(sid)

    assert _get(sid).status == SessionStatus.failed
    assert src.exists()


# ──────────────────────────────────────────────────────────────────────────────
# Priority: runner picks pending_vad before pending_stt
# ──────────────────────────────────────────────────────────────────────────────

async def test_runner_tick_prefers_vad_over_stt(tmp_path, monkeypatch):
    parent_path = tmp_path / "parent.opus"
    parent_path.write_bytes(b"fake")
    parent_id = _insert_parent(parent_path)

    # Also create a pending_stt session — should be ignored this tick.
    child_path = tmp_path / "child.opus"
    child_path.write_bytes(b"fake")
    child_id = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=child_id,
                user_id="test",
                audio_path=str(child_path),
                codec="opus",
                started_at=datetime(2026, 6, 3, 9, 0, tzinfo=timezone.utc),
                status=SessionStatus.pending_stt,
            )
        )
        db.commit()

    monkeypatch.setattr(
        runner.settings, "stt_base_url", "http://stt", raising=False
    )

    process_vad = AsyncMock()
    process_stt = AsyncMock()
    with patch.object(runner, "process_vad", new=process_vad), patch.object(
        runner, "process_stt", new=process_stt
    ):
        did = await runner._tick()

    assert did is True
    process_vad.assert_awaited_once_with(parent_id)
    process_stt.assert_not_awaited()
