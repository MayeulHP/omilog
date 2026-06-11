"""Silero VAD backend: probability→silence conversion (pure logic) and the
vad.analyse_with_backend dispatcher (fallback behavior).

No onnxruntime/numpy needed — the prob logic is pure Python and the
dispatcher tests mock both backends, matching the repo's tests-run-anywhere
convention.
"""

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import omilog.pipeline.silero as silero_mod
import omilog.pipeline.vad as vad_mod
from omilog.pipeline.silero import (
    SileroVADError,
    silences_from_probs,
    speech_regions_from_probs,
)

# ──────────────────────────────────────────────────────────────────────────────
# speech_regions_from_probs — hysteresis
# ──────────────────────────────────────────────────────────────────────────────


def test_regions_basic_block():
    probs = [0.0] * 10 + [0.9] * 10 + [0.0] * 10
    regions = speech_regions_from_probs(probs, frame_s=0.1, threshold=0.5)
    assert regions == [(pytest.approx(1.0), pytest.approx(2.0))]


def test_regions_hysteresis_keeps_mid_utterance_dips():
    # Dip to 0.4 stays above the exit threshold (0.5 - 0.15 = 0.35), so the
    # utterance is NOT chopped in two.
    probs = [0.9] * 5 + [0.4] * 3 + [0.9] * 5 + [0.1] * 5
    regions = speech_regions_from_probs(probs, frame_s=0.1, threshold=0.5)
    assert regions == [(pytest.approx(0.0), pytest.approx(1.3))]


def test_regions_open_ended_speech_closes_at_audio_end():
    probs = [0.0] * 5 + [0.9] * 5
    regions = speech_regions_from_probs(probs, frame_s=0.1, threshold=0.5)
    assert regions == [(pytest.approx(0.5), pytest.approx(1.0))]


# ──────────────────────────────────────────────────────────────────────────────
# silences_from_probs — silencedetect-shaped output
# ──────────────────────────────────────────────────────────────────────────────


def _silences(probs, duration_s, **kw):
    defaults = dict(frame_s=0.1, threshold=0.5, min_speech_s=0.3, min_silence_s=0.5)
    defaults.update(kw)
    return silences_from_probs(probs, duration_s=duration_s, **defaults)


def test_silences_speech_block_reports_lead_and_trail():
    probs = [0.0] * 10 + [0.9] * 10 + [0.0] * 10
    assert _silences(probs, 3.0) == [
        (0.0, pytest.approx(1.0)),
        (pytest.approx(2.0), 3.0),
    ]


def test_silences_all_silence_is_one_full_span():
    assert _silences([0.0] * 30, 3.0) == [(0.0, 3.0)]


def test_silences_all_speech_reports_nothing():
    assert _silences([0.9] * 10, 1.0) == []


def test_silences_zero_duration():
    assert _silences([], 0.0) == []


def test_silences_short_blip_is_not_speech():
    # 0.2 s of "speech" (< min_speech 0.3) → whole capture is silence.
    probs = [0.0] * 10 + [0.9] * 2 + [0.0] * 10
    assert _silences(probs, 2.2) == [(0.0, 2.2)]


def test_silences_sub_reportable_gap_is_bridged():
    # Two speech runs 0.3 s apart (< min_silence 0.5) are one utterance —
    # no interior silence reported.
    probs = [0.9] * 10 + [0.0] * 3 + [0.9] * 10
    assert _silences(probs, 2.3) == []


def test_silences_blip_cannot_split_a_long_silence():
    # The property silencedetect lacks: a 0.2 s cough inside a 6 s silence
    # does NOT split it into two sub-gap silences. One long silence comes
    # back, so the conversation-gap boundary check still fires.
    probs = [0.9] * 10 + [0.0] * 30 + [0.9] * 2 + [0.0] * 30 + [0.9] * 10
    assert _silences(probs, 8.2) == [(pytest.approx(1.0), pytest.approx(7.2))]


def test_silences_leading_below_min_silence_not_reported():
    probs = [0.0] * 3 + [0.9] * 10  # 0.3 s lead < min_silence 0.5
    assert _silences(probs, 1.3) == []


# ──────────────────────────────────────────────────────────────────────────────
# analyse_with_backend — dispatch + fail-open fallback
# ──────────────────────────────────────────────────────────────────────────────


def test_silero_error_is_a_vad_error():
    assert issubclass(SileroVADError, vad_mod.VADError)


async def test_dispatch_silencedetect(monkeypatch):
    fake = AsyncMock(return_value=(10.0, [(1.0, 2.0)]))
    monkeypatch.setattr(vad_mod, "analyse", fake)
    duration, silences, backend = await vad_mod.analyse_with_backend(
        Path("x.opus"), backend="silencedetect", threshold_db=-40, min_silence_s=0.5
    )
    assert (duration, silences, backend) == (10.0, [(1.0, 2.0)], "silencedetect")
    fake.assert_awaited_once()


async def test_dispatch_silero_happy_path(monkeypatch, tmp_path):
    model = tmp_path / "silero_vad.onnx"
    model.write_bytes(b"\x00fake")
    monkeypatch.setattr(silero_mod, "SILERO_AVAILABLE", True)
    fake_silero = AsyncMock(return_value=(8.0, [(0.0, 3.0)]))
    monkeypatch.setattr(silero_mod, "analyse", fake_silero)
    fake_ffmpeg = AsyncMock()
    monkeypatch.setattr(vad_mod, "analyse", fake_ffmpeg)

    duration, silences, backend = await vad_mod.analyse_with_backend(
        Path("x.opus"),
        backend="silero",
        threshold_db=-40,
        min_silence_s=0.5,
        silero_model_path=model,
        silero_threshold=0.6,
        silero_min_speech_s=0.4,
    )
    assert (duration, silences, backend) == (8.0, [(0.0, 3.0)], "silero")
    fake_ffmpeg.assert_not_awaited()
    kwargs = fake_silero.await_args.kwargs
    assert kwargs["threshold"] == 0.6
    assert kwargs["min_speech_s"] == 0.4
    assert kwargs["min_silence_s"] == 0.5


async def test_dispatch_silero_deps_missing_falls_back(monkeypatch, tmp_path):
    model = tmp_path / "silero_vad.onnx"
    model.write_bytes(b"\x00fake")
    monkeypatch.setattr(silero_mod, "SILERO_AVAILABLE", False)
    monkeypatch.setattr(silero_mod, "SILERO_IMPORT_ERROR", "ModuleNotFoundError: onnxruntime")
    fake_ffmpeg = AsyncMock(return_value=(10.0, []))
    monkeypatch.setattr(vad_mod, "analyse", fake_ffmpeg)

    _, _, backend = await vad_mod.analyse_with_backend(
        Path("x.opus"),
        backend="silero",
        threshold_db=-40,
        min_silence_s=0.5,
        silero_model_path=model,
    )
    assert backend == "silencedetect"
    fake_ffmpeg.assert_awaited_once()


async def test_dispatch_silero_model_missing_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(silero_mod, "SILERO_AVAILABLE", True)
    fake_ffmpeg = AsyncMock(return_value=(10.0, []))
    monkeypatch.setattr(vad_mod, "analyse", fake_ffmpeg)

    _, _, backend = await vad_mod.analyse_with_backend(
        Path("x.opus"),
        backend="silero",
        threshold_db=-40,
        min_silence_s=0.5,
        silero_model_path=tmp_path / "nope.onnx",
    )
    assert backend == "silencedetect"


async def test_dispatch_silero_inference_failure_falls_back(monkeypatch, tmp_path):
    model = tmp_path / "silero_vad.onnx"
    model.write_bytes(b"\x00fake")
    monkeypatch.setattr(silero_mod, "SILERO_AVAILABLE", True)
    monkeypatch.setattr(
        silero_mod, "analyse", AsyncMock(side_effect=SileroVADError("boom"))
    )
    fake_ffmpeg = AsyncMock(return_value=(10.0, []))
    monkeypatch.setattr(vad_mod, "analyse", fake_ffmpeg)

    _, _, backend = await vad_mod.analyse_with_backend(
        Path("x.opus"),
        backend="silero",
        threshold_db=-40,
        min_silence_s=0.5,
        silero_model_path=model,
    )
    assert backend == "silencedetect"
    fake_ffmpeg.assert_awaited_once()


async def test_dispatch_unknown_backend_uses_silencedetect(monkeypatch):
    fake = AsyncMock(return_value=(10.0, []))
    monkeypatch.setattr(vad_mod, "analyse", fake)
    _, _, backend = await vad_mod.analyse_with_backend(
        Path("x.opus"), backend="whatever", threshold_db=-40, min_silence_s=0.5
    )
    assert backend == "silencedetect"
