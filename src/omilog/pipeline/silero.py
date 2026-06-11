"""Silero VAD backend — neural voice-activity detection via ONNX.

The ffmpeg-silencedetect backend in vad.py is an amplitude gate: in noisy
environments (street, café, TV in the background) the level never drops
below the threshold, so conversations don't split, leading/trailing noise
isn't trimmed, and non-speech audio reaches Whisper — which is the main
trigger for its hallucination loops. Silero VAD classifies *speech*
probability per 32 ms frame instead, which holds up in noise.

This module produces the exact same output shape as vad.analyse() —
``(duration_s, [(silence_start, silence_end), …])`` — so the downstream
segmentation (vad.segment_by_silence_gaps) and the /tune timeline work
unchanged. Dispatch lives in vad.analyse_with_backend(); failures here
fall back to silencedetect, never block the pipeline.

Deps are optional (`.[silero]` extra: onnxruntime + numpy, no torch). The
model file (~2 MB, MIT-licensed) is fetched once by
``scripts/download_silero_vad.py``. Targets the Silero VAD v5 ONNX
interface: 512-sample chunks @ 16 kHz with 64 samples of carried context,
a [2, 1, 128] recurrent state, and a scalar int64 sample rate.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from .vad import VADError

logger = logging.getLogger("omilog.pipeline.silero")

SILERO_IMPORT_ERROR: str | None = None
try:
    import numpy as _np
    import onnxruntime as _ort

    SILERO_AVAILABLE = True
except Exception as _e:  # noqa: BLE001 — any import-time failure means "off"
    _np = None  # type: ignore[assignment]
    _ort = None  # type: ignore[assignment]
    SILERO_AVAILABLE = False
    SILERO_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
    logger.debug("silero deps not available: %s", _e)


class SileroVADError(VADError):
    """Subclasses VADError so callers handling VAD failures uniformly catch
    both backends."""


SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # v5 fixed chunk @ 16 kHz → 32 ms per probability
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
_CONTEXT_SAMPLES = 64  # v5 prepends the previous chunk's tail to each input

# Module-scoped session cache, same pattern as diarize.py: the model is tiny
# but session construction isn't free, and the runner calls this per capture.
_SESSION_CACHE: object | None = None
_SESSION_CACHE_PATH: Path | None = None
_LOAD_LOCK = asyncio.Lock()


def _check_available() -> None:
    if not SILERO_AVAILABLE:
        raise SileroVADError(
            "silero VAD deps not installed. Run `uv sync --extra silero` "
            "(or `pip install -e '.[silero]'`)."
        )


def _check_model_path(model_path: Path) -> None:
    if not model_path.exists():
        raise SileroVADError(
            f"silero VAD model missing: {model_path}\n"
            "Run: .venv/bin/python scripts/download_silero_vad.py"
        )


def _build_session(model_path: str):
    """Synchronous; call from a thread executor. One intra-op thread is
    plenty for a 2 MB model and keeps a Pi's cores free for the web loop
    (same reasoning as diarization's num_threads cap)."""
    opts = _ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    return _ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )


async def _get_session(model_path: Path):
    global _SESSION_CACHE, _SESSION_CACHE_PATH
    async with _LOAD_LOCK:
        if _SESSION_CACHE is None or _SESSION_CACHE_PATH != model_path:
            loop = asyncio.get_event_loop()
            try:
                _SESSION_CACHE = await loop.run_in_executor(
                    None, _build_session, str(model_path)
                )
            except Exception as e:
                raise SileroVADError(f"failed to load silero model: {e}") from e
            _SESSION_CACHE_PATH = model_path
    return _SESSION_CACHE


def speech_probabilities(audio, session) -> list[float]:
    """Run the stateful model over float32 mono 16 kHz audio; one speech
    probability per 32 ms frame. Synchronous — call from an executor."""
    state = _np.zeros((2, 1, 128), dtype=_np.float32)
    context = _np.zeros((1, _CONTEXT_SAMPLES), dtype=_np.float32)
    sr = _np.array(SAMPLE_RATE, dtype=_np.int64)
    probs: list[float] = []
    for off in range(0, len(audio), FRAME_SAMPLES):
        chunk = audio[off : off + FRAME_SAMPLES]
        if len(chunk) < FRAME_SAMPLES:
            chunk = _np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
        x = _np.concatenate([context, chunk[None, :]], axis=1)
        try:
            out, state = session.run(None, {"input": x, "state": state, "sr": sr})
        except Exception as e:
            raise SileroVADError(f"silero inference failed at {off / SAMPLE_RATE:.1f}s: {e}") from e
        context = x[:, -_CONTEXT_SAMPLES:]
        probs.append(float(out[0, 0]))
    return probs


# ──────────────────────────────────────────────────────────────────────────────
# Probabilities → silences (pure Python — unit-testable without numpy/onnx)
# ──────────────────────────────────────────────────────────────────────────────


def speech_regions_from_probs(
    probs: list[float],
    *,
    frame_s: float = FRAME_SECONDS,
    threshold: float = 0.5,
    neg_threshold: float | None = None,
) -> list[tuple[float, float]]:
    """Hysteresis pass: speech starts when prob ≥ threshold and ends when it
    drops below ``neg_threshold`` (default threshold − 0.15, mirroring
    Silero's own iterator) — the gap between the two stops mid-utterance
    dips from chopping one sentence into many regions."""
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)
    regions: list[tuple[float, float]] = []
    start: float | None = None
    for i, p in enumerate(probs):
        t = i * frame_s
        if start is None:
            if p >= threshold:
                start = t
        elif p < neg_threshold:
            regions.append((start, t))
            start = None
    if start is not None:
        regions.append((start, len(probs) * frame_s))
    return regions


def silences_from_probs(
    probs: list[float],
    *,
    duration_s: float,
    frame_s: float = FRAME_SECONDS,
    threshold: float = 0.5,
    neg_threshold: float | None = None,
    min_speech_s: float = 0.3,
    min_silence_s: float = 0.5,
) -> list[tuple[float, float]]:
    """Convert per-frame speech probabilities into silencedetect-shaped
    silence intervals over [0, duration_s].

    Same reporting contract as ffmpeg silencedetect with ``d=min_silence_s``:
    only silences at least that long are reported. Two extra cleanups the
    amplitude gate can't do:
      * speech regions closer than ``min_silence_s`` are bridged (one
        utterance, not many), and
      * surviving regions shorter than ``min_speech_s`` are discarded — an
        isolated cough or door slam no longer splits a long silence in two,
        which matters because the downstream conversation-gap check only
        fires on single silences ≥ the gap threshold.
    """
    if duration_s <= 0:
        return []
    regions = speech_regions_from_probs(
        probs, frame_s=frame_s, threshold=threshold, neg_threshold=neg_threshold
    )
    # Bridge sub-reportable gaps first so a real utterance flickering around
    # the threshold survives the min-speech filter as one region.
    merged: list[list[float]] = []
    for s, e in regions:
        if merged and s - merged[-1][1] < min_silence_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    kept = [
        (s, min(e, duration_s))
        for s, e in merged
        if (min(e, duration_s) - s) >= min_speech_s
    ]
    if not kept:
        return [(0.0, duration_s)]

    silences: list[tuple[float, float]] = []
    if kept[0][0] >= min_silence_s:
        silences.append((0.0, kept[0][0]))
    for (_, prev_end), (next_start, _) in zip(kept, kept[1:]):
        silences.append((prev_end, next_start))  # ≥ min_silence_s by construction
    if duration_s - kept[-1][1] >= min_silence_s:
        silences.append((kept[-1][1], duration_s))
    return silences


# ──────────────────────────────────────────────────────────────────────────────
# Entry point — same return shape as vad.analyse()
# ──────────────────────────────────────────────────────────────────────────────


async def _transcode_to_pcm(src: Path, *, timeout_s: float) -> bytes:
    """Decode to raw s16le 16 kHz mono. Raw PCM rather than WAV-to-pipe
    because ffmpeg can't seek back to patch RIFF sizes on a pipe, and we'd
    rather not parse a header with lying length fields."""
    if shutil.which("ffmpeg") is None:
        raise SileroVADError("ffmpeg not found on PATH")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(src),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-f", "s16le",
        "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError as e:
        proc.kill()
        await proc.wait()
        raise SileroVADError(f"ffmpeg decode timed out after {timeout_s}s") from e
    if proc.returncode != 0:
        raise SileroVADError(
            f"ffmpeg decode exit={proc.returncode}: "
            f"{stderr.decode(errors='replace')[:300]}"
        )
    if not stdout:
        raise SileroVADError("ffmpeg produced no audio — input may be empty or corrupt")
    return stdout


async def analyse(
    src: Path,
    *,
    model_path: Path,
    threshold: float = 0.5,
    min_speech_s: float = 0.3,
    min_silence_s: float = 0.5,
    timeout_s: float = 300.0,
) -> tuple[float, list[tuple[float, float]]]:
    """Return (duration_s, [(silence_start, silence_end), …]) — drop-in for
    vad.analyse(). Raises SileroVADError (a VADError) on any failure."""
    _check_available()
    _check_model_path(model_path)
    pcm = await _transcode_to_pcm(src, timeout_s=timeout_s)
    audio = _np.frombuffer(pcm, dtype=_np.int16).astype(_np.float32) / 32768.0
    duration_s = len(audio) / SAMPLE_RATE

    session = await _get_session(model_path)
    loop = asyncio.get_event_loop()
    probs = await loop.run_in_executor(None, speech_probabilities, audio, session)
    silences = silences_from_probs(
        probs,
        duration_s=duration_s,
        threshold=threshold,
        min_speech_s=min_speech_s,
        min_silence_s=min_silence_s,
    )
    return duration_s, silences
