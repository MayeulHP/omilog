"""Speaker diarization via sherpa-onnx.

100% local — audio never leaves this process. The models are ONNX files
downloaded once from sherpa-onnx GitHub releases via
`scripts/download_diarization_models.py`; runtime needs no internet access.

We use the same pyannote-segmentation-3.0 model as Phase 4's first pass, but
ONNX-converted and combined with NeMo TitaNet for speaker embeddings — same
quality, no torch dep, no HuggingFace dance.

Heuristic: the speaker with the largest cumulative talk time in a conversation
is **the user** (wearable-mic geometry — your voice is the loudest signal from
chest position). Other speakers become S1, S2, … in talk-time-descending order,
stable per conversation.

The sherpa-onnx dep is **optional** (`.[diarization]` extra, ~80 MB). We
try-import at module load; if absent or model files missing, the runner skips
diarization gracefully — transcripts still flow to LLM.
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("omilog.pipeline.diarize")


DIARIZATION_IMPORT_ERROR: str | None = None
try:
    import sherpa_onnx as _sherpa
    import soundfile as _sf
    DIARIZATION_AVAILABLE = True
except Exception as _e:  # noqa: BLE001 — any import-time failure means "off"
    _sherpa = None  # type: ignore[assignment]
    _sf = None  # type: ignore[assignment]
    DIARIZATION_AVAILABLE = False
    # Stash the message so the runner's startup logging can surface the real
    # reason (most common on a fresh Pi: libsndfile1 missing system-wide so
    # `import soundfile` blows up despite the wheel being installed).
    DIARIZATION_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
    logger.debug("diarization deps not available: %s", _e)


class DiarizationError(RuntimeError):
    pass


# Module-scoped cache: ~50 MB of ONNX models, ~5 s to load. asyncio.Lock guards
# the first load so concurrent calls don't load twice.
_DIARIZER_CACHE: Any = None
_LOAD_LOCK = asyncio.Lock()


def _check_available() -> None:
    if not DIARIZATION_AVAILABLE:
        raise DiarizationError(
            "sherpa-onnx not installed. Run `uv sync --extra diarization` "
            "(or `pip install -e '.[diarization]'`)."
        )


def _check_model_paths(seg: Path, emb: Path) -> None:
    missing: list[str] = []
    if not seg.exists():
        missing.append(f"segmentation: {seg}")
    if not emb.exists():
        missing.append(f"embedding: {emb}")
    if missing:
        raise DiarizationError(
            "missing diarization model file(s):\n  "
            + "\n  ".join(missing)
            + "\nRun: .venv/bin/python scripts/download_diarization_models.py"
        )


def _build_diarizer(
    seg_path: str,
    emb_path: str,
    min_speech_s: float,
    min_silence_s: float,
):
    """Construct a sherpa-onnx OfflineSpeakerDiarization. Synchronous; call
    from a thread executor."""
    cfg = _sherpa.OfflineSpeakerDiarizationConfig(
        segmentation=_sherpa.OfflineSpeakerSegmentationModelConfig(
            pyannote=_sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=seg_path,
            ),
        ),
        embedding=_sherpa.SpeakerEmbeddingExtractorConfig(model=emb_path),
        clustering=_sherpa.FastClusteringConfig(num_clusters=-1),
        min_duration_on=min_speech_s,
        min_duration_off=min_silence_s,
    )
    return _sherpa.OfflineSpeakerDiarization(cfg)


async def get_diarizer(
    seg_path: Path,
    emb_path: Path,
    *,
    min_speech_s: float,
    min_silence_s: float,
):
    """Lazy-load and cache. Loading touches disk and allocates ONNX runtime
    state — do it once per process."""
    _check_available()
    _check_model_paths(seg_path, emb_path)
    global _DIARIZER_CACHE
    async with _LOAD_LOCK:
        if _DIARIZER_CACHE is None:
            loop = asyncio.get_event_loop()
            try:
                _DIARIZER_CACHE = await loop.run_in_executor(
                    None,
                    _build_diarizer,
                    str(seg_path),
                    str(emb_path),
                    min_speech_s,
                    min_silence_s,
                )
            except Exception as e:
                raise DiarizationError(
                    f"failed to build sherpa-onnx diarizer: {e}"
                ) from e
    return _DIARIZER_CACHE


def _read_wav_mono_16k(wav_input) -> Any:
    """Read WAV bytes-or-path → float32 mono numpy array at 16 kHz.

    sherpa-onnx wants float32 mono. We assume the caller hands us 16 kHz audio
    (the STT step already decodes everything to that rate); if it's not, we
    raise rather than silently resample.
    """
    if isinstance(wav_input, (bytes, bytearray)):
        src = io.BytesIO(wav_input)
    else:
        src = str(wav_input)
    audio, sr = _sf.read(src, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        raise DiarizationError(
            f"diarizer expects 16 kHz WAV, got {sr} Hz — pre-resample upstream"
        )
    return audio


async def diarize(
    wav_input,
    *,
    seg_path: Path,
    emb_path: Path,
    min_speech_s: float = 0.3,
    min_silence_s: float = 0.5,
) -> list[dict[str, Any]]:
    """Run diarization on a WAV file path *or* WAV bytes.

    Returns turn dicts: [{'start': float_s, 'end': float_s, 'speaker': 'SPEAKER_00'}, …]
    """
    diarizer = await get_diarizer(
        seg_path,
        emb_path,
        min_speech_s=min_speech_s,
        min_silence_s=min_silence_s,
    )
    audio = _read_wav_mono_16k(wav_input)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: diarizer.process(audio).sort_by_start_time(),
        )
    except Exception as e:
        raise DiarizationError(f"sherpa-onnx inference failed: {e}") from e

    turns: list[dict[str, Any]] = []
    for r in result:
        turns.append(
            {
                "start": float(r.start),
                "end": float(r.end),
                "speaker": f"SPEAKER_{int(r.speaker):02d}",
            }
        )
    return turns


# ──────────────────────────────────────────────────────────────────────────────
# Merge with whisper transcript segments + user-heuristic relabel
# (unchanged from the pyannote version — pure data, no backend coupling)
# ──────────────────────────────────────────────────────────────────────────────

def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers_to_segments(
    whisper_segments: list[dict[str, Any]],
    diarization_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for seg in whisper_segments:
        ws_start = float(seg.get("start", 0) or 0)
        ws_end = float(seg.get("end", ws_start) or ws_start)
        if ws_end <= ws_start:
            continue
        best_speaker: str | None = None
        best_overlap = 0.0
        for turn in diarization_turns:
            ov = _overlap(ws_start, ws_end, turn["start"], turn["end"])
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = turn["speaker"]
        if best_speaker is not None:
            seg["speaker"] = best_speaker
    return whisper_segments


def relabel_user_and_others(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    duration_by_speaker: dict[str, float] = {}
    for seg in segments:
        sp = seg.get("speaker")
        if not sp:
            continue
        start = float(seg.get("start", 0) or 0)
        end = float(seg.get("end", start) or start)
        if end <= start:
            continue
        duration_by_speaker[sp] = duration_by_speaker.get(sp, 0.0) + (end - start)

    if not duration_by_speaker:
        return segments

    ranked = sorted(duration_by_speaker.items(), key=lambda kv: -kv[1])
    remap: dict[str, str] = {ranked[0][0]: "USER"}
    for i, (sp, _) in enumerate(ranked[1:], start=1):
        remap[sp] = f"S{i}"

    for seg in segments:
        sp = seg.get("speaker")
        if sp in remap:
            seg["speaker"] = remap[sp]
    return segments
