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
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("omilog.pipeline.diarize")


# Note: an earlier version of this file used a ctypes.CDLL preload to make
# onnxruntime's symbols globally visible. That helped when sherpa-onnx
# resolved dependencies via the dynamic linker but not when its C extension
# did its own explicit dlopen("libonnxruntime.so") (which Linux can only
# resolve against a file on a search path, not against an already-loaded
# library). On aarch64 Linux, the right fix is start.sh adding sherpa-onnx's
# *own* bundled lib dir to LD_LIBRARY_PATH (ABI-matched), not loading the
# pip `onnxruntime` package's copy (which is built against a different
# onnxruntime version and produces "VERS_1.X not found" errors).

DIARIZATION_IMPORT_ERROR: str | None = None
try:
    import sherpa_onnx as _sherpa
    import soundfile as _sf
    import numpy as _np
    DIARIZATION_AVAILABLE = True
except Exception as _e:  # noqa: BLE001 — any import-time failure means "off"
    _sherpa = None  # type: ignore[assignment]
    _sf = None  # type: ignore[assignment]
    _np = None  # type: ignore[assignment]
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


# Note: _read_wav_mono_16k is defined further down (used by both diarize()
# and the embedding compute path).


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


# ──────────────────────────────────────────────────────────────────────────────
# Speaker embeddings (Phase 5: cross-conversation linking)
# ──────────────────────────────────────────────────────────────────────────────


_EMBEDDING_EXTRACTOR: Any = None


def _get_embedding_extractor(emb_path: str):
    """Lazy-load and cache the sherpa-onnx SpeakerEmbeddingExtractor.

    The OfflineSpeakerDiarization pipeline uses one internally but doesn't
    expose its outputs, so we instantiate a separate one against the same
    model file and call it on per-segment audio slices.
    """
    global _EMBEDDING_EXTRACTOR
    if _EMBEDDING_EXTRACTOR is None:
        cfg = _sherpa.SpeakerEmbeddingExtractorConfig(model=emb_path)
        _EMBEDDING_EXTRACTOR = _sherpa.SpeakerEmbeddingExtractor(cfg)
    return _EMBEDDING_EXTRACTOR


def compute_speaker_embeddings(
    wav_input,
    segments: list[dict[str, Any]],
    *,
    emb_path: Path,
    min_segment_seconds: float = 0.5,
) -> dict[str, list[float]]:
    """For each unique speaker label in segments, return one averaged embedding.

    Iterates segments, extracts the audio slice for each, computes its
    embedding, then averages per label. Segments shorter than
    ``min_segment_seconds`` are skipped (NeMo TitaNet needs at least ~0.5 s
    of audio to produce a stable embedding).

    Returns ``{speaker_label: list_of_floats}``. Labels with no successful
    embedding (all their segments were too short or errored) are omitted.
    """
    _check_available()
    audio = _read_wav_mono_16k(wav_input)
    sr = 16000  # _read_wav_mono_16k enforces this

    extractor = _get_embedding_extractor(str(emb_path))

    per_label: dict[str, list] = {}
    min_samples = int(min_segment_seconds * sr)
    for seg in segments:
        label = seg.get("speaker")
        if not label:
            continue
        start_sample = int(float(seg.get("start", 0) or 0) * sr)
        end_sample = int(float(seg.get("end", 0) or 0) * sr)
        if end_sample - start_sample < min_samples:
            continue
        slice_audio = audio[start_sample:end_sample]
        try:
            stream = extractor.create_stream()
            stream.accept_waveform(sr, slice_audio)
            stream.input_finished()
            emb = extractor.compute(stream)
        except Exception as e:  # noqa: BLE001
            logger.debug("embedding compute failed for label=%s: %s", label, e)
            continue
        arr = _np.array(emb, dtype=_np.float32)
        per_label.setdefault(label, []).append(arr)

    return {
        label: _np.mean(_np.stack(arrs), axis=0).tolist()
        for label, arrs in per_label.items()
        if arrs
    }


def cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors (lists or arrays). Returns 0.0
    if either vector has zero norm (degenerate)."""
    arr_a = _np.asarray(a, dtype=_np.float32)
    arr_b = _np.asarray(b, dtype=_np.float32)
    na = float(_np.linalg.norm(arr_a))
    nb = float(_np.linalg.norm(arr_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(_np.dot(arr_a, arr_b) / (na * nb))


def _read_wav_mono_16k(wav_input):
    """Return float32 mono numpy array at 16 kHz. Wraps the bytes-or-path one
    above, kept here so the embedding code can reuse it without touching the
    original signature."""
    import io as _io

    if isinstance(wav_input, (bytes, bytearray)):
        src = _io.BytesIO(wav_input)
    else:
        src = str(wav_input)
    audio, sr = _sf.read(src, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        raise DiarizationError(
            f"embedding extractor expects 16 kHz WAV, got {sr} Hz"
        )
    return audio


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
