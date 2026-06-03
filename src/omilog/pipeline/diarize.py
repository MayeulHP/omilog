"""Speaker diarization via pyannote-audio.

Optional Phase 4 stage. After STT returns whisper's segment timestamps, we
run pyannote-audio over the same audio to discover speaker turns, then map
each whisper segment to a speaker by time overlap.

Heuristic: the speaker with the largest cumulative talk time in a single
conversation is **the user** (wearable-mic geometry — your voice is the
loudest signal from chest-level distance). Other speakers become S1, S2…
in talk-time-descending order, stable per conversation.

The pyannote-audio dep is **optional** (`.[diarization]` extra, ~2 GB of
torch). We try-import at module load; if absent or misconfigured, the
runner skips diarization gracefully — transcripts still flow to LLM.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("omilog.pipeline.diarize")


try:
    from pyannote.audio import Pipeline as _PyannotePipeline
    PYANNOTE_AVAILABLE = True
except Exception as _e:  # noqa: BLE001 — any import-time failure means "off"
    _PyannotePipeline = None  # type: ignore[assignment]
    PYANNOTE_AVAILABLE = False
    logger.debug("pyannote-audio not available at import time: %s", _e)


class DiarizationError(RuntimeError):
    pass


# Module-scoped cache: loading the pipeline is ~30 s and ~2 GB of state, and
# we use the same model for every session. asyncio.Lock guards the first load
# so concurrent calls don't load twice.
_PIPELINE_CACHE: Any = None
_LOAD_LOCK = asyncio.Lock()


async def get_pipeline(hf_token: str, model: str):
    if not PYANNOTE_AVAILABLE:
        raise DiarizationError(
            "pyannote-audio is not installed. Run `uv sync --extra diarization` "
            "(or `pip install -e '.[diarization]'`)."
        )
    if not hf_token:
        raise DiarizationError(
            "OMILOG_HF_TOKEN is not set. Generate one at "
            "https://huggingface.co/settings/tokens and accept the licenses on "
            "the model pages — see docs/diarization-setup.md."
        )
    global _PIPELINE_CACHE
    async with _LOAD_LOCK:
        if _PIPELINE_CACHE is None:
            loop = asyncio.get_event_loop()
            try:
                _PIPELINE_CACHE = await loop.run_in_executor(
                    None,
                    lambda: _PyannotePipeline.from_pretrained(
                        model, use_auth_token=hf_token
                    ),
                )
            except Exception as e:
                raise DiarizationError(
                    f"failed to load pyannote pipeline {model!r}: {e}"
                ) from e
    return _PIPELINE_CACHE


async def diarize(
    wav_path: Path,
    *,
    hf_token: str,
    model: str,
) -> list[dict[str, Any]]:
    """Run diarization on a WAV file. Returns turn dicts:
        [{'start': float_s, 'end': float_s, 'speaker': 'SPEAKER_00'}, …]

    Pyannote expects a torchaudio-readable file — WAV mono 16 kHz works best.
    The caller is responsible for handing us such a file (the runner uses the
    already-decoded WAV from the STT step via a temp file).
    """
    pipeline = await get_pipeline(hf_token, model)
    loop = asyncio.get_event_loop()
    try:
        annotation = await loop.run_in_executor(None, pipeline, str(wav_path))
    except Exception as e:
        raise DiarizationError(f"pyannote inference failed: {e}") from e

    turns: list[dict[str, Any]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,  # SPEAKER_00 / SPEAKER_01 …
            }
        )
    return turns


# ──────────────────────────────────────────────────────────────────────────────
# Merge with whisper transcript segments + user-heuristic relabel
# ──────────────────────────────────────────────────────────────────────────────

def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers_to_segments(
    whisper_segments: list[dict[str, Any]],
    diarization_turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """For each whisper segment, attribute the speaker who overlapped it most.

    Whisper segments without start/end timestamps are left untouched.
    """
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
    """Replace pyannote's `SPEAKER_NN` with `USER` (longest cumulative talker)
    and `S1..Sn` for everyone else, ranked by talking time descending."""
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
