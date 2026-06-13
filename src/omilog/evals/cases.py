"""Eval case I/O: the on-disk format under eval/cases/, plus the helpers
shared by scripts/eval_bootstrap.py, scripts/eval_run.py and the /eval
labeling UI.

A case directory looks like:

    eval/cases/2026-06-08-dinner/
        audio.opus              # any ffmpeg-readable audio
        reference.txt           # hand-corrected transcript (plain spoken text)
        reference_turns.json    # [{"start": s, "end": e, "speaker": "USER", "text": "…"}]
        case.json               # metadata; "verified": true once hand-corrected

reference.txt feeds WER, reference_turns.json feeds DER. The turns rows may
carry a "text" field (the labeling UI edits one row table and both files
are derived from it; scoring ignores text on turns) and may omit "speaker"
(rows without one are excluded from DER scoring — a case can be text-only).
Everything under eval/ is gitignored — it is personal audio and must never
reach the public repo.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("omilog.evals.cases")

REFERENCE_TEXT_FILE = "reference.txt"
REFERENCE_TURNS_FILE = "reference_turns.json"
CASE_META_FILE = "case.json"

# Case names double as directory names and URL path segments — keep them
# boring so there's no traversal or quoting surface.
CASE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")


@dataclass
class EvalCase:
    name: str
    path: Path
    audio_path: Path
    reference_text: str
    reference_turns: list[dict[str, Any]] | None
    verified: bool
    meta: dict[str, Any] = field(default_factory=dict)


def load_case(path: Path) -> EvalCase | None:
    """Load one case directory; None (with a warning) when malformed or
    incomplete — one broken label file shouldn't block a suite run."""
    ref_file = path / REFERENCE_TEXT_FILE
    if not ref_file.is_file():
        logger.warning("eval case %s: no %s — skipping", path.name, REFERENCE_TEXT_FILE)
        return None
    audio_candidates = sorted(path.glob("audio.*"))
    if not audio_candidates:
        logger.warning("eval case %s: no audio.* file — skipping", path.name)
        return None
    reference_text = ref_file.read_text(encoding="utf-8")
    if not reference_text.strip():
        logger.warning("eval case %s: empty reference.txt — skipping", path.name)
        return None

    turns: list[dict[str, Any]] | None = None
    turns_file = path / REFERENCE_TURNS_FILE
    if turns_file.is_file():
        try:
            loaded = json.loads(turns_file.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                turns = [t for t in loaded if isinstance(t, dict)]
        except ValueError as e:
            logger.warning(
                "eval case %s: unparseable %s (%s) — DER will be skipped",
                path.name,
                REFERENCE_TURNS_FILE,
                e,
            )

    meta: dict[str, Any] = {}
    meta_file = path / CASE_META_FILE
    if meta_file.is_file():
        try:
            loaded = json.loads(meta_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded
        except ValueError as e:
            logger.warning("eval case %s: unparseable %s (%s)", path.name, CASE_META_FILE, e)

    return EvalCase(
        name=path.name,
        path=path,
        audio_path=audio_candidates[0],
        reference_text=reference_text,
        reference_turns=turns if turns else None,
        verified=bool(meta.get("verified", False)),
        meta=meta,
    )


def load_cases(cases_dir: Path) -> list[EvalCase]:
    """Load every well-formed case under ``cases_dir``."""
    if not cases_dir.is_dir():
        return []
    cases = (load_case(p) for p in sorted(cases_dir.iterdir()) if p.is_dir())
    return [c for c in cases if c is not None]


# ──────────────────────────────────────────────────────────────────────────────
# Row-based editing (labeling UI + bootstrap)
# ──────────────────────────────────────────────────────────────────────────────


def rows_from_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-segment editing rows {start, end, speaker?, text} from stored
    whisper segments. Unlike turns_from_segments this keeps one row per
    segment (text lives at segment granularity) and keeps rows without a
    speaker label."""
    rows: list[dict[str, Any]] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        row: dict[str, Any] = {
            "start": round(float(seg.get("start", 0) or 0), 2),
            "end": round(float(seg.get("end", 0) or 0), 2),
            "text": text,
        }
        if seg.get("speaker"):
            row["speaker"] = str(seg["speaker"])
        rows.append(row)
    return rows


def write_reference_files(case_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Derive and write both reference files from editing rows.

    reference.txt gets the chronological text lines (WER input);
    reference_turns.json gets the rows verbatim (DER reads start/end/speaker
    and ignores text; rows without a speaker are excluded from scoring)."""
    ordered = sorted(rows, key=lambda r: float(r.get("start", 0) or 0))
    text = "\n".join((r.get("text") or "").strip() for r in ordered if (r.get("text") or "").strip())
    (case_dir / REFERENCE_TEXT_FILE).write_text(text + "\n", encoding="utf-8")
    (case_dir / REFERENCE_TURNS_FILE).write_text(
        json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_case_meta(case_dir: Path) -> dict[str, Any]:
    meta_file = case_dir / CASE_META_FILE
    if not meta_file.is_file():
        return {}
    try:
        loaded = json.loads(meta_file.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def update_case_meta(case_dir: Path, **updates: Any) -> dict[str, Any]:
    """Merge ``updates`` into case.json, preserving unrelated keys."""
    meta = read_case_meta(case_dir)
    meta.update(updates)
    (case_dir / CASE_META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


def turns_from_segments(
    segments: list[dict[str, Any]],
    *,
    gap_tolerance_s: float = 1.0,
) -> list[dict[str, Any]]:
    """Merge speaker-labeled transcript segments into turns.

    Consecutive segments with the same speaker are folded into one turn when
    the gap between them is at most ``gap_tolerance_s`` — Whisper leaves
    small inter-segment holes inside what is clearly one speaking turn, and
    scoring those holes as missed speech would be noise, not signal. Used
    both to seed reference_turns.json (bootstrap) and to build hypothesis
    turns (eval run), so the quantization is symmetric.
    """
    labeled = []
    for seg in segments:
        spk = seg.get("speaker")
        if not spk:
            continue
        start = float(seg.get("start", 0) or 0)
        end = float(seg.get("end", start) or start)
        if end <= start:
            continue
        labeled.append((start, end, str(spk)))
    labeled.sort()

    turns: list[dict[str, Any]] = []
    for start, end, spk in labeled:
        if (
            turns
            and turns[-1]["speaker"] == spk
            and start - turns[-1]["end"] <= gap_tolerance_s
        ):
            turns[-1]["end"] = max(turns[-1]["end"], end)
        else:
            turns.append({"start": start, "end": end, "speaker": spk})
    return turns


# ──────────────────────────────────────────────────────────────────────────────
# Config snapshots — recorded in the results history so a metrics row is
# always attributable to the exact knob settings that produced it.
# ──────────────────────────────────────────────────────────────────────────────


def stt_config_snapshot(settings: Any) -> dict[str, Any]:
    return {
        "base_url": settings.stt_base_url,
        "inference_path": settings.stt_inference_path,
        "language": settings.stt_language,
        "initial_prompt": settings.stt_initial_prompt,
        "temperature": settings.stt_temperature,
        "model_name": settings.stt_model_name,
    }


def diarization_config_snapshot(settings: Any) -> dict[str, Any]:
    return {
        "segmentation_model": str(settings.diarization_segmentation_model),
        "embedding_model": str(settings.diarization_embedding_model),
        "min_speech_s": settings.diarization_min_speech_seconds,
        "min_silence_s": settings.diarization_min_silence_seconds,
        "num_clusters": settings.diarization_num_clusters,
        "cluster_threshold": settings.diarization_cluster_threshold,
        "post_merge_threshold": settings.diarization_post_merge_threshold,
    }


def fingerprint(snapshot: dict[str, Any]) -> str:
    """Stable short hash of a config snapshot — used to validate the STT
    response cache (a cached hypothesis is only reusable if it was produced
    by the same STT configuration)."""
    blob = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
