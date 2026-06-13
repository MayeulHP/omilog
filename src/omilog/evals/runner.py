"""Score eval cases against the current STT/diarization config — shared by
the scripts/eval_run.py CLI and the /eval web UI's per-case "Run eval"
button. Runs the same code path as the production pipeline (transcode →
whisper → repeat-collapse → diarize → post-merge → assign → relabel).

Not covered: cross-conversation speaker linking and is_user promotion (they
need DB state and would mutate it) — DER scores the per-conversation
pipeline including the talk-time USER heuristic.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import settings
from ..pipeline import diarize as diarize_mod
from ..pipeline.audio import transcode_to_wav_bytes
from ..pipeline.stt import collapse_repeated_segments, transcribe_wav
from .cases import (
    EvalCase,
    diarization_config_snapshot,
    fingerprint,
    stt_config_snapshot,
    turns_from_segments,
)
from .metrics import (
    diarization_error_rate,
    user_attribution_accuracy,
    word_error_rate,
)

HISTORY_FILE = "history.jsonl"


async def stt_hypothesis(
    case: EvalCase,
    wav_bytes: bytes,
    *,
    reuse_stt: bool,
    cache_dir: Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Transcribe (or reuse a cached transcription of) the case audio.

    The cache is keyed on the STT config fingerprint, so reuse is safe: a
    hypothesis produced under different STT settings is never reused. This
    makes diarization-knob iteration cheap — only the diarizer re-runs.
    """
    fp = fingerprint(stt_config_snapshot(settings))
    cache_path = cache_dir / f"{case.name}.stt.json"
    if reuse_stt and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            cached = None
        if cached and cached.get("fingerprint") == fp:
            return cached.get("text", ""), list(cached.get("segments") or [])

    result = await transcribe_wav(
        wav_bytes,
        base_url=settings.stt_base_url,
        inference_path=settings.stt_inference_path,
        language=settings.stt_language,
        timeout_s=settings.stt_timeout_s,
        initial_prompt=settings.stt_initial_prompt,
        temperature=settings.stt_temperature,
    )
    segments = collapse_repeated_segments(list(result.segments or []))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "fingerprint": fp,
                "text": result.text,
                "segments": segments,
                "language": result.language,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result.text, segments


async def diarize_hypothesis(
    wav_bytes: bytes,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Mirror runner._diarize_or_continue minus the DB-coupled linking."""
    turns = await diarize_mod.diarize(
        wav_bytes,
        seg_path=settings.diarization_segmentation_model,
        emb_path=settings.diarization_embedding_model,
        min_speech_s=settings.diarization_min_speech_seconds,
        min_silence_s=settings.diarization_min_silence_seconds,
        num_threads=settings.diarization_num_threads,
        num_clusters=settings.diarization_num_clusters,
        cluster_threshold=settings.diarization_cluster_threshold,
    )
    if settings.diarization_post_merge_threshold < 1.0:
        turns = await diarize_mod.post_merge_clusters(
            wav_bytes,
            turns,
            emb_path=settings.diarization_embedding_model,
            threshold=settings.diarization_post_merge_threshold,
            num_threads=settings.diarization_num_threads,
        )
    labeled = diarize_mod.assign_speakers_to_segments([dict(s) for s in segments], turns)
    return diarize_mod.relabel_user_and_others(labeled)


async def eval_case(
    case: EvalCase,
    *,
    do_diarize: bool,
    reuse_stt: bool,
    cache_dir: Path,
    collar: float = 0.25,
) -> dict[str, Any]:
    """Score one case; returns a metrics row (see keys below). Raises on
    transcode/STT/diarize failure — callers decide how to report."""
    row: dict[str, Any] = {"verified": case.verified, "error": None}
    wav_bytes = await transcode_to_wav_bytes(case.audio_path)
    text, segments = await stt_hypothesis(
        case, wav_bytes, reuse_stt=reuse_stt, cache_dir=cache_dir
    )

    # Score the collapsed segments (what the UI shows and the LLM reads),
    # not the raw `text` field, which still contains hallucination loops.
    hyp_text = " ".join(
        s.get("text", "").strip() for s in segments if s.get("text", "").strip()
    ) or text
    wer = word_error_rate(case.reference_text, hyp_text)
    row.update(
        wer=wer.wer,
        substitutions=wer.substitutions,
        insertions=wer.insertions,
        deletions=wer.deletions,
        ref_words=wer.ref_words,
        hyp_words=wer.hyp_words,
    )

    # Reference rows are stored at segment granularity (the labeling UI edits
    # them); bridge ≤1s same-speaker gaps exactly like the hypothesis side so
    # the quantization is symmetric. Rows without a speaker drop out here —
    # a text-only case simply skips DER.
    ref_turns = turns_from_segments(case.reference_turns or [])
    if do_diarize and ref_turns:
        labeled = await diarize_hypothesis(wav_bytes, segments)
        hyp_turns = turns_from_segments(labeled)
        der = diarization_error_rate(ref_turns, hyp_turns, collar=collar)
        row.update(
            der=der.der,
            miss_s=der.miss_s,
            false_alarm_s=der.false_alarm_s,
            confusion_s=der.confusion_s,
            ref_speech_s=der.ref_speech_s,
            ref_speakers=der.ref_speakers,
            hyp_speakers=der.hyp_speakers,
        )
        ua = user_attribution_accuracy(ref_turns, hyp_turns)
        if ua is not None:
            row.update(user_acc=ua.accuracy, ref_user_s=ua.ref_user_s)
    return row


def aggregate(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows.values() if not r.get("error")]
    agg: dict[str, Any] = {"cases": len(rows), "failed": len(rows) - len(ok)}
    ref_words = sum(r["ref_words"] for r in ok if "ref_words" in r)
    if ref_words:
        errors = sum(
            r["substitutions"] + r["insertions"] + r["deletions"]
            for r in ok
            if "ref_words" in r
        )
        agg["wer"] = errors / ref_words
        agg["ref_words"] = ref_words
    der_rows = [r for r in ok if "der" in r]
    ref_speech = sum(r["ref_speech_s"] for r in der_rows)
    if ref_speech:
        err_s = sum(r["miss_s"] + r["false_alarm_s"] + r["confusion_s"] for r in der_rows)
        agg["der"] = err_s / ref_speech
        agg["ref_speech_s"] = ref_speech
    ua_rows = [r for r in ok if "user_acc" in r]
    ref_user = sum(r["ref_user_s"] for r in ua_rows)
    if ref_user:
        agg["user_acc"] = sum(r["user_acc"] * r["ref_user_s"] for r in ua_rows) / ref_user
    return agg


def build_record(
    rows: dict[str, dict[str, Any]],
    *,
    note: str,
    collar: float,
    with_diarization: bool,
) -> dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "collar": collar,
        "stt_config": stt_config_snapshot(settings),
        "diarization_config": (
            diarization_config_snapshot(settings) if with_diarization else None
        ),
        "cases": rows,
        "aggregate": aggregate(rows),
    }


def append_history(results_dir: Path, record: dict[str, Any]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    history = results_dir / HISTORY_FILE
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return history


def latest_metrics_by_case(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Most recent per-case metrics row from the history, with the record's
    timestamp and note folded in — feeds the /eval list page."""
    history = results_dir / HISTORY_FILE
    out: dict[str, dict[str, Any]] = {}
    if not history.is_file():
        return out
    for line in history.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        cases = record.get("cases")
        if not isinstance(cases, dict):
            continue
        for name, row in cases.items():
            if isinstance(row, dict):
                out[name] = {**row, "ts": record.get("ts"), "note": record.get("note")}
    return out


def resolve_do_diarize(requested: bool | None) -> tuple[bool, str | None]:
    """(do_diarize, reason_if_disabled). None = follow settings."""
    do = settings.diarization_enabled if requested is None else requested
    if do and not diarize_mod.DIARIZATION_AVAILABLE:
        return False, f"diarization deps unavailable ({diarize_mod.DIARIZATION_IMPORT_ERROR})"
    return do, None
