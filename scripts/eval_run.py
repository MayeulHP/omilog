"""Score the current STT (+ diarization) configuration against the eval set.

Runs the same code path as the production pipeline (transcode → whisper →
repeat-collapse → diarize → post-merge → assign → relabel) on every case
under eval/cases/, computes WER / DER / USER-attribution against the
hand-corrected references, prints a table, and appends a JSON line with the
full config snapshot to eval/results/history.jsonl so runs are comparable
over time.

Not covered: cross-conversation speaker linking and is_user promotion (they
need DB state and would mutate it) — DER here scores the per-conversation
pipeline including the talk-time USER heuristic.

Usage:
    .venv/bin/python scripts/eval_run.py
    .venv/bin/python scripts/eval_run.py --case 2026-06-08-dinner --reuse-stt
    .venv/bin/python scripts/eval_run.py --no-diarize --note "baseline turbo-q5"
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omilog.config import settings
from omilog.evals.cases import (
    EvalCase,
    diarization_config_snapshot,
    fingerprint,
    load_cases,
    stt_config_snapshot,
    turns_from_segments,
)
from omilog.evals.metrics import (
    diarization_error_rate,
    user_attribution_accuracy,
    word_error_rate,
)
from omilog.pipeline import diarize as diarize_mod
from omilog.pipeline.audio import transcode_to_wav_bytes
from omilog.pipeline.stt import collapse_repeated_segments, transcribe_wav


async def _stt_hypothesis(
    case: EvalCase,
    wav_bytes: bytes,
    *,
    reuse_stt: bool,
    cache_dir: Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Transcribe (or reuse a cached transcription of) the case audio.

    The cache is keyed on the STT config fingerprint, so --reuse-stt is safe:
    a hypothesis produced under different STT settings is never reused. This
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


async def _diarize_hypothesis(
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


async def _eval_case(
    case: EvalCase,
    *,
    do_diarize: bool,
    reuse_stt: bool,
    cache_dir: Path,
    collar: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {"verified": case.verified, "error": None}
    wav_bytes = await transcode_to_wav_bytes(case.audio_path)
    text, segments = await _stt_hypothesis(
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

    if do_diarize and case.reference_turns:
        labeled = await _diarize_hypothesis(wav_bytes, segments)
        hyp_turns = turns_from_segments(labeled)
        der = diarization_error_rate(case.reference_turns, hyp_turns, collar=collar)
        row.update(
            der=der.der,
            miss_s=der.miss_s,
            false_alarm_s=der.false_alarm_s,
            confusion_s=der.confusion_s,
            ref_speech_s=der.ref_speech_s,
            ref_speakers=der.ref_speakers,
            hyp_speakers=der.hyp_speakers,
        )
        ua = user_attribution_accuracy(case.reference_turns, hyp_turns)
        if ua is not None:
            row.update(user_acc=ua.accuracy, ref_user_s=ua.ref_user_s)
    return row


def _fmt_pct(value: Any) -> str:
    return f"{value * 100:5.1f}" if isinstance(value, (int, float)) else "    -"


def _print_table(rows: dict[str, dict[str, Any]]) -> None:
    header = (
        f"{'case':<30} {'WER%':>5}  {'S/I/D':>11}  {'DER%':>5}  "
        f"{'miss/fa/conf s':>16}  {'USER%':>5}  {'spk r/h':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, row in rows.items():
        mark = " " if row.get("verified") else "⚠"
        if row.get("error"):
            print(f"{mark}{name:<29} FAILED: {row['error']}")
            continue
        sid = f"{row['substitutions']}/{row['insertions']}/{row['deletions']}"
        if "der" in row:
            mfc = (
                f"{row['miss_s']:.1f}/{row['false_alarm_s']:.1f}/{row['confusion_s']:.1f}"
            )
            spk = f"{row['ref_speakers']}/{row['hyp_speakers']}"
        else:
            mfc, spk = "-", "-"
        print(
            f"{mark}{name:<29} {_fmt_pct(row.get('wer'))}  {sid:>11}  "
            f"{_fmt_pct(row.get('der'))}  {mfc:>16}  "
            f"{_fmt_pct(row.get('user_acc'))}  {spk:>7}"
        )


def _aggregate(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows.values() if not r.get("error")]
    agg: dict[str, Any] = {"cases": len(rows), "failed": len(rows) - len(ok)}
    ref_words = sum(r["ref_words"] for r in ok)
    if ref_words:
        errors = sum(r["substitutions"] + r["insertions"] + r["deletions"] for r in ok)
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


async def _run(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases_dir)
    if args.case:
        wanted = set(args.case)
        unknown = wanted - {c.name for c in cases}
        if unknown:
            print(f"unknown case(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 1
        cases = [c for c in cases if c.name in wanted]
    if not cases:
        print(
            f"no eval cases under {args.cases_dir} — create one with "
            "scripts/eval_bootstrap.py <session-uuid>",
            file=sys.stderr,
        )
        return 1

    if not settings.stt_base_url and not args.reuse_stt:
        print("OMILOG_STT_BASE_URL not set (and no --reuse-stt)", file=sys.stderr)
        return 1

    do_diarize = settings.diarization_enabled if args.diarize is None else args.diarize
    if do_diarize and not diarize_mod.DIARIZATION_AVAILABLE:
        print(
            "diarization deps unavailable "
            f"({diarize_mod.DIARIZATION_IMPORT_ERROR}) — DER will be skipped",
            file=sys.stderr,
        )
        do_diarize = False

    cache_dir = args.results_dir / "cache"
    rows: dict[str, dict[str, Any]] = {}
    for case in cases:
        print(f"… {case.name}", file=sys.stderr)
        try:
            rows[case.name] = await _eval_case(
                case,
                do_diarize=do_diarize,
                reuse_stt=args.reuse_stt,
                cache_dir=cache_dir,
                collar=args.collar,
            )
        except Exception as e:  # noqa: BLE001 — keep scoring the other cases
            rows[case.name] = {"verified": case.verified, "error": f"{type(e).__name__}: {e}"}

    agg = _aggregate(rows)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": args.note,
        "collar": args.collar,
        "stt_config": stt_config_snapshot(settings),
        "diarization_config": diarization_config_snapshot(settings) if do_diarize else None,
        "cases": rows,
        "aggregate": agg,
    }

    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        _print_table(rows)
        print()
        parts = [f"{agg['cases']} case(s)"]
        if "wer" in agg:
            parts.append(f"WER {agg['wer'] * 100:.1f}% over {agg['ref_words']} ref words")
        if "der" in agg:
            parts.append(f"DER {agg['der'] * 100:.1f}% over {agg['ref_speech_s']:.0f}s speech")
        if "user_acc" in agg:
            parts.append(f"USER attribution {agg['user_acc'] * 100:.0f}%")
        if agg["failed"]:
            parts.append(f"{agg['failed']} FAILED")
        print("aggregate: " + " | ".join(parts))
        unverified = [n for n, r in rows.items() if not r.get("verified")]
        if unverified:
            print(
                f"⚠ unverified reference(s) — machine output, not ground truth yet: "
                f"{', '.join(unverified)}"
            )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    history = args.results_dir / "history.jsonl"
    with history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    if not args.json:
        print(f"appended to {history}")

    return 1 if agg["failed"] else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", type=Path, default=Path("eval/cases"))
    parser.add_argument("--results-dir", type=Path, default=Path("eval/results"))
    parser.add_argument(
        "--case",
        action="append",
        help="run only this case (repeatable); default: all cases",
    )
    diarize_group = parser.add_mutually_exclusive_group()
    diarize_group.add_argument(
        "--diarize",
        dest="diarize",
        action="store_true",
        default=None,
        help="force diarization scoring on (default: follow OMILOG_DIARIZATION_ENABLED)",
    )
    diarize_group.add_argument(
        "--no-diarize",
        dest="diarize",
        action="store_false",
        help="skip diarization / DER even if enabled in config",
    )
    parser.add_argument(
        "--reuse-stt",
        action="store_true",
        help="reuse cached STT output when the STT config is unchanged "
        "(fast diarization-knob iteration)",
    )
    parser.add_argument(
        "--collar",
        type=float,
        default=0.25,
        help="DER no-score collar in seconds around reference turn boundaries "
        "(default 0.25)",
    )
    parser.add_argument("--note", default="", help="free-text label stored in the history row")
    parser.add_argument(
        "--json", action="store_true", help="print the full result record as JSON"
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
