"""Score the current STT (+ diarization) configuration against the eval set.

Thin CLI over omilog.evals.runner (the /eval web UI shares the same code):
runs the production STT→diarize path on every case under eval/cases/,
computes WER / DER / USER-attribution against the hand-corrected
references, prints a table, and appends a JSON line with the full config
snapshot to eval/results/history.jsonl so runs are comparable over time.

Usage:
    .venv/bin/python scripts/eval_run.py
    .venv/bin/python scripts/eval_run.py --case 2026-06-08-dinner --reuse-stt
    .venv/bin/python scripts/eval_run.py --no-diarize --note "baseline turbo-q5"
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from omilog.config import settings
from omilog.evals.cases import load_cases
from omilog.evals.runner import (
    append_history,
    build_record,
    eval_case,
    resolve_do_diarize,
)


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
            f"no eval cases under {args.cases_dir} — create one from a "
            "conversation page in the web UI, or with "
            "scripts/eval_bootstrap.py <session-uuid>",
            file=sys.stderr,
        )
        return 1

    if not settings.stt_base_url and not args.reuse_stt:
        print("OMILOG_STT_BASE_URL not set (and no --reuse-stt)", file=sys.stderr)
        return 1

    do_diarize, reason = resolve_do_diarize(args.diarize)
    if reason:
        print(f"{reason} — DER will be skipped", file=sys.stderr)

    cache_dir = args.results_dir / "cache"
    rows: dict[str, dict[str, Any]] = {}
    for case in cases:
        print(f"… {case.name}", file=sys.stderr)
        try:
            rows[case.name] = await eval_case(
                case,
                do_diarize=do_diarize,
                reuse_stt=args.reuse_stt,
                cache_dir=cache_dir,
                collar=args.collar,
            )
        except Exception as e:  # noqa: BLE001 — keep scoring the other cases
            rows[case.name] = {"verified": case.verified, "error": f"{type(e).__name__}: {e}"}

    record = build_record(
        rows, note=args.note, collar=args.collar, with_diarization=do_diarize
    )
    agg = record["aggregate"]

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

    history = append_history(args.results_dir, record)
    if not args.json:
        print(f"appended to {history}")

    return 1 if agg["failed"] else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-dir", type=Path, default=settings.eval_cases_dir)
    parser.add_argument("--results-dir", type=Path, default=settings.eval_results_dir)
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
