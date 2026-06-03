"""Diagnose VAD behaviour on a stored audio file.

Useful when:
  - A conversation got merged or split unexpectedly and you want to know why.
  - You're tuning OMILOG_VAD_THRESHOLD_DB or OMILOG_VAD_GAP_SECONDS for your
    mic / recording level.

Doesn't touch the DB; just runs `ffmpeg silencedetect` and the segmentation
math. Reuses the same code paths as the production runner.

Usage:
    .venv/bin/python scripts/debug_vad.py storage/<uuid>.opus
    .venv/bin/python scripts/debug_vad.py storage/<uuid>.opus --threshold -45 --gap 90
"""

import argparse
import asyncio
import sys
from pathlib import Path

from omilog.config import settings
from omilog.pipeline import vad


async def _run(
    path: Path,
    threshold_db: float,
    min_silence_s: float,
    gap_threshold_s: float,
    pad_s: float,
) -> int:
    print(f"file:        {path}")
    if not path.exists():
        print("missing", file=sys.stderr)
        return 1
    print(f"threshold:   {threshold_db} dB")
    print(f"min_silence: {min_silence_s}s")
    print(f"gap:         {gap_threshold_s}s")
    print(f"pad:         {pad_s}s")
    print()

    try:
        duration_s, silences = await vad.analyse(
            path,
            threshold_db=threshold_db,
            min_silence_s=min_silence_s,
        )
    except vad.VADError as e:
        print(f"VAD analyse failed: {e}", file=sys.stderr)
        return 1

    print(f"duration: {duration_s:.2f}s")
    print(f"silences: {len(silences)}")
    for s, e in silences:
        long_marker = "  ◆ ≥ gap" if (e - s) >= gap_threshold_s else ""
        print(f"  [{s:7.2f} .. {e:7.2f}] = {e - s:6.2f}s{long_marker}")
    print()

    convs = vad.segment_by_silence_gaps(
        duration_s,
        silences,
        gap_threshold_s=gap_threshold_s,
        pad_s=pad_s,
    )
    print(f"conversations after segmentation: {len(convs)}")
    if not convs:
        print("  (all silence — would mark status=silent)")
        return 0

    speech_total = sum(end - start for start, end in convs)
    for i, (start, end) in enumerate(convs):
        print(f"  [{i}] {start:7.2f}..{end:7.2f}  ({end - start:6.2f}s)")
    saved = duration_s - speech_total
    if duration_s > 0:
        pct = 100 * saved / duration_s
        print(f"\nspeech kept: {speech_total:.1f}s / {duration_s:.1f}s — would save {pct:.0f}% of file")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--threshold", type=float, default=settings.vad_threshold_db,
        help=f"silencedetect noise threshold dB (default: {settings.vad_threshold_db})",
    )
    parser.add_argument(
        "--min-silence", type=float, default=settings.vad_min_silence_seconds,
        help=f"min silence duration (default: {settings.vad_min_silence_seconds}s)",
    )
    parser.add_argument(
        "--gap", type=float, default=settings.vad_gap_seconds,
        help=f"conversation gap threshold (default: {settings.vad_gap_seconds}s)",
    )
    parser.add_argument(
        "--pad", type=float, default=settings.vad_pad_seconds,
        help=f"symmetric pad around conversations (default: {settings.vad_pad_seconds}s)",
    )
    args = parser.parse_args()
    return asyncio.run(
        _run(args.path, args.threshold, args.min_silence, args.gap, args.pad)
    )


if __name__ == "__main__":
    raise SystemExit(main())
