"""Export a processed session into an eval case skeleton for hand-correction.

Copies the session's audio plus its machine transcript / speaker turns into
eval/cases/<name>/ as a *starting point*. You then listen to the audio,
correct reference.txt and reference_turns.json by hand, and flip
"verified": true in case.json. scripts/eval_run.py scores against these.

Usage:
    .venv/bin/python scripts/eval_bootstrap.py <session-uuid>
    .venv/bin/python scripts/eval_bootstrap.py <session-uuid> --name dinner-noisy
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from uuid import UUID

from sqlmodel import Session, select

from omilog.db import engine, init_db
from omilog.evals.cases import (
    CASE_META_FILE,
    REFERENCE_TEXT_FILE,
    REFERENCE_TURNS_FILE,
    turns_from_segments,
)
from omilog.models import AudioSession, Transcript


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", help="audio session UUID to export")
    parser.add_argument(
        "--name",
        help="case directory name (default: <date>-<uuid-prefix>)",
    )
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path("eval/cases"),
        help="root directory for eval cases (default: eval/cases)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing case directory of the same name",
    )
    args = parser.parse_args()

    try:
        sid = UUID(args.session)
    except ValueError:
        print(f"invalid UUID: {args.session}", file=sys.stderr)
        return 1

    init_db()
    with Session(engine) as db:
        sess = db.get(AudioSession, sid)
        if sess is None:
            print(f"session {sid} not found", file=sys.stderr)
            return 1
        transcript = db.exec(
            select(Transcript)
            .where(Transcript.audio_session_id == sid)
            .order_by(Transcript.created_at.desc())  # type: ignore[attr-defined]
            .limit(1)
        ).first()
        audio_path = Path(sess.audio_path) if sess.audio_path else None
        started_at = sess.started_at
        duration_s = sess.duration_s
        language = transcript.language if transcript else None

    if audio_path is None or not audio_path.exists():
        print(
            f"audio file missing for {sid} ({audio_path}) — already rotated off "
            "disk? Archive sessions you plan to use for eval (📌 in the UI).",
            file=sys.stderr,
        )
        return 1
    if transcript is None:
        print(
            f"no transcript for {sid} — run the pipeline (or replay_session.py) first "
            "so there is a machine hypothesis to hand-correct",
            file=sys.stderr,
        )
        return 1

    segments: list[dict] = []
    if transcript.segments_json:
        try:
            loaded = json.loads(transcript.segments_json)
            if isinstance(loaded, list):
                segments = [s for s in loaded if isinstance(s, dict)]
        except ValueError:
            pass

    name = args.name or f"{started_at:%Y-%m-%d}-{str(sid)[:8]}"
    case_dir = args.cases_dir / name
    if case_dir.exists() and not args.force:
        print(f"{case_dir} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    case_dir.mkdir(parents=True, exist_ok=True)

    suffix = audio_path.suffix or ".opus"
    shutil.copy2(audio_path, case_dir / f"audio{suffix}")

    # One segment per line: easy to correct while scrubbing the audio. WER
    # normalization collapses the newlines, so line structure is free.
    lines = [s.get("text", "").strip() for s in segments]
    text = "\n".join(line for line in lines if line) or transcript.text
    (case_dir / REFERENCE_TEXT_FILE).write_text(text + "\n", encoding="utf-8")

    turns = turns_from_segments(segments)
    if turns:
        (case_dir / REFERENCE_TURNS_FILE).write_text(
            json.dumps(turns, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    meta = {
        "source_session_id": str(sid),
        "recorded_at": started_at.isoformat(),
        "duration_s": duration_s,
        "language": language,
        "verified": False,
        "notes": "",
    }
    (case_dir / CASE_META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"exported {sid} → {case_dir}/")
    print(f"  audio{suffix}")
    print(f"  {REFERENCE_TEXT_FILE}        ← machine transcript, CORRECT ME")
    if turns:
        print(f"  {REFERENCE_TURNS_FILE}  ← machine speaker turns, CORRECT ME")
    else:
        print(
            f"  (no speaker labels on this transcript — create {REFERENCE_TURNS_FILE} "
            "by hand if you want DER scoring)"
        )
    print(f"  {CASE_META_FILE}             ← set \"verified\": true when done")
    print("Then: .venv/bin/python scripts/eval_run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
