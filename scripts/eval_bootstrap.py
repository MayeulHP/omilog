"""Export a processed session into an eval case skeleton for hand-correction.

Copies the session's audio plus its machine transcript / speaker turns into
eval/cases/<name>/ as a *starting point*. Correct it in the /eval web UI
(audio player + row editor), or by hand: reference.txt (words),
reference_turns.json (who spoke when), then flip "verified": true in
case.json. scripts/eval_run.py scores against these.

--hq re-transcribes the audio now with quality-leaning settings (pinned
language + a vocabulary prompt built from known speaker/people names) so
the draft needs fewer fixes. Needs the STT backend reachable.

Usage:
    .venv/bin/python scripts/eval_bootstrap.py <session-uuid>
    .venv/bin/python scripts/eval_bootstrap.py <session-uuid> --name dinner-noisy --hq
"""

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID

from omilog.config import settings
from omilog.db import init_db
from omilog.evals.bootstrap import BootstrapError, create_case
from omilog.evals.cases import CASE_META_FILE, REFERENCE_TEXT_FILE, REFERENCE_TURNS_FILE


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
        default=settings.eval_cases_dir,
        help="root directory for eval cases (default: eval/cases)",
    )
    parser.add_argument(
        "--hq",
        action="store_true",
        help="re-transcribe now with quality-leaning settings for a better draft "
        "(needs the STT backend reachable)",
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
    try:
        case_dir = asyncio.run(
            create_case(
                sid,
                name=args.name,
                cases_dir=args.cases_dir,
                hq=args.hq,
                force=args.force,
            )
        )
    except BootstrapError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(f"exported {sid} → {case_dir}/")
    print(f"  {REFERENCE_TEXT_FILE} + {REFERENCE_TURNS_FILE}  ← machine draft, CORRECT ME")
    print(f"  {CASE_META_FILE}  ← set \"verified\": true when done")
    print("Easiest path: open /eval in the web UI. Then: scripts/eval_run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
