"""Re-run STT on an existing session.

Useful when you've upgraded the whisper model, fixed an STT bug, or just want
to retry a failed session without re-recording it.

Usage:
    .venv/bin/python scripts/replay_session.py <session-uuid>
    .venv/bin/python scripts/replay_session.py --all-failed
"""

import argparse
import asyncio
import sys
from uuid import UUID

from sqlmodel import Session, select

from omilog.config import assert_runtime_secrets, settings
from omilog.db import engine, init_db
from omilog.models import AudioSession, SessionStatus
from omilog.pipeline.runner import process_one


def _reset_to_pending(session_id: UUID) -> bool:
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None:
            return False
        sess.status = SessionStatus.pending_stt
        sess.error_msg = None
        db.add(sess)
        db.commit()
        return True


async def _replay(session_id: UUID) -> None:
    if not _reset_to_pending(session_id):
        print(f"session {session_id} not found", file=sys.stderr)
        return
    await process_one(session_id)


async def _replay_all_failed() -> None:
    with Session(engine) as db:
        rows = db.exec(
            select(AudioSession).where(AudioSession.status == SessionStatus.failed)
        ).all()
        ids = [r.id for r in rows]
    print(f"replaying {len(ids)} failed sessions")
    for sid in ids:
        print(f"→ {sid}")
        await _replay(sid)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("session", nargs="?", help="session UUID to replay")
    group.add_argument(
        "--all-failed",
        action="store_true",
        help="replay every session currently in `failed` state",
    )
    args = parser.parse_args()

    assert_runtime_secrets()
    init_db()

    if not settings.stt_base_url:
        print(
            "OMILOG_STT_BASE_URL not set — nothing to replay against.",
            file=sys.stderr,
        )
        return 1

    if args.all_failed:
        asyncio.run(_replay_all_failed())
    else:
        try:
            sid = UUID(args.session)
        except ValueError:
            print(f"invalid UUID: {args.session}", file=sys.stderr)
            return 1
        asyncio.run(_replay(sid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
