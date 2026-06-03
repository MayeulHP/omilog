"""Re-run pipeline stages on existing sessions.

Useful after upgrading a model, fixing a prompt, or recovering from a
transient backend error.

Usage:
    .venv/bin/python scripts/replay_session.py <session-uuid>          # auto-detect stage
    .venv/bin/python scripts/replay_session.py <session-uuid> --stage stt
    .venv/bin/python scripts/replay_session.py <session-uuid> --stage llm
    .venv/bin/python scripts/replay_session.py --all-failed
"""

import argparse
import asyncio
import sys
from uuid import UUID

from sqlmodel import Session, select

from omilog.config import assert_runtime_secrets, settings
from omilog.db import engine, init_db
from omilog.models import AudioSession, SessionStatus, Transcript
from omilog.pipeline.runner import process_llm, process_stt, process_vad


def _detect_stage(session_id: UUID) -> str | None:
    """Pick which stage to replay.

    - Parent capture (status was segmented / pending_vad) → vad
    - Child with no transcript yet                       → stt
    - Child with a transcript                             → llm
    """
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None:
            return None
        if sess.status in (SessionStatus.segmented, SessionStatus.pending_vad) or (
            sess.parent_id is None
            and sess.status in (SessionStatus.recording, SessionStatus.failed)
            and settings.vad_enabled
        ):
            return "vad"
        has_transcript = (
            db.exec(
                select(Transcript)
                .where(Transcript.audio_session_id == session_id)
                .limit(1)
            ).first()
            is not None
        )
        return "llm" if has_transcript else "stt"


def _reset_status(session_id: UUID, target: SessionStatus) -> bool:
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None:
            return False
        sess.status = target
        sess.error_msg = None
        db.add(sess)
        db.commit()
        return True


async def _replay(session_id: UUID, stage: str) -> None:
    if stage == "vad":
        if not settings.vad_enabled:
            print("OMILOG_VAD_ENABLED is false; cannot replay VAD.", file=sys.stderr)
            return
        _reset_status(session_id, SessionStatus.pending_vad)
        await process_vad(session_id)
    elif stage == "stt":
        if not settings.stt_base_url:
            print("OMILOG_STT_BASE_URL not set; cannot replay STT.", file=sys.stderr)
            return
        _reset_status(session_id, SessionStatus.pending_stt)
        await process_stt(session_id)
    elif stage == "llm":
        if not settings.llm_base_url:
            print("OMILOG_LLM_BASE_URL not set; cannot replay LLM.", file=sys.stderr)
            return
        _reset_status(session_id, SessionStatus.pending_llm)
        await process_llm(session_id)
    else:
        print(f"unknown stage: {stage}", file=sys.stderr)


async def _replay_all_failed() -> None:
    with Session(engine) as db:
        rows = db.exec(
            select(AudioSession).where(AudioSession.status == SessionStatus.failed)
        ).all()
        ids = [r.id for r in rows]
    print(f"replaying {len(ids)} failed sessions")
    for sid in ids:
        stage = _detect_stage(sid)
        if stage is None:
            print(f"→ {sid}: vanished, skipping")
            continue
        print(f"→ {sid}: stage={stage}")
        await _replay(sid, stage)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("session", nargs="?", help="session UUID to replay")
    group.add_argument(
        "--all-failed",
        action="store_true",
        help="replay every session in `failed` state, auto-detecting the stage",
    )
    parser.add_argument(
        "--stage",
        choices=["vad", "stt", "llm", "auto"],
        default="auto",
        help="which stage to re-run; auto picks vad / stt / llm based on session state",
    )
    args = parser.parse_args()

    assert_runtime_secrets()
    init_db()

    if args.all_failed:
        asyncio.run(_replay_all_failed())
        return 0

    try:
        sid = UUID(args.session)
    except (ValueError, TypeError):
        print(f"invalid UUID: {args.session}", file=sys.stderr)
        return 1

    stage = args.stage
    if stage == "auto":
        detected = _detect_stage(sid)
        if detected is None:
            print(f"session {sid} not found", file=sys.stderr)
            return 1
        stage = detected
        print(f"auto-detected stage: {stage}", file=sys.stderr)

    asyncio.run(_replay(sid, stage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
