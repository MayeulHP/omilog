"""One-shot backfill: re-score conversations stuck at quality_score=0.5.

Run this once after upgrading to the LLM-quality-scoring version, against
your existing archive. Each candidate conversation gets a small dedicated
LLM call (much cheaper than a full extraction) that returns just
``quality_score`` and ``quality_reasoning``; those two fields are updated
in place. Calendar events, action items, mentioned people, etc. are NOT
touched — this script is deliberately surgical to avoid clobbering
extractions you may have already engaged with.

Safety
------
- Idempotent: only conversations still at the default ``quality_score=0.5``
  with no ``quality_override`` set are touched. Interrupt with Ctrl-C and
  re-run; previously-scored rows are skipped automatically.
- Respects manual overrides: if you've already 👍/👎'd a conversation,
  this script skips it.
- ``--dry-run`` shows what would be scored without calling the LLM.
- ``--limit N`` caps the number of conversations per run, so you can
  bite off a batch at a time on a slow LLM box.

Usage
-----
    .venv/bin/python scripts/backfill_quality.py --dry-run
    .venv/bin/python scripts/backfill_quality.py --limit 10
    .venv/bin/python scripts/backfill_quality.py

After it's done its job, delete this file — we won't need it again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# This script lives under scripts/ — add the project root to sys.path so
# `from omilog…` resolves when run directly with no package install.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from sqlmodel import Session, select  # noqa: E402

from omilog.config import settings  # noqa: E402
from omilog.db import engine  # noqa: E402
from omilog.models import Conversation, Transcript  # noqa: E402
from omilog.pipeline.extract import _clamped_float, _string_or_none  # noqa: E402
from omilog.pipeline.llm import LLMError, chat_json  # noqa: E402

logger = logging.getLogger("backfill_quality")


# Same anchors as the main extraction prompt, but as a standalone scorer
# that doesn't re-extract anything else. Returns ~50 output tokens vs the
# 1k-ish of a full extraction, so this is much cheaper to run at scale.
_QUALITY_PROMPT = """/no_think

You score conversation transcripts on a 0.0 to 1.0 usefulness scale.

Use these anchors:
- 0.0: pure noise (ambient TV transcribed, mumbling to self, transcript that doesn't reflect a real interaction)
- 0.2: real-but-trivial (greetings, brief logistics, "ok bye", weather chatter)
- 0.5: ordinary daily conversation, some content but nothing memorable
- 0.7: clear conversation with concrete content worth remembering (a plan made, news shared, a real decision)
- 1.0: substantive multi-party discussion with decisions made, important personal news, or memorable content

Be conservative. When in doubt, pick the lower of two adjacent anchors. A transcript with no real participants (likely captured ambient audio) is always 0.0.

Output STRICT JSON, no prose, no markdown fences, no <think> tags:
{"quality_score": 0.0, "quality_reasoning": "one short sentence"}"""


def _strip_quirks(text: str) -> str:
    """Strip <think> blocks and ``` fences. The main parser has the same
    logic but it's not exported, so this is a smaller local copy."""
    s = text.strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl >= 0:
            s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


async def _score_one(transcript_text: str) -> tuple[float, str | None]:
    """Send one transcript to the LLM, return (score, reasoning)."""
    messages = [
        {"role": "system", "content": _QUALITY_PROMPT},
        {"role": "user", "content": f"Transcript:\n{transcript_text[:24000]}"},
    ]
    chat = await chat_json(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        messages=messages,
        temperature=0.0,
        max_tokens=200,
        timeout_s=settings.llm_timeout_s,
    )
    cleaned = _strip_quirks(chat.text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # Same fallback as the main parser: json_repair is forgiving.
        import json_repair  # type: ignore[import-untyped]

        obj = json_repair.loads(cleaned)
    if not isinstance(obj, dict):
        raise ValueError(f"unexpected LLM output: {chat.text[:200]!r}")
    score = _clamped_float(obj.get("quality_score"))
    reasoning = _string_or_none(obj.get("quality_reasoning"))
    if score is None:
        raise ValueError(
            f"no parseable quality_score in LLM output: {chat.text[:200]!r}"
        )
    return score, reasoning


def _bucket(q: float) -> str:
    if q < 0.3:
        return "noise"
    if q < 0.7:
        return "normal"
    return "subst"


async def main(*, dry_run: bool, limit: int | None) -> int:
    if not settings.llm_base_url:
        print(
            "OMILOG_LLM_BASE_URL is empty; nothing to do. "
            "Set it in .env and re-run.",
            file=sys.stderr,
        )
        return 1

    # Candidates: stuck at the default 0.5 with no manual override. The
    # default-from-fresh-row case looks identical to "LLM didn't return a
    # score and we fell back to 0.5", which is correct — we want to score
    # both kinds. If the LLM did say "exactly 0.5", re-scoring it shouldn't
    # change anything material.
    with Session(engine) as db:
        stmt = (
            select(Conversation)
            .where(Conversation.quality_score == 0.5)
            .where(Conversation.quality_override == None)  # noqa: E711
            .order_by(Conversation.started_at.desc())
        )
        if limit:
            stmt = stmt.limit(limit)
        convs = list(db.exec(stmt).all())

    if not convs:
        print("Nothing to backfill — every conversation already has a score.")
        return 0

    print(f"Found {len(convs)} conversation(s) at the default score.")
    if dry_run:
        print("\nDry-run; would score these:")
        for c in convs:
            print(
                f"  {c.id} {c.started_at:%Y-%m-%d %H:%M} "
                f"{(c.title or '(untitled)')[:60]}"
            )
        return 0

    print("Re-scoring (Ctrl-C is safe; rerun to resume).\n")

    updated = 0
    skipped = 0
    failed = 0
    for c in convs:
        with Session(engine) as db:
            t = db.exec(
                select(Transcript)
                .where(Transcript.audio_session_id == c.audio_session_id)
                .order_by(Transcript.created_at.desc())
                .limit(1)
            ).first()
        if not t or not (t.text or "").strip():
            print(f"  skip {c.id}: no transcript", file=sys.stderr)
            skipped += 1
            continue

        try:
            score, reasoning = await _score_one(t.text)
        except (LLMError, ValueError) as e:
            print(f"  fail {c.id}: {e}", file=sys.stderr)
            failed += 1
            continue
        except Exception as e:  # noqa: BLE001 — surface anything else but keep going
            print(f"  fail {c.id}: unexpected {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
            continue

        with Session(engine) as db:
            row = db.get(Conversation, c.id)
            if row is None:
                continue  # deleted between query and write — fine, just skip
            row.quality_score = score
            row.quality_reasoning = reasoning
            row.updated_at = datetime.now(timezone.utc)
            db.add(row)
            db.commit()

        print(
            f"  {c.id} {c.started_at:%Y-%m-%d %H:%M} "
            f"→ {score:.2f} ({_bucket(score)}) — "
            f"{(reasoning or '(no reasoning)')[:80]}"
        )
        updated += 1

    print(
        f"\nDone. Updated {updated}, "
        f"skipped {skipped} (no transcript), "
        f"failed {failed}."
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="list candidates without calling the LLM",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap number of conversations to score in this run",
    )
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(dry_run=args.dry_run, limit=args.limit)))
