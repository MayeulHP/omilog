"""Daily-summary generation: bundle one day's substantive conversations into
a single LLM narrative.

Reads conversations with ``effective_quality >= quality_threshold`` for the
given day, hands the LLM each conversation's title + summary + extracted
items, asks for a 4-6 sentence narrative in the dominant language. Result
is cached in a ``DailySummary`` row keyed by ``(user_id, date)``; the same
day re-renders instantly until the user explicitly regenerates.

The threshold is configurable per call. ``settings.daily_summary_threshold``
provides the system-wide default; the user may override at request time
via a UI form (`/daily/{date}?threshold=…`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime, time, timedelta
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from sqlmodel import Session, select

from ..config import settings
from ..db import engine
from ..models import (
    ActionItem,
    ActionItemStatus,
    CalendarEvent,
    Conversation,
    DailySummary,
    PersonMention,
)
from .llm import LLMError, chat_json

logger = logging.getLogger("omilog.pipeline.daily")


# ──────────────────────────────────────────────────────────────────────────────
# LLM prompt
# ──────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """/no_think

You write a daily summary of someone's day based on transcripts of their conversations.

You receive a list of conversations from one calendar day, each with the time, a short title, a summary, and the extracted highlights (calendar events committed to, action items, people who came up). Conversations are pre-filtered to be substantive — small talk and noise have already been removed.

Write a 4 to 6 sentence narrative summarising the day. Focus on:
- Recurring themes and topics
- People who came up across multiple conversations
- Decisions made, plans formed, news shared
- Outstanding action items if any feel important

Do NOT list each conversation separately. Write it like a journal entry, in the conversation's dominant language (French if mostly French, English if mostly English, default to English if mixed). Be conservative: if the day was sparse, write 1 to 2 sentences rather than padding. Don't invent details that aren't in the source material.

Output STRICT JSON, no prose outside it, no markdown fences:
{"narrative": "4-6 sentence narrative as one string"}"""


def _effective_quality(c: Conversation) -> float:
    """Same logic as web/routes.py — duplicated to avoid an import loop and
    because the override-or-score formula is the kind of thing that should
    have exactly one definition per consumer (and stay in sync via tests)."""
    return c.quality_override if c.quality_override is not None else c.quality_score


def _utc_day_bounds(d: date_cls, tz_label: str) -> tuple[datetime, datetime]:
    """Return UTC start/end-of-day timestamps for the given local date.

    Conversations are stored in UTC; the user's "today" is a local-time
    concept. Without this, a recording made at 23:30 Paris time would
    cross-pollute into the next UTC day's summary.
    """
    try:
        tz = ZoneInfo(tz_label)
    except Exception:
        tz = ZoneInfo("UTC")
    start_local = datetime.combine(d, time.min, tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(ZoneInfo("UTC")), end_local.astimezone(ZoneInfo("UTC"))


def _fetch_eligible(
    user_id: str,
    d: date_cls,
    threshold: float,
) -> list[Conversation]:
    """Conversations on ``d`` whose effective quality is >= ``threshold``,
    ordered by start time."""
    start_utc, end_utc = _utc_day_bounds(d, settings.local_timezone)
    with Session(engine) as db:
        rows = list(
            db.exec(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .where(Conversation.started_at >= start_utc)
                .where(Conversation.started_at < end_utc)
                .order_by(Conversation.started_at)
            ).all()
        )
    return [c for c in rows if _effective_quality(c) >= threshold]


@dataclass
class _ConvDigest:
    """Pre-rendered single-conversation block for the LLM context."""
    conv_id: UUID
    started_at: datetime
    rendered: str


def _render_conversation(
    db: Session, c: Conversation, tz: ZoneInfo
) -> _ConvDigest:
    """Format one conversation into a few lines for the daily-summary prompt.

    Includes events and action items inline, plus mentioned people, so the
    LLM has enough hooks to recognise cross-conversation patterns ("Marie
    came up in three conversations, twice about her job").
    """
    # SQLite + SQLAlchemy sometimes returns naive datetimes (tz info gets
    # stripped on the round-trip even if we wrote tz-aware). Without this
    # guard, .astimezone(tz) on a naive datetime treats it as local-system
    # time, which silently corrupts the display hour for anyone running
    # the server in a non-UTC OS timezone. Production writes UTC, so treat
    # naive == UTC.
    started = c.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=ZoneInfo("UTC"))
    started = started.astimezone(tz)
    parts = [f"[{started:%H:%M}] {c.title or '(untitled)'}: {c.summary or '(no summary)'}"]
    events = list(
        db.exec(
            select(CalendarEvent).where(CalendarEvent.conversation_id == c.id)
        ).all()
    )
    if events:
        parts.append("  events:")
        for e in events:
            when = f" {e.starts_at:%Y-%m-%d %H:%M}" if e.starts_at else ""
            parts.append(f"    - {e.title}{when}")
    actions = list(
        db.exec(
            select(ActionItem)
            .where(ActionItem.conversation_id == c.id)
            .where(ActionItem.status == ActionItemStatus.open)
        ).all()
    )
    if actions:
        parts.append("  open action items:")
        for a in actions:
            owner = f" ({a.owner})" if a.owner else ""
            parts.append(f"    - {a.text}{owner}")
    people = list(
        db.exec(
            select(PersonMention).where(PersonMention.conversation_id == c.id)
        ).all()
    )
    if people:
        names = ", ".join(p.name for p in people)
        parts.append(f"  people mentioned: {names}")
    return _ConvDigest(
        conv_id=c.id,
        started_at=c.started_at,
        rendered="\n".join(parts),
    )


def _build_context(conversations: list[Conversation], d: date_cls) -> str:
    """Render the user-message body for the LLM call."""
    try:
        tz = ZoneInfo(settings.local_timezone)
    except Exception:
        tz = ZoneInfo("UTC")
    with Session(engine) as db:
        digests = [_render_conversation(db, c, tz) for c in conversations]
    header = f"Date: {d.isoformat()} ({settings.local_timezone}).\n"
    return header + "\n\nConversations:\n\n" + "\n\n".join(d.rendered for d in digests)


# ──────────────────────────────────────────────────────────────────────────────
# Parser — narrow because the prompt only asks for one field
# ──────────────────────────────────────────────────────────────────────────────

def _parse_narrative(text: str) -> str:
    """Tolerant: handles ```fences, leading <think>, json_repair fallback."""
    s = text.strip()
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl >= 0:
            s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    s = s.strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        try:
            import json_repair  # type: ignore[import-untyped]

            obj = json_repair.loads(s)
        except Exception:  # noqa: BLE001
            obj = None
    if not isinstance(obj, dict):
        raise ValueError(f"LLM output not a JSON object: {text[:200]!r}")
    narrative = obj.get("narrative")
    if not isinstance(narrative, str) or not narrative.strip():
        raise ValueError(f"missing 'narrative' field in LLM output: {text[:200]!r}")
    return narrative.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DailyResult:
    """Returned by ``generate``. ``narrative`` is None when there were no
    eligible conversations on that day (sparse day, no day-level summary
    necessary). Caller still writes a DailySummary row so the UI can show
    "nothing substantive on this day" rather than an empty-state prompt."""
    narrative: str | None
    conversation_ids: list[UUID]
    quality_threshold: float


async def generate(
    user_id: str,
    d: date_cls,
    *,
    quality_threshold: float = 0.3,
) -> DailyResult:
    """Fetch eligible conversations, call LLM, return narrative.

    Doesn't persist anything — the caller (web route) decides whether to
    overwrite the cached row. Keeps this function unit-testable in
    isolation from the DailySummary table.
    """
    eligible = _fetch_eligible(user_id, d, quality_threshold)
    if not eligible:
        return DailyResult(
            narrative=None,
            conversation_ids=[],
            quality_threshold=quality_threshold,
        )

    if not settings.llm_base_url:
        raise LLMError("OMILOG_LLM_BASE_URL not configured")

    context = _build_context(eligible, d)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]
    chat = await chat_json(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout_s=settings.llm_timeout_s,
    )
    narrative = _parse_narrative(chat.text)
    return DailyResult(
        narrative=narrative,
        conversation_ids=[c.id for c in eligible],
        quality_threshold=quality_threshold,
    )


def store(user_id: str, d: date_cls, result: DailyResult) -> DailySummary:
    """Persist a DailyResult as a DailySummary row.

    One row per (user_id, date) — existing row gets replaced atomically.
    Returns the stored row.
    """
    date_str = d.isoformat()
    with Session(engine) as db:
        existing = db.exec(
            select(DailySummary)
            .where(DailySummary.user_id == user_id)
            .where(DailySummary.date == date_str)
        ).first()
        if existing is not None:
            db.delete(existing)
            db.flush()
        row = DailySummary(
            user_id=user_id,
            date=date_str,
            narrative=result.narrative or "",
            conversation_ids_json=json.dumps(
                [str(cid) for cid in result.conversation_ids]
            ),
            conversation_count=len(result.conversation_ids),
            quality_threshold=result.quality_threshold,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def get_cached(user_id: str, d: date_cls) -> DailySummary | None:
    """Return the cached row for this day, or None if never generated."""
    date_str = d.isoformat()
    with Session(engine) as db:
        return db.exec(
            select(DailySummary)
            .where(DailySummary.user_id == user_id)
            .where(DailySummary.date == date_str)
        ).first()


def conversation_ids_for(row: DailySummary) -> list[UUID]:
    """Decode the conversation_ids_json blob into UUIDs. Tolerant of
    malformed data — returns empty rather than blowing up the page."""
    try:
        ids = json.loads(row.conversation_ids_json)
    except (json.JSONDecodeError, TypeError):
        return []
    out: list[UUID] = []
    for raw in ids:
        if not isinstance(raw, str):
            continue
        try:
            out.append(UUID(raw))
        except ValueError:
            continue
    return out


def list_recent(user_id: str, *, days: int = 14) -> list[DailySummary]:
    """Last N days of summaries, most-recent first. Drives the /daily index
    page once it's wired in (currently we only ship the per-day view)."""
    with Session(engine) as db:
        return list(
            db.exec(
                select(DailySummary)
                .where(DailySummary.user_id == user_id)
                .order_by(DailySummary.date.desc())
                .limit(days)
            ).all()
        )


def expose_for_tests() -> dict[str, Any]:
    """Test helper exposing the otherwise-private formatting + parsing
    functions, so test_daily.py can exercise them without an LLM call."""
    return {
        "_effective_quality": _effective_quality,
        "_utc_day_bounds": _utc_day_bounds,
        "_fetch_eligible": _fetch_eligible,
        "_render_conversation": _render_conversation,
        "_build_context": _build_context,
        "_parse_narrative": _parse_narrative,
    }
