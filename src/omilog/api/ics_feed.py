"""ICS endpoints.

Two surfaces:
  GET /calendar.ics?token=…  → token-gated full feed (calendar apps subscribe)
  GET /events/{id}/download.ics → cookie-auth one-off download

The feed is intentionally token-gated rather than cookie-auth'd because no
mainstream calendar app supports cookie auth on subscribed calendars.
"""

import logging
import secrets
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlmodel import Session, select

from ..auth import current_user
from ..config import settings
from ..db import engine
from ..ics import build_single_event_calendar, build_vcalendar
from ..models import CalendarEvent, Conversation

router = APIRouter(tags=["ics"])
logger = logging.getLogger("omilog.ics")


@router.get("/calendar.ics")
async def calendar_feed(
    token: str = Query(default=""),
    min_confidence: float | None = Query(default=None, ge=0.0, le=1.0),
):
    if not settings.ics_feed_token:
        raise HTTPException(
            403, "ICS feed disabled (set OMILOG_ICS_FEED_TOKEN to enable)"
        )
    if not token or not secrets.compare_digest(token, settings.ics_feed_token):
        raise HTTPException(403, "invalid token")

    threshold = (
        min_confidence
        if min_confidence is not None
        else settings.ics_feed_min_confidence
    )
    with Session(engine) as db:
        rows = db.exec(
            select(CalendarEvent, Conversation)
            .join(Conversation, CalendarEvent.conversation_id == Conversation.id)
            .where(CalendarEvent.confidence >= threshold)
            .where(CalendarEvent.starts_at.is_not(None))
            .order_by(CalendarEvent.starts_at.asc())
        ).all()

    body = build_vcalendar(
        ((evt, str(conv.id), conv.title) for evt, conv in rows),
        prodid=settings.ics_prodid,
        calname=settings.ics_calname,
    )
    logger.info("ics: served feed with %d events", len(rows))
    return Response(
        content=body,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": 'inline; filename="omilog.ics"'},
    )


@router.get("/events/{event_id}/download.ics")
async def event_download(
    event_id: UUID,
    user: Annotated[str, Depends(current_user)],
):
    with Session(engine) as db:
        row = db.exec(
            select(CalendarEvent, Conversation)
            .join(Conversation, CalendarEvent.conversation_id == Conversation.id)
            .where(CalendarEvent.id == event_id)
            .where(Conversation.user_id == user)
        ).first()
        if row is None:
            raise HTTPException(404, "event not found")
        evt, conv = row

    if evt.starts_at is None:
        raise HTTPException(400, "event has no start time, cannot export")

    body = build_single_event_calendar(
        evt,
        conversation_id=str(conv.id),
        conversation_title=conv.title,
        prodid=settings.ics_prodid,
        calname=settings.ics_calname,
    )

    # Mark exported so the UI can show a visual indicator.
    with Session(engine) as db:
        fresh = db.get(CalendarEvent, event_id)
        if fresh is not None:
            fresh.exported_to_ics = True
            db.add(fresh)
            db.commit()

    safe_title = "".join(c for c in (evt.title or "event") if c.isalnum() or c in " -_")[:60]
    safe_title = safe_title.strip().replace(" ", "_") or "event"
    filename = f"{safe_title}-{evt.id}.ics"

    return Response(
        content=body,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
