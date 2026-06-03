"""Extracted calendar events.

`?upcoming=true` filters to events with starts_at >= now (UTC). Useful for
"what's on the radar" views; pair with the future ICS export script to push
to a real calendar.
"""

import json
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from ..auth import current_user
from ..db import engine
from ..models import CalendarEvent, Conversation

router = APIRouter(prefix="/api", tags=["events"])


@router.get("/events")
async def list_events(
    user: Annotated[str, Depends(current_user)],
    upcoming: bool = Query(default=False),
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=100, le=500, ge=1),
) -> list[dict[str, Any]]:
    with Session(engine) as db:
        # Join through conversations so we filter by user_id.
        stmt = (
            select(CalendarEvent, Conversation)
            .join(Conversation, CalendarEvent.conversation_id == Conversation.id)
            .where(Conversation.user_id == user)
            .where(CalendarEvent.confidence >= min_confidence)
        )
        if upcoming:
            stmt = stmt.where(CalendarEvent.starts_at >= datetime.now(timezone.utc))
        stmt = stmt.order_by(CalendarEvent.starts_at.asc().nullslast()).limit(limit)
        rows = db.exec(stmt).all()

    out = []
    for evt, conv in rows:
        out.append(
            {
                "id": str(evt.id),
                "conversation_id": str(conv.id),
                "conversation_title": conv.title,
                "title": evt.title,
                "starts_at": evt.starts_at.isoformat() if evt.starts_at else None,
                "ends_at": evt.ends_at.isoformat() if evt.ends_at else None,
                "location": evt.location,
                "attendees": json.loads(evt.attendees_json) if evt.attendees_json else [],
                "confidence": evt.confidence,
                "exported_to_ics": evt.exported_to_ics,
            }
        )
    return out
