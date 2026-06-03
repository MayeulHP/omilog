"""Conversation list + detail.

The detail endpoint bundles transcript, events, action items, and people for
one conversation — the "one screen of context" anything consuming this API
(the future web UI, a future MCP server) will want.
"""

import json
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from ..auth import current_user
from ..db import engine
from ..models import (
    ActionItem,
    CalendarEvent,
    Conversation,
    PersonMention,
    Transcript,
)

router = APIRouter(prefix="/api", tags=["conversations"])


def _conv_summary(c: Conversation) -> dict[str, Any]:
    return {
        "id": str(c.id),
        "audio_session_id": str(c.audio_session_id),
        "title": c.title,
        "summary": c.summary,
        "topics": json.loads(c.topics_json) if c.topics_json else [],
        "extraction_repaired": c.extraction_repaired,
        "started_at": c.started_at.isoformat() if c.started_at else None,
        "ended_at": c.ended_at.isoformat() if c.ended_at else None,
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }


def _event_dict(e: CalendarEvent) -> dict[str, Any]:
    return {
        "id": str(e.id),
        "title": e.title,
        "starts_at": e.starts_at.isoformat() if e.starts_at else None,
        "ends_at": e.ends_at.isoformat() if e.ends_at else None,
        "location": e.location,
        "attendees": json.loads(e.attendees_json) if e.attendees_json else [],
        "confidence": e.confidence,
        "exported_to_ics": e.exported_to_ics,
    }


def _action_dict(a: ActionItem) -> dict[str, Any]:
    return {
        "id": str(a.id),
        "text": a.text,
        "owner": a.owner,
        "due_at": a.due_at.isoformat() if a.due_at else None,
        "status": a.status.value,
    }


def _person_dict(p: PersonMention) -> dict[str, Any]:
    return {
        "id": str(p.id),
        "name": p.name,
        "context": p.context,
    }


def _transcript_dict(t: Transcript) -> dict[str, Any]:
    return {
        "id": str(t.id),
        "text": t.text,
        "segments": json.loads(t.segments_json) if t.segments_json else [],
        "language": t.language,
        "model": t.model,
        "created_at": t.created_at.isoformat() if t.created_at else None,
    }


@router.get("/conversations")
async def list_conversations(
    user: Annotated[str, Depends(current_user)],
    limit: int = Query(default=50, le=200, ge=1),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    with Session(engine) as db:
        rows = db.exec(
            select(Conversation)
            .where(Conversation.user_id == user)
            .order_by(Conversation.started_at.desc())
            .offset(offset)
            .limit(limit)
        ).all()
    return [_conv_summary(c) for c in rows]


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: UUID,
    user: Annotated[str, Depends(current_user)],
) -> dict[str, Any]:
    with Session(engine) as db:
        conv = db.get(Conversation, conversation_id)
        if conv is None or conv.user_id != user:
            raise HTTPException(404, "conversation not found")
        transcript = db.exec(
            select(Transcript)
            .where(Transcript.audio_session_id == conv.audio_session_id)
            .order_by(Transcript.created_at.desc())
            .limit(1)
        ).first()
        events = db.exec(
            select(CalendarEvent).where(CalendarEvent.conversation_id == conv.id)
        ).all()
        actions = db.exec(
            select(ActionItem).where(ActionItem.conversation_id == conv.id)
        ).all()
        people = db.exec(
            select(PersonMention).where(PersonMention.conversation_id == conv.id)
        ).all()

    out = _conv_summary(conv)
    out["transcript"] = _transcript_dict(transcript) if transcript else None
    out["calendar_events"] = [_event_dict(e) for e in events]
    out["action_items"] = [_action_dict(a) for a in actions]
    out["people_mentioned"] = [_person_dict(p) for p in people]
    return out
