"""Extracted action items.

Default lists status=open. Pass `?status=done` / `?status=dismissed` to filter
explicitly, or `?status=all` to skip the filter.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from ..auth import current_user
from ..db import engine
from ..models import ActionItem, ActionItemStatus, Conversation

router = APIRouter(prefix="/api", tags=["action-items"])


@router.get("/action-items")
async def list_action_items(
    user: Annotated[str, Depends(current_user)],
    status: str = Query(default="open"),
    limit: int = Query(default=100, le=500, ge=1),
) -> list[dict[str, Any]]:
    with Session(engine) as db:
        stmt = (
            select(ActionItem, Conversation)
            .join(Conversation, ActionItem.conversation_id == Conversation.id)
            .where(Conversation.user_id == user)
        )
        if status != "all":
            try:
                target = ActionItemStatus(status)
            except ValueError:
                target = ActionItemStatus.open
            stmt = stmt.where(ActionItem.status == target)
        stmt = stmt.order_by(ActionItem.due_at.asc().nullslast()).limit(limit)
        rows = db.exec(stmt).all()

    return [
        {
            "id": str(item.id),
            "conversation_id": str(conv.id),
            "conversation_title": conv.title,
            "text": item.text,
            "owner": item.owner,
            "due_at": item.due_at.isoformat() if item.due_at else None,
            "status": item.status.value,
        }
        for item, conv in rows
    ]
