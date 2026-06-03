"""Aggregated people mentions (proto-CRM).

For each name, returns the count of mentions and the most recent context.
Trivial aggregation in Python — fine until we have thousands of conversations.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from ..auth import current_user
from ..db import engine
from ..models import Conversation, PersonMention

router = APIRouter(prefix="/api", tags=["people"])


@router.get("/people")
async def list_people(
    user: Annotated[str, Depends(current_user)],
    limit: int = Query(default=200, le=1000, ge=1),
) -> list[dict[str, Any]]:
    with Session(engine) as db:
        stmt = (
            select(PersonMention, Conversation)
            .join(Conversation, PersonMention.conversation_id == Conversation.id)
            .where(Conversation.user_id == user)
            .order_by(PersonMention.mentioned_at.desc())
            .limit(limit * 10)  # over-fetch since we aggregate after
        )
        rows = db.exec(stmt).all()

    buckets: dict[str, dict[str, Any]] = {}
    for mention, conv in rows:
        key = mention.name.strip().lower()
        if not key:
            continue
        slot = buckets.setdefault(
            key,
            {
                "name": mention.name,
                "mention_count": 0,
                "latest_context": None,
                "latest_conversation_id": None,
                "latest_conversation_title": None,
                "latest_mentioned_at": None,
            },
        )
        slot["mention_count"] += 1
        if slot["latest_mentioned_at"] is None:
            # Rows are already ordered desc; the first one we see is the latest.
            slot["latest_context"] = mention.context
            slot["latest_conversation_id"] = str(conv.id)
            slot["latest_conversation_title"] = conv.title
            slot["latest_mentioned_at"] = mention.mentioned_at.isoformat()

    out = sorted(buckets.values(), key=lambda r: r["mention_count"], reverse=True)
    return out[:limit]
