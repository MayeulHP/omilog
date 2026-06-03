from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ..auth import current_user

router = APIRouter(prefix="/api", tags=["conversations"])


@router.get("/conversations")
async def list_conversations(_user: Annotated[str, Depends(current_user)]) -> list:
    return []


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    _user: Annotated[str, Depends(current_user)],
):
    raise HTTPException(404, "Not found")
