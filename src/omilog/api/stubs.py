"""Permissive stubs for endpoints the Chronicle app expects but we haven't
characterized yet. Log the request shape so we can fill these in for real
once we have a mitmproxy capture."""

import logging

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["stubs"])
logger = logging.getLogger("omilog.stubs")


@router.get("/clients")
async def list_clients() -> list:
    return []


@router.post("/clients")
async def register_client(req: Request) -> dict:
    body = await req.body()
    logger.info("POST /api/clients body=%r headers=%s", body[:512], dict(req.headers))
    return {"id": "stub-client", "status": "ok"}


@router.get("/users/me")
async def users_me() -> dict:
    return {"id": "stub-user", "email": "local@omilog"}
