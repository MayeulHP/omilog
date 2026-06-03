from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readiness")
async def readiness() -> dict[str, str]:
    # Phase 0: same shape as /health. Phase 1+: verify DB + GPU host reachability.
    return {"status": "ok"}
