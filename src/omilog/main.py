import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .api import audio_ws, auth, conversations, health, stubs
from .config import assert_runtime_secrets, settings
from .db import init_db


def _configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    _configure_logging()
    assert_runtime_secrets()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    logging.getLogger("omilog").info(
        "omilog up: storage=%s db=%s",
        settings.storage_dir,
        settings.db_path,
    )
    yield


app = FastAPI(title="omilog", version="0.1.0", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(health.router)
app.include_router(audio_ws.router)
app.include_router(conversations.router)
app.include_router(stubs.router)
