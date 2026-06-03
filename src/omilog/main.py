import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import (
    action_items,
    audio_upload,
    audio_ws,
    auth,
    conversations,
    events,
    health,
    people,
    stubs,
)
from .config import assert_runtime_secrets, settings
from .db import init_db
from .pipeline.runner import run_forever
from .web import auth as web_auth
from .web import routes as web_routes


def _configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _ui_url() -> str:
    """Display URL for the startup banner. Maps wildcard binds to localhost so
    the line is clickable in a terminal; if the user bound to 0.0.0.0, the
    note below mentions the LAN/tailnet implication."""
    host = settings.host
    if host in ("0.0.0.0", "::"):
        host = "localhost"
    return f"http://{host}:{settings.port}/"


@asynccontextmanager
async def lifespan(_: FastAPI):
    _configure_logging()
    assert_runtime_secrets()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    log = logging.getLogger("omilog")
    log.info(
        "omilog up: storage=%s db=%s",
        settings.storage_dir,
        settings.db_path,
    )
    log.info("omilog: web UI at %s", _ui_url())
    if settings.host == "0.0.0.0":
        log.info("        (also reachable on this machine's LAN / tailnet IP)")

    stop = asyncio.Event()
    runner = asyncio.create_task(run_forever(stop), name="omilog-pipeline-runner")

    try:
        yield
    finally:
        stop.set()
        runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass


app = FastAPI(title="omilog", version="0.1.0", lifespan=lifespan)

app.mount(
    "/static",
    StaticFiles(directory=str(Path(__file__).parent / "web" / "static")),
    name="static",
)

web_auth.install_handler(app)

app.include_router(auth.router)
app.include_router(health.router)
app.include_router(audio_ws.router)
app.include_router(audio_upload.router)
app.include_router(conversations.router)
app.include_router(events.router)
app.include_router(action_items.router)
app.include_router(people.router)
app.include_router(stubs.router)
app.include_router(web_routes.router)
