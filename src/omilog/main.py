import asyncio
import logging
import threading
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
    ics_feed,
    people,
    stubs,
)
from .config import assert_runtime_secrets, settings
from .db import init_db
from .pipeline.runner import run_forever
from .web import auth as web_auth
from .web import eval_ui as web_eval_ui
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


def _start_pipeline_thread(
    log: logging.Logger,
) -> tuple[threading.Thread, dict]:
    """Spawn the pipeline runner in a dedicated daemon thread with its own
    asyncio loop. Returns ``(thread, ctl)`` where ``ctl`` carries the
    loop + stop event so the caller can signal shutdown via
    ``ctl['loop'].call_soon_threadsafe(ctl['stop'].set)``.

    Running the pipeline in its own loop is the difference between a Pi
    that feels locked up while STT/diarize/LLM work is in flight and one
    that stays responsive. The web server's loop is free, the pipeline's
    sync chunks (multipart encoding, ONNX inference, JSON parsing) hold
    the GIL only as long as Python actually needs it, and the OS scheduler
    timeshares CPU between the two threads.
    """
    ctl: dict = {}
    ready = threading.Event()

    def target() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            stop_event = asyncio.Event()
            ctl["loop"] = loop
            ctl["stop"] = stop_event
            ready.set()
            loop.run_until_complete(run_forever(stop_event))
        except Exception:
            log.exception("pipeline thread crashed")
        finally:
            loop.close()

    thread = threading.Thread(
        target=target, daemon=True, name="omilog-pipeline"
    )
    thread.start()
    if not ready.wait(timeout=5.0):
        # Thread didn't reach the ready signal — something blew up before
        # it could populate ctl. Caller's join will time out cleanly.
        raise RuntimeError("pipeline thread failed to initialise within 5s")
    return thread, ctl


def _stop_pipeline_thread(thread: threading.Thread, ctl: dict) -> None:
    """Counterpart to _start_pipeline_thread. call_soon_threadsafe is the
    canonical way to set an asyncio.Event from outside its owning loop —
    direct .set() would touch the loop's internals from the wrong thread."""
    loop = ctl.get("loop")
    stop_event = ctl.get("stop")
    if loop is not None and stop_event is not None and not loop.is_closed():
        loop.call_soon_threadsafe(stop_event.set)
    thread.join(timeout=15.0)


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

    if settings.pipeline_in_thread:
        log.info("pipeline: running in dedicated thread (loop isolation)")
        thread, ctl = _start_pipeline_thread(log)
        try:
            yield
        finally:
            _stop_pipeline_thread(thread, ctl)
    else:
        # Legacy in-loop behavior. Web requests share scheduling with
        # every pipeline tick — fine when there's no STT/diarize work,
        # janky during the heavy phases on a small box. Kept around for
        # debugging only; see settings.pipeline_in_thread docstring.
        log.info("pipeline: running in main asyncio loop (legacy mode)")
        stop = asyncio.Event()
        runner = asyncio.create_task(
            run_forever(stop), name="omilog-pipeline-runner"
        )
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
app.include_router(ics_feed.router)
app.include_router(stubs.router)
app.include_router(web_routes.router)
app.include_router(web_eval_ui.router)
