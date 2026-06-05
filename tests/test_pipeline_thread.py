"""Pipeline-in-thread tests.

The default-on threaded pipeline isolates the worker from the web
server's asyncio loop, so heavy STT / diarize / LLM phases don't make
the UI feel locked. The conftest force-disables the thread mode for
the rest of the suite (shutdown overhead). These tests explicitly
opt in via monkeypatch.
"""

from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from omilog.config import settings
from omilog.main import app


def _pipeline_threads_alive() -> list[threading.Thread]:
    return [
        t for t in threading.enumerate()
        if t.is_alive() and t.name == "omilog-pipeline"
    ]


def test_pipeline_thread_starts_and_stops(monkeypatch):
    """When pipeline_in_thread is on, lifespan startup must spawn a daemon
    thread named 'omilog-pipeline', and lifespan shutdown must join it."""
    monkeypatch.setattr(settings, "pipeline_in_thread", True)

    # Sanity: no leftover threads from a prior test.
    leftover = _pipeline_threads_alive()
    for t in leftover:
        # If anything's still alive from before, let it die before counting.
        t.join(timeout=2.0)

    with TestClient(app) as client:
        # The thread shows up in the global enumeration. (Not guaranteed
        # to be the SAME object across iterations, so we look by name.)
        alive = _pipeline_threads_alive()
        assert any(t.daemon for t in alive), (
            "expected at least one daemon 'omilog-pipeline' thread alive "
            "during the lifespan, found: "
            f"{[(t.name, t.daemon, t.is_alive()) for t in alive]}"
        )
        # And the web still works while the pipeline is alive.
        r = client.get("/login")
        assert r.status_code == 200

    # After lifespan shutdown, give the thread up to a second to drain
    # and then assert it's gone. The lifespan calls join(timeout=15) so
    # in practice it's already gone by the time we get here.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and _pipeline_threads_alive():
        time.sleep(0.05)
    assert _pipeline_threads_alive() == [], (
        "pipeline thread should have exited by lifespan shutdown"
    )


def test_pipeline_legacy_mode_does_not_spawn_thread(monkeypatch):
    """The opt-out path keeps the pipeline in the main asyncio loop —
    no 'omilog-pipeline' thread should appear."""
    monkeypatch.setattr(settings, "pipeline_in_thread", False)

    # Drain anything still alive from prior tests.
    for t in _pipeline_threads_alive():
        t.join(timeout=2.0)

    with TestClient(app):
        alive = _pipeline_threads_alive()
        assert alive == [], (
            "no pipeline thread should exist in legacy in-loop mode, "
            f"got: {[t.name for t in alive]}"
        )


def test_pipeline_thread_survives_a_web_request_burst(monkeypatch):
    """End-to-end smoke test of the isolation: 20 quick GETs to /login
    while the pipeline is idle. All succeed, none time out. This is a
    proxy for the original symptom (UI hung while pipeline busy) — if
    something in the threaded setup deadlocks the main loop, this test
    will hang or fail."""
    monkeypatch.setattr(settings, "pipeline_in_thread", True)

    with TestClient(app) as client:
        for _ in range(20):
            r = client.get("/login")
            assert r.status_code == 200
