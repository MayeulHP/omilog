"""Set required env vars BEFORE any omilog import.

The `omilog.config.Settings` instance is constructed at module import time, so
the env has to be in place before the first `from omilog... import ...` in
any test.
"""

import os
import tempfile
from pathlib import Path

# Per-session temp dir. Pytest reuses this conftest across all tests, so we
# only set up once.
_TMP = Path(tempfile.mkdtemp(prefix="omilog-tests-"))

os.environ.setdefault("OMILOG_USERNAME", "test")
os.environ.setdefault("OMILOG_JWT_SECRET", "test-secret-not-for-prod")
os.environ.setdefault("OMILOG_DB_PATH", str(_TMP / "test.db"))
os.environ.setdefault("OMILOG_STORAGE_DIR", str(_TMP / "storage"))

# We need a bcrypt hash. Compute one here so tests carry a known plaintext.
import bcrypt  # noqa: E402

TEST_PASSWORD = "correct horse battery staple"
os.environ.setdefault(
    "OMILOG_PASSWORD_HASH",
    bcrypt.hashpw(TEST_PASSWORD.encode("utf-8"), bcrypt.gensalt()).decode("utf-8"),
)

# Now safe to import the app.
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlmodel import Session, delete  # noqa: E402

from omilog.db import engine, init_db  # noqa: E402
from omilog.main import app  # noqa: E402
from omilog.models import (  # noqa: E402
    ActionItem,
    AudioSession,
    CalendarEvent,
    Conversation,
    DailySummary,
    Decision,
    PersonMention,
    Speaker,
    Transcript,
    WakeAction,
    WakeInvocation,
)


@pytest.fixture(scope="session", autouse=True)
def _init_schema():
    Path(os.environ["OMILOG_STORAGE_DIR"]).mkdir(parents=True, exist_ok=True)
    init_db()


@pytest.fixture(autouse=True)
def _isolate_db():
    """Wipe the DB before each test.

    Tests share one SQLite file (per-test SQLite would mean reimporting omilog
    each test, which is expensive). Without a wipe, ordering bugs creep in:
    test_ics seeds rows under user=test, then test_phase0's "/api/conversations
    returns []" assertion fails because someone else's rows are still there.

    Children → parents in FK order.
    """
    with Session(engine) as db:
        # Disable FK checks for the wipe — order of bulk deletes can run afoul
        # of mid-transaction FK enforcement in SQLite even when topologically
        # correct (SQLAlchemy doesn't always flush in the order we'd expect).
        # Tests themselves still run with FKs on.
        db.exec(text("PRAGMA foreign_keys=OFF"))
        for model in (
            WakeInvocation,
            Transcript,
            PersonMention,
            ActionItem,
            CalendarEvent,
            Decision,
            WakeAction,
            Speaker,
            DailySummary,
            Conversation,
            AudioSession,
        ):
            db.exec(delete(model))
        db.exec(text("PRAGMA foreign_keys=ON"))
        db.commit()
    yield


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def password() -> str:
    return TEST_PASSWORD


@pytest.fixture()
def auth_token(client: TestClient) -> str:
    r = client.post(
        "/auth/jwt/login",
        data={"username": "test", "password": TEST_PASSWORD},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]
