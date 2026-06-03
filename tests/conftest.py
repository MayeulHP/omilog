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

from omilog.db import init_db  # noqa: E402
from omilog.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _init_schema():
    Path(os.environ["OMILOG_STORAGE_DIR"]).mkdir(parents=True, exist_ok=True)
    init_db()


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
