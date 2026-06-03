"""/config UI tests.

The endpoint preserves comments and unrelated keys, writes both editable
keys it knew about and ignores anything else the form might've sent.
"""

from pathlib import Path

from fastapi.testclient import TestClient


def _login(client: TestClient, password: str) -> None:
    client.post("/login", data={"username": "test", "password": password})


def test_config_page_renders_with_current_values(client: TestClient, password: str):
    _login(client, password)
    r = client.get("/config")
    assert r.status_code == 200
    # Section header rendered
    assert "STT (whisper.cpp)" in r.text
    assert "LLM (llama.cpp)" in r.text
    assert "VAD" in r.text
    # A known key shows up in the form
    assert "OMILOG_VAD_THRESHOLD_DB" in r.text
    assert "OMILOG_LLM_MAX_TOKENS" in r.text


def test_config_page_requires_auth(client: TestClient):
    r = client.get("/config", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_config_save_updates_env(client: TestClient, password: str, tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# my settings\n"
        "OMILOG_USERNAME=test\n"
        "OMILOG_PASSWORD_HASH=abc\n"
        "OMILOG_JWT_SECRET=xyz\n"
        "OMILOG_VAD_THRESHOLD_DB=-30\n"
        "OMILOG_VAD_ENABLED=true\n"
        "# trailing comment\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _login(client, password)

    # Submit a minimal form: just the keys we want to update; other unmentioned
    # text/number fields land as empty strings (acceptable — user intent).
    r = client.post(
        "/config",
        data={
            "OMILOG_VAD_THRESHOLD_DB": "-45",
            "OMILOG_LLM_BASE_URL": "http://gpu.tailnet:8081/v1",
            "OMILOG_LLM_MAX_TOKENS": "8192",
            # Checkbox NOT present in form = unchecked. Should write "false".
            # "OMILOG_VAD_ENABLED": ...
            "OMILOG_LOG_LEVEL": "DEBUG",
        },
    )
    assert r.status_code == 200, r.text
    assert "Saved" in r.text

    lines = env_file.read_text(encoding="utf-8").splitlines()
    # Comments preserved
    assert "# my settings" in lines
    assert "# trailing comment" in lines
    # Secrets we never expose are untouched
    assert "OMILOG_USERNAME=test" in lines
    assert "OMILOG_PASSWORD_HASH=abc" in lines
    assert "OMILOG_JWT_SECRET=xyz" in lines
    # Updated values in place
    assert "OMILOG_VAD_THRESHOLD_DB=-45" in lines
    assert "OMILOG_LLM_BASE_URL=http://gpu.tailnet:8081/v1" in lines
    assert "OMILOG_LLM_MAX_TOKENS=8192" in lines
    assert "OMILOG_LOG_LEVEL=DEBUG" in lines
    # Checkbox absent → false written
    assert "OMILOG_VAD_ENABLED=false" in lines


def test_config_save_rejects_newlines(client: TestClient, password: str, tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OMILOG_USERNAME=test\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _login(client, password)
    r = client.post(
        "/config",
        data={"OMILOG_STT_BASE_URL": "http://x\nmalicious"},
    )
    assert r.status_code == 400


def test_config_save_requires_auth(client: TestClient):
    r = client.post(
        "/config",
        data={"OMILOG_STT_BASE_URL": "http://x"},
        follow_redirects=False,
    )
    assert r.status_code == 303
