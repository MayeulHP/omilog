"""LLM system prompt override + /config/prompt editor tests."""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from omilog.config import settings
from omilog.pipeline import extract


def _login(client: TestClient, password: str) -> None:
    client.post("/login", data={"username": "test", "password": password})


# ──────────────────────────────────────────────────────────────────────────────
# build_system_prompt with file override
# ──────────────────────────────────────────────────────────────────────────────

def test_override_file_present_replaces_default(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("custom prompt yo", encoding="utf-8")
    assert extract.build_system_prompt("French", p) == "custom prompt yo"


def test_override_file_missing_falls_back_to_default(tmp_path):
    p = tmp_path / "does-not-exist.txt"
    rendered = extract.build_system_prompt("French", p)
    assert "Conversations are most often in French." in rendered
    assert "[USER]" in rendered  # default speaker label boilerplate


def test_override_file_empty_falls_back_to_default(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("   \n  \n", encoding="utf-8")
    rendered = extract.build_system_prompt("French", p)
    assert "Conversations are most often in French." in rendered


def test_build_messages_threads_override_path(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("OVERRIDE ME", encoding="utf-8")
    msgs = extract.build_messages(
        transcript_text="hi",
        transcript_segments=[{"start": 0.0, "text": "hi"}],
        now=datetime(2026, 6, 4, 10, 0),
        timezone_label="UTC",
        primary_language="French",  # should be ignored
        system_prompt_override_path=p,
    )
    assert msgs[0]["content"] == "OVERRIDE ME"


# ──────────────────────────────────────────────────────────────────────────────
# /config/prompt editor
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def prompt_path(tmp_path, monkeypatch):
    """Sandbox the prompt file location for each test."""
    p = tmp_path / "prompts" / "system_prompt.txt"
    monkeypatch.setattr(settings, "llm_system_prompt_file", p, raising=False)
    return p


def test_prompt_page_requires_auth(client: TestClient, prompt_path):
    r = client.get("/config/prompt", follow_redirects=False)
    assert r.status_code == 303


def test_prompt_page_shows_default_when_file_missing(
    client: TestClient, password: str, prompt_path
):
    _login(client, password)
    r = client.get("/config/prompt")
    assert r.status_code == 200
    assert "Default in use" in r.text
    # Default prompt sample line should be in the textarea contents.
    assert "[USER]" in r.text


def test_prompt_page_shows_customized_when_file_present(
    client: TestClient, password: str, prompt_path
):
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("my override", encoding="utf-8")
    _login(client, password)
    r = client.get("/config/prompt")
    assert r.status_code == 200
    assert "Customized" in r.text
    assert "my override" in r.text


def test_prompt_save_writes_file(client: TestClient, password: str, prompt_path):
    _login(client, password)
    r = client.post(
        "/config/prompt",
        data={"prompt": "the new prompt", "action": "save"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"] == "/config/prompt"
    assert prompt_path.exists()
    assert prompt_path.read_text(encoding="utf-8") == "the new prompt"


def test_prompt_reset_deletes_file(client: TestClient, password: str, prompt_path):
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("override that will go", encoding="utf-8")
    _login(client, password)
    r = client.post(
        "/config/prompt",
        data={"prompt": "ignored", "action": "reset"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert not prompt_path.exists()


def test_prompt_empty_save_treated_as_reset(
    client: TestClient, password: str, prompt_path
):
    """Saving an all-whitespace prompt deletes the override rather than storing
    a no-op file. Otherwise the user would see "Customized" but the runtime
    would fall back to default anyway, which would be confusing."""
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text("something", encoding="utf-8")
    _login(client, password)
    r = client.post(
        "/config/prompt",
        data={"prompt": "   \n\n   ", "action": "save"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert not prompt_path.exists()
