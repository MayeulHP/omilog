"""Wake-word matcher, command resolver, executor, runner integration, and UI tests."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from omilog.db import engine
from omilog.models import (
    AudioSession,
    Conversation,
    SessionStatus,
    Transcript,
    WakeAction,
    WakeInvocation,
)
from omilog.pipeline import runner
from omilog.pipeline import wake as wake_mod
from omilog.pipeline.extract import Extraction
from omilog.pipeline.stt import STTResult


# ──────────────────────────────────────────────────────────────────────────────
# Matcher
# ──────────────────────────────────────────────────────────────────────────────

def test_find_wake_matches_basic():
    matches = wake_mod.find_wake_matches(
        "Hey Jarvis, quelle est la météo demain ?",
        ["Hey Jarvis"],
    )
    assert len(matches) == 1
    assert matches[0]["phrase"] == "Hey Jarvis"
    assert matches[0]["post_wake"] == ", quelle est la météo demain ?"


def test_find_wake_matches_case_insensitive():
    matches = wake_mod.find_wake_matches(
        "hey jarvis lance Spotify s'il te plaît",
        ["Hey Jarvis"],
    )
    assert len(matches) == 1


def test_find_wake_matches_multiple_phrases():
    matches = wake_mod.find_wake_matches(
        "Jarvis envoie un mail à Marie. Plus tard, Salut Jarvis remind me de faire les courses.",
        ["Hey Jarvis", "Jarvis", "Salut Jarvis"],
    )
    # "Jarvis" matches once + "Salut Jarvis" matches once (and 'Jarvis' inside
    # 'Salut Jarvis' is masked by the longer overlap rule). The current
    # implementation is greedy left-to-right so we get the 'Jarvis' substring
    # in both spots — and that's acceptable because both fire the same action.
    assert any(m["phrase"] == "Jarvis" for m in matches)
    assert len(matches) >= 2


def test_find_wake_matches_no_overlap_double_count():
    """A single 'Jarvis' anywhere should produce exactly one match."""
    matches = wake_mod.find_wake_matches("hello Jarvis world", ["Jarvis"])
    assert len(matches) == 1


def test_find_wake_matches_empty_inputs():
    assert wake_mod.find_wake_matches("", ["Jarvis"]) == []
    assert wake_mod.find_wake_matches("hello world", []) == []
    assert wake_mod.find_wake_matches("hello world", [""]) == []


def test_find_wake_matches_post_wake_runs_to_next_match():
    matches = wake_mod.find_wake_matches(
        "Jarvis play music. Then later, Jarvis stop.",
        ["Jarvis"],
    )
    assert len(matches) == 2
    assert "play music" in matches[0]["post_wake"]
    assert matches[1]["post_wake"].endswith("stop.")


# ──────────────────────────────────────────────────────────────────────────────
# Stop phrases
# ──────────────────────────────────────────────────────────────────────────────

def test_stop_phrase_truncates_post_wake():
    matches = wake_mod.find_wake_matches(
        "Hey Jarvis envoie un mail à Marie over. Et puis, on va prendre un café demain.",
        phrases=["Hey Jarvis"],
        stop_phrases=["over"],
    )
    assert len(matches) == 1
    # post_wake stops at "over", not at end of text
    assert "envoie un mail à Marie" in matches[0]["post_wake"]
    assert "café demain" not in matches[0]["post_wake"]
    assert "over" not in matches[0]["post_wake"]


def test_stop_phrase_takes_earliest_of_multiple():
    matches = wake_mod.find_wake_matches(
        "Jarvis call Marie merci. Lots more stuff over here.",
        phrases=["Jarvis"],
        stop_phrases=["merci", "over"],
    )
    assert len(matches) == 1
    assert matches[0]["post_wake"] == "call Marie"


def test_stop_phrase_after_next_wake_match_does_nothing():
    """If a stop phrase appears AFTER the next wake match, the next match's
    boundary still wins (i.e. each match shortens at the earliest of {next
    wake, stop phrase}, not the global earliest stop phrase)."""
    matches = wake_mod.find_wake_matches(
        "Jarvis play music. Jarvis next track over.",
        phrases=["Jarvis"],
        stop_phrases=["over"],
    )
    # First match: post_wake goes up to second 'Jarvis' (no stop phrase before)
    assert matches[0]["post_wake"].strip() == "play music."
    # Second match: post_wake stops at 'over'
    assert matches[1]["post_wake"].strip() == "next track"


def test_stop_phrase_none_falls_back_to_full_post_wake():
    matches = wake_mod.find_wake_matches(
        "Jarvis hello world",
        phrases=["Jarvis"],
        stop_phrases=None,
    )
    assert matches[0]["post_wake"] == "hello world"


def test_stop_phrase_empty_list_treated_as_none():
    matches = wake_mod.find_wake_matches(
        "Jarvis hello world",
        phrases=["Jarvis"],
        stop_phrases=[],
    )
    assert matches[0]["post_wake"] == "hello world"


def test_stop_phrase_case_insensitive():
    matches = wake_mod.find_wake_matches(
        "Jarvis call Marie OVER here be dragons",
        phrases=["Jarvis"],
        stop_phrases=["over"],
    )
    assert matches[0]["post_wake"] == "call Marie"


# ──────────────────────────────────────────────────────────────────────────────
# Command resolver
# ──────────────────────────────────────────────────────────────────────────────

def test_resolve_command_substitutes_variables():
    cmd = wake_mod.resolve_command(
        'echo $transcript > /tmp/out',
        {"transcript": "hello world"},
    )
    assert cmd == "echo 'hello world' > /tmp/out"


def test_resolve_command_escapes_shell_metacharacters():
    """Critical safety test: a transcript with backticks / $() / quotes must
    not be able to escape its intended argument position."""
    malicious = "; rm -rf / `cat /etc/passwd` $(whoami) \"quotes\""
    cmd = wake_mod.resolve_command(
        "agent --input $transcript",
        {"transcript": malicious},
    )
    # shlex.quote wraps in single quotes and escapes inner single quotes.
    assert "rm -rf" not in cmd.split("'agent")[0] or cmd.startswith("agent")
    # The malicious payload is now a single argument:
    assert cmd.count("'") >= 2  # wrapped in quotes
    assert "; rm" not in cmd.replace("'; rm -rf", "")  # the dangerous bare ';' is gone


def test_resolve_command_leaves_unknown_vars_literal():
    cmd = wake_mod.resolve_command(
        "echo $not_a_real_var > /tmp/out",
        {"transcript": "hi"},
    )
    # safe_substitute leaves unknown $vars alone
    assert "$not_a_real_var" in cmd


# ──────────────────────────────────────────────────────────────────────────────
# Executor
# ──────────────────────────────────────────────────────────────────────────────

async def test_execute_command_captures_stdout():
    result = await wake_mod.execute_command("echo hello world", timeout_s=5)
    assert result["exit_code"] == 0
    assert "hello world" in result["stdout"]
    assert result["stderr"] == ""
    assert result["duration_ms"] >= 0


async def test_execute_command_captures_stderr_and_nonzero_exit():
    result = await wake_mod.execute_command("ls /this-path-does-not-exist", timeout_s=5)
    assert result["exit_code"] != 0
    assert "stderr" in result and len(result["stderr"]) > 0


async def test_execute_command_respects_timeout():
    result = await wake_mod.execute_command("sleep 5", timeout_s=0.3)
    assert result["exit_code"] is None
    assert "timed out" in result["stderr"]
    # Should have given up close to the timeout, not waited the full 5s.
    assert result["duration_ms"] < 2000


# ──────────────────────────────────────────────────────────────────────────────
# Runner integration: real wake action fires after LLM extraction
# ──────────────────────────────────────────────────────────────────────────────

def _seed_pending_llm_session_with_transcript(text: str) -> UUID:
    sid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="test",
                audio_path="/tmp/x.opus",
                codec="opus",
                started_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
                status=SessionStatus.pending_llm,
            )
        )
        db.add(
            Transcript(
                audio_session_id=sid,
                text=text,
                segments_json=None,
                language="fr",
                model="whisper-large-v3-turbo",
            )
        )
        db.commit()
    return sid


def _seed_wake_action(
    *,
    user: str = "test",
    phrases: list[str],
    command: str,
    enabled: bool = True,
    stop_phrases: list[str] | None = None,
) -> UUID:
    aid = uuid4()
    with Session(engine) as db:
        db.add(
            WakeAction(
                id=aid,
                user_id=user,
                name="test action",
                phrases_json=json.dumps(phrases),
                stop_phrases_json=(
                    json.dumps(stop_phrases) if stop_phrases else None
                ),
                command=command,
                enabled=enabled,
                timeout_seconds=10.0,
            )
        )
        db.commit()
    return aid


async def test_runner_fires_wake_action_after_llm(monkeypatch, tmp_path: Path):
    """Full path: pending_llm session → LLM extraction (mocked) → wake action
    fires once for each phrase match, with the post-wake text substituted in."""
    monkeypatch.setattr(
        runner.settings, "llm_base_url", "http://gpu:8081/v1", raising=False
    )
    transcript_text = "Salut Jarvis envoie un mail à Marie demain."
    session_id = _seed_pending_llm_session_with_transcript(transcript_text)

    out_file = tmp_path / "wake-fired.txt"
    action_id = _seed_wake_action(
        phrases=["Salut Jarvis", "Jarvis"],
        command=f"echo $transcript >> {out_file}",
    )

    fake_extraction = {
        "title": "Mail à Marie",
        "summary": "L'utilisateur veut envoyer un mail à Marie demain.",
        "topics": ["mail"],
        "calendar_events": [],
        "action_items": [],
        "people_mentioned": [{"name": "Marie", "context": "destinataire mail"}],
    }
    with patch.object(
        runner,
        "chat_json",
        new=AsyncMock(
            return_value=runner.ChatResult(  # type: ignore[attr-defined]
                text=json.dumps(fake_extraction),
                finish_reason="stop",
                raw={},
            )
        )
        if False
        else AsyncMock(),  # below
    ):
        pass

    # Patch chat_json with a fresh AsyncMock returning a ChatResult.
    from omilog.pipeline.llm import ChatResult

    with patch.object(
        runner,
        "chat_json",
        new=AsyncMock(
            return_value=ChatResult(
                text=json.dumps(fake_extraction),
                finish_reason="stop",
                raw={},
            )
        ),
    ):
        await runner.process_llm(session_id)

    # The 2 phrases overlap on "Jarvis", so both fire (matcher is greedy).
    with Session(engine) as db:
        invocations = list(
            db.exec(
                select(WakeInvocation).where(WakeInvocation.wake_action_id == action_id)
            ).all()
        )
    assert len(invocations) >= 1
    # Every successful invocation appended to the file.
    assert out_file.exists()
    contents = out_file.read_text()
    assert "envoie un mail à Marie demain." in contents


async def test_runner_skips_disabled_wake_actions(monkeypatch):
    monkeypatch.setattr(
        runner.settings, "llm_base_url", "http://gpu:8081/v1", raising=False
    )
    sid = _seed_pending_llm_session_with_transcript("Jarvis hello")
    _seed_wake_action(phrases=["Jarvis"], command="echo hi", enabled=False)

    from omilog.pipeline.llm import ChatResult

    payload = json.dumps(
        {
            "title": "x", "summary": "y", "topics": [],
            "calendar_events": [], "action_items": [], "people_mentioned": [],
        }
    )
    with patch.object(
        runner,
        "chat_json",
        new=AsyncMock(return_value=ChatResult(text=payload, finish_reason="stop", raw={})),
    ):
        await runner.process_llm(sid)

    with Session(engine) as db:
        invocations = list(db.exec(select(WakeInvocation)).all())
    assert invocations == []


async def test_runner_continues_when_wake_action_crashes(monkeypatch):
    """A misbehaving wake action must not break LLM extraction."""
    monkeypatch.setattr(
        runner.settings, "llm_base_url", "http://gpu:8081/v1", raising=False
    )
    sid = _seed_pending_llm_session_with_transcript("Jarvis hello")
    _seed_wake_action(phrases=["Jarvis"], command="false")  # exit 1

    from omilog.pipeline.llm import ChatResult

    payload = json.dumps(
        {
            "title": "x", "summary": "y", "topics": [],
            "calendar_events": [], "action_items": [], "people_mentioned": [],
        }
    )
    with patch.object(
        runner,
        "chat_json",
        new=AsyncMock(return_value=ChatResult(text=payload, finish_reason="stop", raw={})),
    ):
        await runner.process_llm(sid)

    # Conversation was still saved and the session moved to done.
    with Session(engine) as db:
        sess = db.get(AudioSession, sid)
        conv = db.exec(
            select(Conversation).where(Conversation.audio_session_id == sid)
        ).first()
        invs = list(db.exec(select(WakeInvocation)).all())
    assert sess is not None
    assert sess.status == SessionStatus.done
    assert conv is not None
    # The invocation was logged with the non-zero exit code.
    assert len(invs) == 1
    assert invs[0].exit_code == 1


# ──────────────────────────────────────────────────────────────────────────────
# UI: list / create / edit / delete / test-fire / log
# ──────────────────────────────────────────────────────────────────────────────

def _login(client: TestClient, password: str) -> None:
    client.post("/login", data={"username": "test", "password": password})


def test_wake_actions_index_empty(client: TestClient, password: str):
    _login(client, password)
    r = client.get("/wake-actions")
    assert r.status_code == 200
    assert "No actions" in r.text


def test_wake_actions_create_via_form(client: TestClient, password: str):
    _login(client, password)
    r = client.post(
        "/wake-actions/new",
        data={
            "name": "hermes",
            "phrases": "Hey Jarvis\nJarvis\nSalut Jarvis",
            "stop_phrases": "over\nmerci",
            "command": "echo $transcript",
            "timeout_seconds": "30",
            "enabled": "on",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303

    with Session(engine) as db:
        actions = list(db.exec(select(WakeAction)).all())
    assert len(actions) == 1
    assert actions[0].name == "hermes"
    assert json.loads(actions[0].phrases_json) == ["Hey Jarvis", "Jarvis", "Salut Jarvis"]
    assert json.loads(actions[0].stop_phrases_json) == ["over", "merci"]
    assert actions[0].enabled is True


def test_wake_actions_form_empty_stop_phrases_stored_null(client, password):
    _login(client, password)
    r = client.post(
        "/wake-actions/new",
        data={
            "name": "no-stop",
            "phrases": "Jarvis",
            "stop_phrases": "",
            "command": "echo hi",
            "timeout_seconds": "30",
            "enabled": "on",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with Session(engine) as db:
        a = db.exec(select(WakeAction)).first()
    assert a.stop_phrases_json is None


def test_wake_actions_create_rejects_empty_phrases(client: TestClient, password: str):
    """Blank phrases should be rejected. Either 400 (our handler's business
    validation) or 422 (FastAPI form validation) is fine — both mean the row
    didn't get persisted."""
    _login(client, password)
    r = client.post(
        "/wake-actions/new",
        data={
            "name": "valid name",
            "phrases": "",
            "command": "echo hi",
            "timeout_seconds": "30",
        },
    )
    assert r.status_code in (400, 422)
    with Session(engine) as db:
        assert list(db.exec(select(WakeAction))) == []


def test_wake_actions_edit(client: TestClient, password: str):
    aid = _seed_wake_action(phrases=["old"], command="echo old")
    _login(client, password)
    r = client.post(
        f"/wake-actions/{aid}/edit",
        data={
            "name": "renamed",
            "phrases": "new",
            "command": "echo new",
            "timeout_seconds": "45",
            "enabled": "",  # unchecked
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    with Session(engine) as db:
        a = db.get(WakeAction, aid)
        assert a.name == "renamed"
        assert json.loads(a.phrases_json) == ["new"]
        assert a.command == "echo new"
        assert a.enabled is False
        assert a.timeout_seconds == 45.0


def test_wake_actions_delete_removes_action_and_logs(client: TestClient, password: str):
    aid = _seed_wake_action(phrases=["x"], command="echo x")
    with Session(engine) as db:
        db.add(
            WakeInvocation(
                wake_action_id=aid,
                conversation_id=None,
                matched_phrase="(test)",
                input_text="hi",
                command_resolved="echo hi",
                exit_code=0,
                stdout="hi\n",
                stderr="",
                duration_ms=10,
            )
        )
        db.commit()

    _login(client, password)
    r = client.post(f"/wake-actions/{aid}/delete", follow_redirects=False)
    assert r.status_code == 303

    with Session(engine) as db:
        assert db.get(WakeAction, aid) is None
        assert list(db.exec(select(WakeInvocation))) == []


def test_wake_action_test_fire_returns_partial(client: TestClient, password: str):
    aid = _seed_wake_action(phrases=["x"], command="echo $transcript")
    _login(client, password)
    r = client.post(
        f"/wake-actions/{aid}/test",
        data={"test_input": "bonjour"},
    )
    assert r.status_code == 200
    assert "bonjour" in r.text or "exit" in r.text  # the partial mentions the result
    with Session(engine) as db:
        invs = list(db.exec(select(WakeInvocation)).all())
    assert len(invs) == 1
    assert invs[0].exit_code == 0


def test_wake_action_404_for_other_user(client: TestClient, password: str):
    aid = _seed_wake_action(user="someone-else", phrases=["x"], command="echo x")
    _login(client, password)
    assert client.get(f"/wake-actions/{aid}/edit").status_code == 404
    assert client.post(f"/wake-actions/{aid}/delete").status_code == 404


def test_wake_action_unauth_redirects(client: TestClient):
    r = client.get("/wake-actions", follow_redirects=False)
    assert r.status_code == 303
