"""The /eval labeling UI + the shared bootstrap core it sits on.

Mirrors the house pattern: no ffmpeg / STT / diarization — the HQ path and
the run button mock the underlying calls.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

import omilog.evals.bootstrap as bootstrap_mod
import omilog.evals.runner as eval_runner
from omilog.config import settings
from omilog.db import engine
from omilog.evals.bootstrap import BootstrapError, create_case
from omilog.evals.cases import load_case, load_cases
from omilog.models import AudioSession, Conversation, SessionStatus, Transcript
from omilog.pipeline.stt import STTResult

SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": "bonjour à tous", "speaker": "USER"},
    {"start": 2.5, "end": 4.0, "text": "salut", "speaker": "S1"},
    {"start": 4.2, "end": 6.0, "text": "ça va"},  # no speaker label
]


@pytest.fixture(autouse=True)
def _eval_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "eval_cases_dir", tmp_path / "cases")
    monkeypatch.setattr(settings, "eval_results_dir", tmp_path / "results")
    return tmp_path


def _seed_session(tmp_path: Path, *, with_conversation: bool = True, audio: bool = True):
    audio_path = tmp_path / f"{uuid4()}.opus"
    if audio:
        audio_path.write_bytes(b"\x00fake-opus")
    sess = AudioSession(
        user_id="test",
        status=SessionStatus.done,
        audio_path=str(audio_path) if audio else None,
        duration_s=6.0,
        codec="opus",
    )
    transcript = Transcript(
        audio_session_id=sess.id,
        text="bonjour à tous\nsalut\nça va",
        segments_json=json.dumps(SEGMENTS),
        language="fr",
    )
    conv = None
    if with_conversation:
        conv = Conversation(
            audio_session_id=sess.id,
            user_id="test",
            title="Test conv",
            started_at=sess.started_at,
        )
    with Session(engine) as db:
        db.add(sess)
        db.add(transcript)
        if conv:
            db.add(conv)
        db.commit()
        db.refresh(sess)
        if conv:
            db.refresh(conv)
    return sess, conv


def _login(client: TestClient, password: str) -> None:
    client.post("/login", data={"username": "test", "password": password})


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap core
# ──────────────────────────────────────────────────────────────────────────────


async def test_create_case_writes_row_based_references(tmp_path):
    sess, _ = _seed_session(tmp_path)
    case_dir = await create_case(sess.id, name="my-case")
    assert case_dir == settings.eval_cases_dir / "my-case"
    assert (case_dir / "audio.opus").read_bytes() == b"\x00fake-opus"

    turns = json.loads((case_dir / "reference_turns.json").read_text())
    assert [t["text"] for t in turns] == ["bonjour à tous", "salut", "ça va"]
    assert turns[0]["speaker"] == "USER"
    assert "speaker" not in turns[2]
    assert (case_dir / "reference.txt").read_text() == "bonjour à tous\nsalut\nça va\n"
    meta = json.loads((case_dir / "case.json").read_text())
    assert meta["verified"] is False
    assert meta["source_session_id"] == str(sess.id)

    # And the harness loads it back.
    (case,) = load_cases(settings.eval_cases_dir)
    assert case.name == "my-case"
    assert len(case.reference_turns) == 3


async def test_create_case_rejects_bad_name_and_collision(tmp_path):
    sess, _ = _seed_session(tmp_path)
    with pytest.raises(BootstrapError, match="name"):
        await create_case(sess.id, name="../escape")
    await create_case(sess.id, name="dup")
    with pytest.raises(BootstrapError, match="already exists"):
        await create_case(sess.id, name="dup")


async def test_create_case_requires_audio_on_disk(tmp_path):
    sess, _ = _seed_session(tmp_path, audio=False)
    with pytest.raises(BootstrapError, match="audio"):
        await create_case(sess.id, name="no-audio")


async def test_create_case_hq_retranscribes_and_keeps_speakers(tmp_path, monkeypatch):
    sess, _ = _seed_session(tmp_path)
    monkeypatch.setattr(settings, "stt_base_url", "http://fake-stt")
    monkeypatch.setattr(
        bootstrap_mod, "transcode_to_wav_bytes", AsyncMock(return_value=b"RIFFwav")
    )
    fresh = [
        {"start": 0.1, "end": 2.4, "text": "bonjour à tous les deux"},
        {"start": 2.6, "end": 3.9, "text": "salut salut"},
    ]
    fake_stt = AsyncMock(
        return_value=STTResult(
            text="bonjour à tous les deux salut salut",
            segments=fresh,
            language="fr",
            raw={},
        )
    )
    monkeypatch.setattr(bootstrap_mod, "transcribe_wav", fake_stt)

    case_dir = await create_case(sess.id, name="hq-case", hq=True)
    turns = json.loads((case_dir / "reference_turns.json").read_text())
    # Fresh text, speakers carried over from the stored segments by overlap.
    assert turns[0]["text"] == "bonjour à tous les deux"
    assert turns[0]["speaker"] == "USER"
    assert turns[1]["speaker"] == "S1"
    meta = json.loads((case_dir / "case.json").read_text())
    assert meta["hq_draft"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Web UI
# ──────────────────────────────────────────────────────────────────────────────


def test_eval_pages_require_login(client: TestClient):
    r = client.get("/eval", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


def test_eval_index_empty_then_lists_case(client: TestClient, password, tmp_path):
    _login(client, password)
    assert "No cases yet" in client.get("/eval").text

    sess, conv = _seed_session(tmp_path)
    r = client.post(f"/eval/create/{conv.id}", data={"name": "listed-case"})
    assert r.status_code == 200  # followed redirect to the case page
    page = client.get("/eval").text
    assert "listed-case" in page
    assert "unverified" in page


def test_eval_create_error_renders_html(client: TestClient, password, tmp_path):
    _login(client, password)
    sess, conv = _seed_session(tmp_path, audio=False)
    r = client.post(f"/eval/create/{conv.id}", data={})
    assert r.status_code == 400
    assert "Couldn't create eval case" in r.text


def test_eval_case_page_renders_rows(client: TestClient, password, tmp_path):
    _login(client, password)
    sess, conv = _seed_session(tmp_path)
    client.post(f"/eval/create/{conv.id}", data={"name": "render-case"})
    page = client.get("/eval/render-case").text
    assert "bonjour à tous" in page
    assert 'value="USER"' in page
    assert "/eval/render-case/audio" in page


def test_eval_case_name_traversal_404(client: TestClient, password):
    # The HTTP client normalizes "/eval/.." before routing, so exercise the
    # guard directly for the traversal shapes plus one end-to-end odd name.
    from fastapi import HTTPException

    from omilog.web.eval_ui import _case_dir_or_404

    for bad in ("..", "a/b", ".hidden", "-leading-dash", ""):
        with pytest.raises(HTTPException):
            _case_dir_or_404(bad)

    _login(client, password)
    assert client.get("/eval/.hidden").status_code == 404
    assert client.get("/eval/no-such-case").status_code == 404


def test_eval_audio_served(client: TestClient, password, tmp_path):
    _login(client, password)
    sess, conv = _seed_session(tmp_path)
    client.post(f"/eval/create/{conv.id}", data={"name": "audio-case"})
    r = client.get("/eval/audio-case/audio")
    assert r.status_code == 200
    assert r.content == b"\x00fake-opus"
    assert r.headers["content-type"].startswith("audio/ogg")


def test_eval_save_derives_both_files(client: TestClient, password, tmp_path):
    _login(client, password)
    sess, conv = _seed_session(tmp_path)
    client.post(f"/eval/create/{conv.id}", data={"name": "save-case"})

    r = client.post(
        "/eval/save-case/save",
        data={
            "start": ["2.5", "0.0", "4.2"],  # deliberately out of order
            "end": ["4.0", "2.5", "6.0"],
            "speaker": ["S1", "USER", ""],
            "text": ["salut corrigé", "bonjour corrigé", "ça va bien"],
            "verified": "true",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303

    case_dir = settings.eval_cases_dir / "save-case"
    # reference.txt chronological regardless of posted order
    assert (case_dir / "reference.txt").read_text() == (
        "bonjour corrigé\nsalut corrigé\nça va bien\n"
    )
    turns = json.loads((case_dir / "reference_turns.json").read_text())
    assert [t.get("speaker") for t in turns] == ["USER", "S1", None]
    meta = json.loads((case_dir / "case.json").read_text())
    assert meta["verified"] is True
    assert meta["source_session_id"]  # untouched by save

    case = load_case(case_dir)
    assert case.verified is True


def test_eval_save_drops_blank_rows(client: TestClient, password, tmp_path):
    _login(client, password)
    sess, conv = _seed_session(tmp_path)
    client.post(f"/eval/create/{conv.id}", data={"name": "blank-case"})
    client.post(
        "/eval/blank-case/save",
        data={
            "start": ["0.0", "3.0"],
            "end": ["2.0", "3.0"],
            "speaker": ["USER", ""],
            "text": ["du texte", "   "],
        },
    )
    turns = json.loads(
        (settings.eval_cases_dir / "blank-case" / "reference_turns.json").read_text()
    )
    assert len(turns) == 1


def test_eval_run_button_appends_history(
    client: TestClient, password, tmp_path, monkeypatch
):
    _login(client, password)
    sess, conv = _seed_session(tmp_path)
    client.post(f"/eval/create/{conv.id}", data={"name": "run-case"})

    monkeypatch.setattr(settings, "stt_base_url", "http://fake-stt")
    fake_row = {
        "verified": False,
        "error": None,
        "wer": 0.25,
        "substitutions": 2,
        "insertions": 1,
        "deletions": 0,
        "ref_words": 12,
        "hyp_words": 13,
    }
    monkeypatch.setattr(eval_runner, "eval_case", AsyncMock(return_value=fake_row))

    r = client.post("/eval/run-case/run")
    assert r.status_code == 200
    assert "WER 25.0%" in r.text
    assert "unverified" in r.text

    history = settings.eval_results_dir / "history.jsonl"
    record = json.loads(history.read_text().splitlines()[-1])
    assert record["note"] == "web:run-case"
    assert record["cases"]["run-case"]["wer"] == 0.25

    # And the index now shows the run.
    assert "25.0%" in client.get("/eval").text


def test_eval_run_button_reports_failure(
    client: TestClient, password, tmp_path, monkeypatch
):
    _login(client, password)
    sess, conv = _seed_session(tmp_path)
    client.post(f"/eval/create/{conv.id}", data={"name": "fail-case"})
    monkeypatch.setattr(settings, "stt_base_url", "http://fake-stt")
    monkeypatch.setattr(
        eval_runner, "eval_case", AsyncMock(side_effect=RuntimeError("gpu on fire"))
    )
    r = client.post("/eval/fail-case/run")
    assert r.status_code == 200
    assert "gpu on fire" in r.text
    assert not (settings.eval_results_dir / "history.jsonl").exists()


def test_eval_delete_case(client: TestClient, password, tmp_path):
    _login(client, password)
    sess, conv = _seed_session(tmp_path)
    client.post(f"/eval/create/{conv.id}", data={"name": "doomed"})
    assert (settings.eval_cases_dir / "doomed").is_dir()
    r = client.post("/eval/doomed/delete", follow_redirects=False)
    assert r.status_code == 303
    assert not (settings.eval_cases_dir / "doomed").exists()
