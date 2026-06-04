"""VAD tuning page tests.

Mocks vad.analyse so tests don't actually shell out to ffmpeg.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session

from omilog.db import engine
from omilog.models import (
    AudioSession,
    Conversation,
    SessionStatus,
    Transcript,
)
from omilog.pipeline import vad as vad_mod


def _seed_session_with_file(
    *, user: str = "test", duration: float = 100.0, tmp_path: Path | None = None
) -> tuple[UUID, Path]:
    """Create a session row plus a real-but-fake audio file on disk."""
    sid = uuid4()
    storage_dir = (
        tmp_path or Path(os.environ.get("OMILOG_STORAGE_DIR", "storage"))
    )
    storage_dir.mkdir(parents=True, exist_ok=True)
    audio_path = storage_dir / f"{sid}.opus"
    audio_path.write_bytes(b"fake-opus" * 64)
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id=user,
                audio_path=str(audio_path),
                codec="opus",
                duration_s=duration,
                started_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
                status=SessionStatus.failed,
                error_msg="for testing",
            )
        )
        db.commit()
    return sid, audio_path


def _login(client: TestClient, password: str) -> None:
    client.post("/login", data={"username": "test", "password": password})


# ──────────────────────────────────────────────────────────────────────────────
# /tune index
# ──────────────────────────────────────────────────────────────────────────────

def test_tune_index_lists_session_with_file(client: TestClient, password: str):
    sid, _ = _seed_session_with_file()
    _login(client, password)
    r = client.get("/tune")
    assert r.status_code == 200
    assert str(sid) in r.text


def test_tune_index_skips_session_without_file(client: TestClient, password: str):
    sid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="test",
                audio_path="/tmp/nonexistent.opus",
                codec="opus",
                started_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
                status=SessionStatus.failed,
            )
        )
        db.commit()
    _login(client, password)
    r = client.get("/tune")
    assert r.status_code == 200
    assert str(sid) not in r.text


def test_tune_index_requires_auth(client: TestClient):
    r = client.get("/tune", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/login"


# ──────────────────────────────────────────────────────────────────────────────
# /tune/{id}
# ──────────────────────────────────────────────────────────────────────────────

def test_tune_session_page_renders(client: TestClient, password: str):
    sid, _ = _seed_session_with_file()
    _login(client, password)
    r = client.get(f"/tune/{sid}")
    assert r.status_code == 200
    assert "threshold_db" in r.text
    assert "gap_seconds" in r.text


def test_tune_session_404_for_other_user(client: TestClient, password: str):
    sid, _ = _seed_session_with_file(user="not-me")
    _login(client, password)
    r = client.get(f"/tune/{sid}")
    assert r.status_code == 404


def test_tune_session_404_when_file_missing(client: TestClient, password: str):
    sid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="test",
                audio_path="/tmp/missing.opus",
                codec="opus",
                started_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
                status=SessionStatus.failed,
            )
        )
        db.commit()
    _login(client, password)
    r = client.get(f"/tune/{sid}")
    assert r.status_code == 404


# ──────────────────────────────────────────────────────────────────────────────
# POST /tune/{id}/analyze — HTMX endpoint
# ──────────────────────────────────────────────────────────────────────────────

def test_tune_analyze_returns_results_fragment(client: TestClient, password: str):
    sid, _ = _seed_session_with_file()
    _login(client, password)

    fake_silences = [(0.0, 1.0), (50.0, 130.0)]  # one long enough to split
    with patch.object(
        vad_mod, "analyse", new=AsyncMock(return_value=(200.0, fake_silences))
    ):
        r = client.post(
            f"/tune/{sid}/analyze",
            data={
                "threshold_db": "-40",
                "min_silence_s": "0.5",
                "gap_seconds": "60",
                "pad_seconds": "0.4",
            },
        )
    assert r.status_code == 200
    assert "Resulting conversations" in r.text or "Silence regions" in r.text
    # Two conversations (split by the 80s middle silence + leading trim)
    assert "200.0" in r.text  # duration shown


def test_tune_analyze_surfaces_vad_error(client: TestClient, password: str):
    sid, _ = _seed_session_with_file()
    _login(client, password)
    with patch.object(
        vad_mod,
        "analyse",
        new=AsyncMock(side_effect=vad_mod.VADError("ffmpeg exit=183")),
    ):
        r = client.post(
            f"/tune/{sid}/analyze",
            data={
                "threshold_db": "-40",
                "min_silence_s": "0.5",
                "gap_seconds": "60",
                "pad_seconds": "0.4",
            },
        )
    assert r.status_code == 400
    assert "VAD analysis failed" in r.text


def test_tune_analyze_404_other_user(client: TestClient, password: str):
    sid, _ = _seed_session_with_file(user="not-me")
    _login(client, password)
    r = client.post(
        f"/tune/{sid}/analyze",
        data={
            "threshold_db": "-40",
            "min_silence_s": "0.5",
            "gap_seconds": "60",
            "pad_seconds": "0.4",
        },
    )
    assert r.status_code == 404


# ──────────────────────────────────────────────────────────────────────────────
# POST /tune/apply-defaults — writes to .env
# ──────────────────────────────────────────────────────────────────────────────

def test_tune_apply_defaults_updates_env_in_place(
    client: TestClient, password: str, tmp_path, monkeypatch
):
    # Sandbox the .env target to tmp_path/.env so the test never touches the
    # repo's real .env file. Routes resolve `.env` relative to cwd; chdir.
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# omilog config\n"
        "OMILOG_VAD_THRESHOLD_DB=-30\n"
        "OMILOG_USERNAME=test\n"
        "OMILOG_VAD_GAP_SECONDS=60\n"
        "# comment after\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    _login(client, password)
    r = client.post(
        "/tune/apply-defaults",
        data={
            "threshold_db": "-45",
            "min_silence_s": "0.7",
            "gap_seconds": "90",
            "pad_seconds": "0.5",
        },
    )
    assert r.status_code == 200
    assert "Saved" in r.text

    out = env_file.read_text(encoding="utf-8").splitlines()
    # Original comments preserved
    assert "# omilog config" in out
    assert "# comment after" in out
    # Unrelated keys preserved
    assert "OMILOG_USERNAME=test" in out
    # In-place updates for tuned keys
    assert "OMILOG_VAD_THRESHOLD_DB=-45.0" in out
    assert "OMILOG_VAD_GAP_SECONDS=90.0" in out
    # New keys appended
    assert any(line.startswith("OMILOG_VAD_MIN_SILENCE_SECONDS=0.7") for line in out)
    assert any(line.startswith("OMILOG_VAD_PAD_SECONDS=0.5") for line in out)


def test_tune_apply_defaults_404_without_env(
    client: TestClient, password: str, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # tmp_path has no .env
    _login(client, password)
    r = client.post(
        "/tune/apply-defaults",
        data={
            "threshold_db": "-40",
            "min_silence_s": "0.5",
            "gap_seconds": "60",
            "pad_seconds": "0.4",
        },
    )
    assert r.status_code == 400


def test_tune_apply_defaults_requires_auth(client: TestClient):
    r = client.post(
        "/tune/apply-defaults",
        data={
            "threshold_db": "-40",
            "min_silence_s": "0.5",
            "gap_seconds": "60",
            "pad_seconds": "0.4",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303  # redirect to /login


# ──────────────────────────────────────────────────────────────────────────────
# Audio streaming + tune page enrichment
# ──────────────────────────────────────────────────────────────────────────────

def test_session_audio_streams_file(client: TestClient, password: str):
    sid, audio_path = _seed_session_with_file()
    _login(client, password)
    r = client.get(f"/sessions/{sid}/audio")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/ogg"
    assert r.content == audio_path.read_bytes()


def test_session_audio_404_when_file_missing(client: TestClient, password: str):
    """File path lives inside storage_dir but the file got deleted — distinct
    from the path-traversal case (which is tested separately and 403s)."""
    from uuid import uuid4

    sid = uuid4()
    storage_dir = Path(os.environ["OMILOG_STORAGE_DIR"])
    deleted_path = storage_dir / f"{sid}.opus"  # never created on disk
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="test",
                audio_path=str(deleted_path),
                codec="opus",
                started_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
                status=SessionStatus.failed,
            )
        )
        db.commit()
    _login(client, password)
    assert client.get(f"/sessions/{sid}/audio").status_code == 404


def test_session_audio_404_other_user(client: TestClient, password: str):
    sid, _ = _seed_session_with_file(user="not-me")
    _login(client, password)
    assert client.get(f"/sessions/{sid}/audio").status_code == 404


def test_session_audio_requires_auth(client: TestClient):
    sid, _ = _seed_session_with_file()
    r = client.get(f"/sessions/{sid}/audio", follow_redirects=False)
    assert r.status_code == 303


def test_session_audio_rejects_path_outside_storage(
    client: TestClient, password: str, tmp_path
):
    """If audio_path points outside storage_dir (e.g. hand-edited DB), the
    endpoint must refuse — defense in depth against a stored-path-traversal."""
    from uuid import uuid4

    outside = tmp_path / "outside.opus"
    outside.write_bytes(b"bad")
    sid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="test",
                audio_path=str(outside),  # not under OMILOG_STORAGE_DIR
                codec="opus",
                started_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
                status=SessionStatus.failed,
            )
        )
        db.commit()
    _login(client, password)
    assert client.get(f"/sessions/{sid}/audio").status_code == 403


def test_tune_index_includes_conversation_title(client: TestClient, password: str):
    from uuid import uuid4
    sid, _ = _seed_session_with_file()
    cid = uuid4()
    with Session(engine) as db:
        db.add(
            Conversation(
                id=cid,
                audio_session_id=sid,
                user_id="test",
                title="Une conversation testable",
                summary="résumé",
                started_at=datetime(2026, 6, 3, tzinfo=timezone.utc),
            )
        )
        db.commit()
    _login(client, password)
    r = client.get("/tune")
    assert r.status_code == 200
    assert "Une conversation testable" in r.text


def test_tune_session_shows_audio_player_and_transcript(
    client: TestClient, password: str
):
    import json as _json

    sid, _ = _seed_session_with_file()
    with Session(engine) as db:
        db.add(
            Transcript(
                audio_session_id=sid,
                text="Bonjour ceci est un transcript de test.",
                segments_json=_json.dumps(
                    [{"start": 0.0, "text": "Bonjour ceci est un transcript de test."}]
                ),
                language="fr",
                model="whisper-large-v3-turbo",
            )
        )
        db.commit()
    _login(client, password)
    r = client.get(f"/tune/{sid}")
    assert r.status_code == 200
    assert "<audio" in r.text
    assert f"/sessions/{sid}/audio" in r.text
    assert "Bonjour ceci est un transcript" in r.text
