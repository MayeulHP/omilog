"""Audio retention rotation + archive flag tests.

Two layers:
- Pure pipeline tests for ``_rotate_old_audio``: which rows get picked up,
  which get spared (archived, recent, wrong status), and that the DB row
  survives with audio_path cleared.
- UI tests for the archive/unpin toggle endpoint + the conversation page's
  banner states (pinned indicator vs. "audio deleted by retention").
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session

from omilog.config import settings
from omilog.db import engine
from omilog.models import AudioSession, Conversation, SessionStatus
from omilog.pipeline import runner


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _login(client: TestClient, password: str) -> None:
    client.post("/login", data={"username": "test", "password": password})


def _seed_session(
    *,
    tmp_storage: Path,
    started_at: datetime,
    status: SessionStatus = SessionStatus.done,
    archived: bool = False,
    user: str = "test",
    write_file: bool = True,
) -> tuple[UUID, Path]:
    """Insert an AudioSession with an actual .opus file on disk so the
    rotation can really unlink it. Storage dir is parameterized so tests
    don't collide on the shared session storage."""
    sid = uuid4()
    audio_path = tmp_storage / f"{sid}.opus"
    if write_file:
        audio_path.write_bytes(b"ogg-opus pretend bytes")
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id=user,
                audio_path=str(audio_path),
                codec="opus",
                started_at=started_at,
                status=status,
                archived=archived,
            )
        )
        db.commit()
    return sid, audio_path


def _seed_conversation_for_session(audio_session_id: UUID, user: str = "test") -> UUID:
    cid = uuid4()
    with Session(engine) as db:
        db.add(
            Conversation(
                id=cid,
                audio_session_id=audio_session_id,
                user_id=user,
                title="t",
                started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            )
        )
        db.commit()
    return cid


# ──────────────────────────────────────────────────────────────────────────────
# _rotate_old_audio: who gets deleted, who gets spared
# ──────────────────────────────────────────────────────────────────────────────


def test_rotation_disabled_by_default(tmp_path, monkeypatch):
    """audio_retention_days=0 is the default and means rotation is OFF."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "audio_retention_days", 0)
    # Ancient session — would be rotated if rotation were on.
    sid, fp = _seed_session(
        tmp_storage=tmp_path,
        started_at=datetime.now(timezone.utc) - timedelta(days=365),
    )
    assert runner._rotate_old_audio() == 0
    assert fp.exists()  # file untouched


def test_rotation_deletes_old_done_session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "audio_retention_days", 30)
    sid, fp = _seed_session(
        tmp_storage=tmp_path,
        started_at=datetime.now(timezone.utc) - timedelta(days=60),
    )
    assert runner._rotate_old_audio() == 1
    assert not fp.exists()
    # DB row still there, audio_path cleared so UI hides the player.
    with Session(engine) as db:
        sess = db.get(AudioSession, sid)
    assert sess is not None
    assert sess.audio_path is None


def test_rotation_skips_recent_session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "audio_retention_days", 30)
    sid, fp = _seed_session(
        tmp_storage=tmp_path,
        started_at=datetime.now(timezone.utc) - timedelta(days=10),  # well within
    )
    assert runner._rotate_old_audio() == 0
    assert fp.exists()


def test_rotation_skips_archived(tmp_path, monkeypatch):
    """The whole point of the archive flag — old + archived → keep."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "audio_retention_days", 30)
    sid, fp = _seed_session(
        tmp_storage=tmp_path,
        started_at=datetime.now(timezone.utc) - timedelta(days=365),
        archived=True,
    )
    assert runner._rotate_old_audio() == 0
    assert fp.exists()


def test_rotation_skips_in_flight_sessions(tmp_path, monkeypatch):
    """A session that's still being processed (pending_stt, pending_llm,
    failed, etc.) must NEVER have its audio yanked — the pipeline still
    needs the file. Even if it's old by the date filter."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "audio_retention_days", 30)
    old = datetime.now(timezone.utc) - timedelta(days=365)
    sid_stt, fp_stt = _seed_session(
        tmp_storage=tmp_path, started_at=old, status=SessionStatus.pending_stt
    )
    sid_llm, fp_llm = _seed_session(
        tmp_storage=tmp_path, started_at=old, status=SessionStatus.pending_llm
    )
    sid_failed, fp_failed = _seed_session(
        tmp_storage=tmp_path, started_at=old, status=SessionStatus.failed
    )
    sid_done, fp_done = _seed_session(
        tmp_storage=tmp_path, started_at=old, status=SessionStatus.done
    )
    assert runner._rotate_old_audio() == 1  # only done gets rotated
    assert fp_stt.exists()
    assert fp_llm.exists()
    assert fp_failed.exists()
    assert not fp_done.exists()


def test_rotation_handles_missing_file_gracefully(tmp_path, monkeypatch):
    """If audio_path is set but the file is already gone (manual unlink,
    crashed mid-write, NAS hiccup), don't crash; clear the path anyway so
    we don't keep retrying."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "audio_retention_days", 30)
    sid, fp = _seed_session(
        tmp_storage=tmp_path,
        started_at=datetime.now(timezone.utc) - timedelta(days=60),
        write_file=False,  # row points at a path that doesn't exist
    )
    # missing_ok=True on unlink means this still counts as "deleted".
    assert runner._rotate_old_audio() == 1
    with Session(engine) as db:
        sess = db.get(AudioSession, sid)
    assert sess.audio_path is None


def test_rotation_rejects_paths_outside_storage_dir(tmp_path, monkeypatch):
    """Defensive: if someone hand-edited an audio_path to point outside
    storage_dir, rotation must NOT delete it. Path-traversal guard."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "audio_retention_days", 30)
    # File somewhere outside the storage root.
    outside = tmp_path.parent / "evil.opus"
    outside.write_bytes(b"important user data")
    sid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="test",
                audio_path=str(outside),
                codec="opus",
                started_at=datetime.now(timezone.utc) - timedelta(days=60),
                status=SessionStatus.done,
            )
        )
        db.commit()
    runner._rotate_old_audio()
    # File survived — the rejection happened BEFORE the unlink.
    assert outside.exists()
    outside.unlink()  # cleanup


def test_rotation_processes_segmented_parents(tmp_path, monkeypatch):
    """Segmented parents technically already lose their audio via VAD's
    cleanup, but cover the status filter so future refactors don't leave
    them stranded as a no-rotation gap."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "audio_retention_days", 30)
    sid, fp = _seed_session(
        tmp_storage=tmp_path,
        started_at=datetime.now(timezone.utc) - timedelta(days=60),
        status=SessionStatus.segmented,
    )
    assert runner._rotate_old_audio() == 1
    assert not fp.exists()


# ──────────────────────────────────────────────────────────────────────────────
# _maybe_rotate_audio: hourly throttle
# ──────────────────────────────────────────────────────────────────────────────


def test_maybe_rotate_runs_on_first_call(tmp_path, monkeypatch):
    """Cold-start should run once immediately so old data isn't waiting an
    hour to be cleaned up post-restart."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "audio_retention_days", 30)
    monkeypatch.setattr(runner, "_LAST_ROTATION_AT", None)

    called = {"count": 0}

    def fake_rotate():
        called["count"] += 1
        return 0

    monkeypatch.setattr(runner, "_rotate_old_audio", fake_rotate)
    runner._maybe_rotate_audio()
    assert called["count"] == 1


def test_maybe_rotate_throttles_within_interval(tmp_path, monkeypatch):
    """Second call right after the first must NOT trigger again."""
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "audio_retention_days", 30)
    monkeypatch.setattr(runner, "_LAST_ROTATION_AT", None)
    called = {"count": 0}
    monkeypatch.setattr(runner, "_rotate_old_audio", lambda: called.update({"count": called["count"] + 1}) or 0)
    runner._maybe_rotate_audio()
    runner._maybe_rotate_audio()  # immediate second call
    assert called["count"] == 1


def test_maybe_rotate_does_nothing_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "audio_retention_days", 0)
    monkeypatch.setattr(runner, "_LAST_ROTATION_AT", None)
    called = {"count": 0}
    monkeypatch.setattr(runner, "_rotate_old_audio", lambda: called.update({"count": called["count"] + 1}) or 0)
    runner._maybe_rotate_audio()
    assert called["count"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# UI: archive toggle endpoint + conversation page rendering
# ──────────────────────────────────────────────────────────────────────────────


def test_archive_endpoint_flips_flag(
    client: TestClient, password: str, tmp_path
):
    _login(client, password)
    sid, _ = _seed_session(
        tmp_storage=tmp_path,
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    cid = _seed_conversation_for_session(sid)
    r = client.post(f"/conversations/{cid}/archive", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == f"/conversations/{cid}"
    with Session(engine) as db:
        sess = db.get(AudioSession, sid)
    assert sess.archived is True
    # Toggle again — should flip back.
    client.post(f"/conversations/{cid}/archive")
    with Session(engine) as db:
        sess = db.get(AudioSession, sid)
    assert sess.archived is False


def test_archive_endpoint_404_for_wrong_user(
    client: TestClient, password: str, tmp_path
):
    _login(client, password)
    sid, _ = _seed_session(
        tmp_storage=tmp_path,
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        user="someone-else",
    )
    cid = _seed_conversation_for_session(sid, user="someone-else")
    r = client.post(f"/conversations/{cid}/archive")
    assert r.status_code == 404


def test_conversation_page_shows_pin_button_when_audio_present(
    client: TestClient, password: str, tmp_path
):
    _login(client, password)
    sid, _ = _seed_session(
        tmp_storage=tmp_path,
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    cid = _seed_conversation_for_session(sid)
    r = client.get(f"/conversations/{cid}")
    assert "Pin audio" in r.text


def test_conversation_page_shows_unpin_when_archived(
    client: TestClient, password: str, tmp_path
):
    _login(client, password)
    sid, _ = _seed_session(
        tmp_storage=tmp_path,
        started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        archived=True,
    )
    cid = _seed_conversation_for_session(sid)
    r = client.get(f"/conversations/{cid}")
    assert "Unpin audio" in r.text
    # Also the pinned banner near the audio element.
    assert "Pinned" in r.text


def test_conversation_page_shows_audio_removed_notice_when_rotated(
    client: TestClient, password: str
):
    """After rotation, audio_path is None but the AudioSession row stays.
    The page should surface that rather than silently dropping the player
    block (user would think nothing was ever recorded)."""
    _login(client, password)
    sid = uuid4()
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=sid,
                user_id="test",
                audio_path=None,  # rotated
                codec="opus",
                started_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
                status=SessionStatus.done,
            )
        )
        db.commit()
    cid = _seed_conversation_for_session(sid)
    r = client.get(f"/conversations/{cid}")
    assert "removed by retention rotation" in r.text
    # And no pin button — nothing to pin anymore.
    assert "Pin audio" not in r.text
