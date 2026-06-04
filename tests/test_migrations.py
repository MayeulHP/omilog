"""Schema-migration tests.

The migration step in db.init_db() exists to handle the case where someone
pulls a new version and their existing SQLite has older tables. SQLModel's
metadata.create_all() creates new tables but won't ALTER existing ones, so
we run an explicit additive-ADD-COLUMN pass on every startup. These tests
build a stale schema, run the migration, and confirm the new columns
appear without touching the existing rows.

Each test uses an isolated temp DB rather than the shared session DB so
we can simulate "no columns yet" cleanly.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine


def _make_legacy_conversations_table(db_path: Path) -> None:
    """Build the conversations table as it looked BEFORE the quality columns
    were added, plus one row of data. Exercises the migration's 'existing
    table + existing rows' path."""
    eng = create_engine(f"sqlite:///{db_path}")
    with eng.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                audio_session_id TEXT,
                user_id TEXT,
                title TEXT,
                summary TEXT,
                topics_json TEXT,
                extraction_repaired INTEGER NOT NULL DEFAULT 0,
                started_at TEXT,
                ended_at TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.exec_driver_sql(
            "INSERT INTO conversations (id, user_id, title, summary) "
            "VALUES ('abc-123', 'old-user', 'pre-migration', 'legacy row')"
        )
    eng.dispose()


def test_migration_adds_missing_columns_to_existing_table(tmp_path: Path):
    """Pull this on a Pi that's been running an old version. The conversations
    table has the old shape. Startup runs _apply_migrations(). After that,
    SELECTs that reference the new columns should succeed."""
    db_path = tmp_path / "legacy.db"
    _make_legacy_conversations_table(db_path)

    # Point a fresh engine at the legacy DB and rebind db.engine just for
    # this test (so the migration step runs against the right file).
    from omilog import db as db_mod

    legacy_engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    orig_engine = db_mod.engine
    db_mod.engine = legacy_engine
    try:
        db_mod._apply_migrations()
        with legacy_engine.begin() as conn:
            cols = {
                row[1]
                for row in conn.exec_driver_sql(
                    "PRAGMA table_info(conversations)"
                ).fetchall()
            }
            assert "quality_score" in cols
            assert "quality_reasoning" in cols
            assert "quality_override" in cols

            # Existing row should still be there, with the score backfilled
            # to the default (0.5).
            row = conn.exec_driver_sql(
                "SELECT id, title, quality_score, quality_reasoning, "
                "quality_override FROM conversations WHERE id='abc-123'"
            ).fetchone()
            assert row is not None
            assert row[0] == "abc-123"
            assert row[1] == "pre-migration"
            assert row[2] == 0.5
            assert row[3] is None
            assert row[4] is None
    finally:
        db_mod.engine = orig_engine
        legacy_engine.dispose()


def test_migration_is_idempotent(tmp_path: Path):
    """Re-running the migration is a no-op. Important because init_db() runs
    on every startup, not just on upgrade."""
    db_path = tmp_path / "legacy.db"
    _make_legacy_conversations_table(db_path)

    from omilog import db as db_mod

    legacy_engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    orig_engine = db_mod.engine
    db_mod.engine = legacy_engine
    try:
        db_mod._apply_migrations()
        db_mod._apply_migrations()  # second run shouldn't error
        db_mod._apply_migrations()  # neither should a third
        with legacy_engine.begin() as conn:
            # And the row is still here, unchanged.
            row = conn.exec_driver_sql(
                "SELECT title, quality_score FROM conversations "
                "WHERE id='abc-123'"
            ).fetchone()
            assert row == ("pre-migration", 0.5)
    finally:
        db_mod.engine = orig_engine
        legacy_engine.dispose()


def test_migration_on_already_current_schema_is_noop(tmp_path: Path):
    """Fresh install (or already-migrated DB): all columns present, the
    migration must not fail or duplicate anything."""
    db_path = tmp_path / "current.db"

    from omilog import db as db_mod
    from sqlmodel import SQLModel

    # Build the current schema via SQLModel against the temp file.
    fresh_engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    # Ensure model classes are imported so metadata is populated.
    from omilog import models  # noqa: F401

    SQLModel.metadata.create_all(fresh_engine)

    orig_engine = db_mod.engine
    db_mod.engine = fresh_engine
    try:
        db_mod._apply_migrations()  # should not error
        with fresh_engine.begin() as conn:
            cols = {
                row[1]
                for row in conn.exec_driver_sql(
                    "PRAGMA table_info(conversations)"
                ).fetchall()
            }
            assert "quality_score" in cols
            assert "quality_reasoning" in cols
            assert "quality_override" in cols
    finally:
        db_mod.engine = orig_engine
        fresh_engine.dispose()
