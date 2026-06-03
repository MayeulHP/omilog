"""One-shot migration to add the Phase 3b `parent_id` column.

SQLModel's `create_all` only creates *missing* tables — it never alters
existing ones. So an existing omilog.db from Phase 2 needs this ALTER once.
No-op when the column already exists, so it's safe to re-run.

Usage:
    .venv/bin/python scripts/migrate_phase3b.py
"""

import sqlite3
import sys

from omilog.config import settings


def main() -> int:
    db_path = str(settings.db_path)
    print(f"migrating {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # Inspect current columns.
        cols = [row[1] for row in cur.execute("PRAGMA table_info(audio_sessions)")]
        if "audio_sessions" not in [
            row[0] for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]:
            print(
                "audio_sessions table doesn't exist yet — run the app once first to "
                "let SQLModel create it, then re-run this script.",
                file=sys.stderr,
            )
            return 1

        if "parent_id" in cols:
            print("parent_id column already present, nothing to do.")
            return 0

        # SQLite is dynamically typed, so we just add the column; SQLAlchemy
        # will round-trip UUIDs as 32-char hex strings into it (same as the id
        # column's storage).
        cur.execute("ALTER TABLE audio_sessions ADD COLUMN parent_id CHAR(32)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS ix_audio_sessions_parent_id "
            "ON audio_sessions(parent_id)"
        )
        conn.commit()
        print("✓ added parent_id column + index")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
