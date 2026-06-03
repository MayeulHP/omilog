"""Adds the conversations.extraction_repaired flag introduced in the
json_repair fallback for partial LLM extractions.

SQLModel.create_all only creates missing tables, never alters existing ones,
so an existing omilog.db needs this ALTER once. Safe to re-run.

Usage:
    .venv/bin/python scripts/migrate_extraction_repaired.py
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
        tables = [
            row[0]
            for row in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        if "conversations" not in tables:
            print(
                "conversations table doesn't exist yet — run the app once first.",
                file=sys.stderr,
            )
            return 1
        cols = [
            row[1]
            for row in cur.execute("PRAGMA table_info(conversations)")
        ]
        if "extraction_repaired" in cols:
            print("extraction_repaired already present, nothing to do.")
            return 0
        cur.execute(
            "ALTER TABLE conversations "
            "ADD COLUMN extraction_repaired INTEGER NOT NULL DEFAULT 0"
        )
        conn.commit()
        print("✓ added extraction_repaired column")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
