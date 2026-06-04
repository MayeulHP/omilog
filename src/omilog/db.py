import logging
from collections.abc import Iterator

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from .config import settings

logger = logging.getLogger("omilog.db")

engine = create_engine(
    f"sqlite:///{settings.db_path}",
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


# Schema migrations are intentionally not handled by Alembic — single-user
# single-host deployment, no rollback needed, just a list of additive ALTERs
# that are safe to re-run. Each entry is (table, column, "DDL fragment"). On
# startup we read the existing column list and skip anything already present.
#
# Rules: ADD COLUMN only (SQLite drops are clumsy), always with a default so
# existing rows backfill cleanly, never reorder or remove. If you ever need a
# destructive change, take a backup and write a one-shot script instead of
# hooking it in here.
_MIGRATIONS: list[tuple[str, str, str]] = [
    # Phase 5: cross-conversation speaker linking
    # (the speakers table itself is created by SQLModel.metadata.create_all)
    # Quality scoring: LLM-judged + manual override
    ("conversations", "quality_score", "REAL NOT NULL DEFAULT 0.5"),
    ("conversations", "quality_reasoning", "TEXT"),
    ("conversations", "quality_override", "REAL"),
]


def _apply_migrations() -> None:
    """Run the additive ALTER list, idempotently. Each migration is a no-op
    if its column already exists."""
    with engine.begin() as conn:
        for table, column, ddl in _MIGRATIONS:
            cols = {
                row[1]
                for row in conn.exec_driver_sql(
                    f"PRAGMA table_info({table})"
                ).fetchall()
            }
            if column in cols:
                continue
            logger.info("db: migrating %s ADD COLUMN %s", table, column)
            # ALTER TABLE … ADD COLUMN can't be parameterized in SQLite —
            # safe here because the inputs are hardcoded literals above.
            conn.exec_driver_sql(
                f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"
            )


def init_db() -> None:
    parent = settings.db_path.parent
    if str(parent) and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)
    # Import models so SQLModel.metadata sees them before create_all.
    from . import models  # noqa: F401

    SQLModel.metadata.create_all(engine)
    _apply_migrations()


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
