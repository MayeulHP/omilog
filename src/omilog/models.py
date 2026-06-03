from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


class SessionStatus(str, Enum):
    recording = "recording"
    pending_stt = "pending_stt"
    pending_llm = "pending_llm"
    done = "done"
    failed = "failed"
    silent = "silent"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AudioSession(SQLModel, table=True):
    __tablename__ = "audio_sessions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: str = Field(index=True)
    client_id: str | None = None
    device_name: str | None = None
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    duration_s: float | None = None
    codec: str | None = None
    sample_rate_hz: int | None = None
    audio_path: str | None = None
    bytes_written: int = 0
    status: SessionStatus = Field(default=SessionStatus.recording)
    error_msg: str | None = None
