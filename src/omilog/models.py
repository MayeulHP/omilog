from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


class SessionStatus(str, Enum):
    recording = "recording"
    pending_vad = "pending_vad"      # raw parent capture, awaiting segmentation
    pending_stt = "pending_stt"
    pending_llm = "pending_llm"
    done = "done"
    failed = "failed"
    silent = "silent"
    segmented = "segmented"           # parent capture, children spawned, file deleted


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AudioSession(SQLModel, table=True):
    __tablename__ = "audio_sessions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: str = Field(index=True)
    client_id: str | None = None
    device_name: str | None = None
    # Self-FK: child sessions (one per VAD-detected conversation) point back to
    # the long parent capture they were carved from. None = top-level.
    parent_id: UUID | None = Field(
        default=None, foreign_key="audio_sessions.id", index=True
    )
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    duration_s: float | None = None
    codec: str | None = None
    sample_rate_hz: int | None = None
    audio_path: str | None = None
    bytes_written: int = 0
    status: SessionStatus = Field(default=SessionStatus.recording, index=True)
    error_msg: str | None = None


class Transcript(SQLModel, table=True):
    __tablename__ = "transcripts"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    audio_session_id: UUID = Field(foreign_key="audio_sessions.id", index=True)
    text: str
    # Whisper "verbose_json" segments verbatim, stored as JSON text so we can
    # replay timing/confidence later without re-running STT.
    segments_json: str | None = None
    language: str | None = None
    model: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    audio_session_id: UUID = Field(foreign_key="audio_sessions.id", index=True)
    user_id: str = Field(index=True)
    title: str | None = None
    summary: str | None = None
    # JSON list of topic strings, raw LLM output. Stored as text for portability.
    topics_json: str | None = None
    # True when extract.parse had to recover the JSON via json_repair —
    # usually means we hit max_tokens and the extraction is partial.
    extraction_repaired: bool = False
    started_at: datetime
    ended_at: datetime | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class CalendarEvent(SQLModel, table=True):
    __tablename__ = "calendar_events"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversations.id", index=True)
    title: str
    description: str | None = None
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    location: str | None = None
    attendees_json: str | None = None  # JSON list of strings
    confidence: float = 0.5
    exported_to_ics: bool = False
    created_at: datetime = Field(default_factory=_utcnow)


class ActionItemStatus(str, Enum):
    open = "open"
    done = "done"
    dismissed = "dismissed"


class ActionItem(SQLModel, table=True):
    __tablename__ = "action_items"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversations.id", index=True)
    text: str
    due_at: datetime | None = None
    owner: str | None = None
    status: ActionItemStatus = Field(default=ActionItemStatus.open, index=True)
    created_at: datetime = Field(default_factory=_utcnow)


class PersonMention(SQLModel, table=True):
    __tablename__ = "people_mentions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    conversation_id: UUID = Field(foreign_key="conversations.id", index=True)
    name: str = Field(index=True)
    context: str | None = None
    mentioned_at: datetime = Field(default_factory=_utcnow)


class WakeAction(SQLModel, table=True):
    __tablename__ = "wake_actions"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: str = Field(index=True)
    name: str
    # JSON list of phrases like ["Hey Jarvis", "Jarvis", "Salut Jarvis"].
    phrases_json: str
    # Optional JSON list of stop phrases. When present, $transcript is cut at
    # the earliest occurrence of any of these phrases after the wake match —
    # works like radio "over" so a long monologue after the request doesn't
    # all end up in the command's argument.
    stop_phrases_json: str | None = None
    # Shell command template, substitutes $transcript / $transcript_full /
    # $conversation_id / $wake_phrase via shlex.quote-safe replacement.
    command: str
    enabled: bool = True
    timeout_seconds: float = 30.0
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class WakeInvocation(SQLModel, table=True):
    __tablename__ = "wake_invocations"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    wake_action_id: UUID = Field(foreign_key="wake_actions.id", index=True)
    # Null when fired via the UI "test" button instead of a real conversation.
    conversation_id: UUID | None = Field(
        default=None, foreign_key="conversations.id", index=True
    )
    matched_phrase: str
    input_text: str
    command_resolved: str
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    duration_ms: int | None = None
    created_at: datetime = Field(default_factory=_utcnow)
