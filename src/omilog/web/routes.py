"""Server-rendered web UI routes.

All HTML lives here. JSON-returning endpoints stay in `api/`. The split keeps
the two response shapes clean and lets API consumers (curl, future MCP server)
keep working untouched.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID


from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select

from ..auth import authenticate, create_access_token
from ..config import settings
from ..db import engine
from ..models import (
    ActionItem,
    ActionItemStatus,
    AudioSession,
    CalendarEvent,
    Conversation,
    PersonMention,
    SessionStatus,
    Transcript,
)
from ..pipeline import vad as vad_mod
from .auth import UIUser

router = APIRouter(tags=["web"])

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ──────────────────────────────────────────────────────────────────────────────
# Jinja filters — keep template code minimal by pre-formatting values.
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_dt(value, fmt: str = "%Y-%m-%d %H:%M") -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    return value.strftime(fmt)


def _fmt_date(value) -> str:
    return _fmt_dt(value, "%Y-%m-%d")


def _fmt_duration(seconds) -> str:
    if not seconds:
        return ""
    s = float(seconds)
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m {sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


templates.env.filters["fmt_dt"] = _fmt_dt
templates.env.filters["fmt_date"] = _fmt_date
templates.env.filters["fmt_duration"] = _fmt_duration


# ──────────────────────────────────────────────────────────────────────────────
# Login / logout
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
):
    if not authenticate(username, password):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Identifiants invalides."},
            status_code=401,
        )
    token = create_access_token(username)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        key=settings.cookie_name,
        value=token,
        max_age=settings.jwt_expire_minutes * 60,
        httponly=True,
        samesite="strict",
        secure=settings.cookie_secure,
    )
    return response


@router.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(settings.cookie_name)
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Conversations
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def index(request: Request, user: UIUser):
    with Session(engine) as db:
        convs = db.exec(
            select(Conversation)
            .where(Conversation.user_id == user)
            .order_by(Conversation.started_at.desc())
            .limit(100)
        ).all()
        # Counts shown on the list view (no joins for simplicity)
        rows = []
        for c in convs:
            n_events = db.exec(
                select(CalendarEvent).where(CalendarEvent.conversation_id == c.id)
            ).all()
            n_actions = db.exec(
                select(ActionItem)
                .where(ActionItem.conversation_id == c.id)
                .where(ActionItem.status == ActionItemStatus.open)
            ).all()
            rows.append(
                {
                    "id": str(c.id),
                    "title": c.title or "(sans titre)",
                    "summary": c.summary or "",
                    "started_at": c.started_at,
                    "topics": json.loads(c.topics_json) if c.topics_json else [],
                    "event_count": len(n_events),
                    "open_actions": len(n_actions),
                    "extraction_repaired": c.extraction_repaired,
                }
            )

        # Surface in-progress / failed sessions so they're not invisible.
        pending = db.exec(
            select(AudioSession)
            .where(AudioSession.user_id == user)
            .where(
                AudioSession.status.in_(
                    [
                        SessionStatus.pending_vad,
                        SessionStatus.pending_stt,
                        SessionStatus.pending_llm,
                        SessionStatus.failed,
                    ]
                )
            )
            .order_by(AudioSession.started_at.desc())
            .limit(10)
        ).all()

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "user": user,
            "conversations": rows,
            "pending": pending,
        },
    )


@router.get("/conversations/{conv_id}", response_class=HTMLResponse)
async def conversation_detail(request: Request, user: UIUser, conv_id: UUID):
    with Session(engine) as db:
        conv = db.get(Conversation, conv_id)
        if conv is None or conv.user_id != user:
            raise HTTPException(404, "Not found")
        transcript = db.exec(
            select(Transcript)
            .where(Transcript.audio_session_id == conv.audio_session_id)
            .order_by(Transcript.created_at.desc())
            .limit(1)
        ).first()
        events = db.exec(
            select(CalendarEvent).where(CalendarEvent.conversation_id == conv.id)
        ).all()
        actions = db.exec(
            select(ActionItem).where(ActionItem.conversation_id == conv.id)
        ).all()
        people = db.exec(
            select(PersonMention).where(PersonMention.conversation_id == conv.id)
        ).all()
        audio_session = db.get(AudioSession, conv.audio_session_id)

    return templates.TemplateResponse(
        request,
        "conversation.html",
        {
            "user": user,
            "conv": conv,
            "transcript": transcript,
            "transcript_segments": json.loads(transcript.segments_json)
            if transcript and transcript.segments_json
            else [],
            "events": events,
            "actions": actions,
            "people": people,
            "audio_session": audio_session,
            "topics": json.loads(conv.topics_json) if conv.topics_json else [],
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Events + action items
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/events", response_class=HTMLResponse)
async def events_page(request: Request, user: UIUser):
    now = datetime.now(timezone.utc)
    with Session(engine) as db:
        upcoming = db.exec(
            select(CalendarEvent, Conversation)
            .join(Conversation, CalendarEvent.conversation_id == Conversation.id)
            .where(Conversation.user_id == user)
            .where(CalendarEvent.starts_at >= now)
            .order_by(CalendarEvent.starts_at.asc())
            .limit(100)
        ).all()
        past = db.exec(
            select(CalendarEvent, Conversation)
            .join(Conversation, CalendarEvent.conversation_id == Conversation.id)
            .where(Conversation.user_id == user)
            .where(CalendarEvent.starts_at < now)
            .order_by(CalendarEvent.starts_at.desc())
            .limit(20)
        ).all()

    # Build the feed URL only when a token is set — guards against showing a
    # broken-by-default link.
    feed_url = None
    if settings.ics_feed_token:
        base = str(request.base_url).rstrip("/")
        feed_url = f"{base}/calendar.ics?token={settings.ics_feed_token}"

    return templates.TemplateResponse(
        request,
        "events.html",
        {
            "user": user,
            "upcoming": [_event_row(e, c) for e, c in upcoming],
            "past": [_event_row(e, c) for e, c in past],
            "feed_url": feed_url,
        },
    )


def _event_row(e: CalendarEvent, c: Conversation) -> dict[str, Any]:
    return {
        "id": str(e.id),
        "conversation_id": str(c.id),
        "conversation_title": c.title or "(sans titre)",
        "title": e.title,
        "starts_at": e.starts_at,
        "ends_at": e.ends_at,
        "location": e.location,
        "attendees": json.loads(e.attendees_json) if e.attendees_json else [],
        "confidence": e.confidence,
        "exported_to_ics": e.exported_to_ics,
    }


@router.get("/actions", response_class=HTMLResponse)
async def actions_page(
    request: Request,
    user: UIUser,
    status: str = "open",
):
    try:
        status_filter = ActionItemStatus(status)
    except ValueError:
        status_filter = ActionItemStatus.open
    with Session(engine) as db:
        rows = db.exec(
            select(ActionItem, Conversation)
            .join(Conversation, ActionItem.conversation_id == Conversation.id)
            .where(Conversation.user_id == user)
            .where(ActionItem.status == status_filter)
            .order_by(ActionItem.due_at.asc().nullslast())
            .limit(200)
        ).all()
    return templates.TemplateResponse(
        request,
        "actions.html",
        {
            "user": user,
            "items": [_action_row(a, c) for a, c in rows],
            "status_filter": status_filter.value,
        },
    )


def _action_row(a: ActionItem, c: Conversation) -> dict[str, Any]:
    return {
        "id": str(a.id),
        "conversation_id": str(c.id),
        "conversation_title": c.title or "(sans titre)",
        "text": a.text,
        "owner": a.owner,
        "due_at": a.due_at,
        "status": a.status.value,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Audio streaming
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/sessions/{session_id}/audio")
async def session_audio(user: UIUser, session_id: UUID):
    """Stream a session's audio file to the browser for the <audio> tag.

    Cookie-authenticated. Path-traversal safe: rejects audio_path values that
    aren't inside storage_dir, just in case anyone ever sets one by hand.
    """
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None or sess.user_id != user:
            raise HTTPException(404, "session not found")
        if not sess.audio_path:
            raise HTTPException(404, "session has no audio file")
        codec = sess.codec or "opus"
    path = Path(sess.audio_path).resolve()
    storage_root = settings.storage_dir.resolve()
    try:
        path.relative_to(storage_root)
    except ValueError:
        raise HTTPException(403, "audio path outside storage_dir") from None
    if not path.exists():
        raise HTTPException(404, "audio file no longer on disk")

    # .opus files are Ogg-Opus → audio/ogg renders in all modern browsers'
    # <audio> element. WAV-uploaded files get audio/wav.
    media_type = {
        "opus": "audio/ogg",
        "ogg": "audio/ogg",
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
        "flac": "audio/flac",
        "webm": "audio/webm",
    }.get(codec, "application/octet-stream")
    return FileResponse(path, media_type=media_type)


# ──────────────────────────────────────────────────────────────────────────────
# VAD tuning page
# ──────────────────────────────────────────────────────────────────────────────


_TUNABLE_VAD_KEYS = (
    "OMILOG_VAD_THRESHOLD_DB",
    "OMILOG_VAD_MIN_SILENCE_SECONDS",
    "OMILOG_VAD_GAP_SECONDS",
    "OMILOG_VAD_PAD_SECONDS",
)


@router.get("/tune", response_class=HTMLResponse)
async def tune_index(request: Request, user: UIUser):
    """List sessions whose audio files still exist on disk — tunable targets.

    Left-joins Conversation so the row can show a real title rather than just a
    UUID; transcript presence is shown as a small marker.
    """
    with Session(engine) as db:
        sessions = db.exec(
            select(AudioSession)
            .where(AudioSession.user_id == user)
            .where(AudioSession.audio_path != None)  # noqa: E711 — SQLAlchemy
            .order_by(AudioSession.started_at.desc())
            .limit(50)
        ).all()
        eligible: list[dict[str, Any]] = []
        for s in sessions:
            if not s.audio_path or not Path(s.audio_path).exists():
                continue
            conv = db.exec(
                select(Conversation).where(Conversation.audio_session_id == s.id)
            ).first()
            has_transcript = (
                db.exec(
                    select(Transcript)
                    .where(Transcript.audio_session_id == s.id)
                    .limit(1)
                ).first()
                is not None
            )
            eligible.append(
                {
                    "session": s,
                    "conversation": conv,
                    "has_transcript": has_transcript,
                }
            )
    return templates.TemplateResponse(
        request,
        "tune_index.html",
        {
            "user": user,
            "rows": eligible,
            "current": _current_vad_defaults(),
        },
    )


@router.get("/tune/{session_id}", response_class=HTMLResponse)
async def tune_session(request: Request, user: UIUser, session_id: UUID):
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None or sess.user_id != user:
            raise HTTPException(404, "session not found")
        if not sess.audio_path or not Path(sess.audio_path).exists():
            raise HTTPException(404, "audio file missing on disk")
        conv = db.exec(
            select(Conversation).where(Conversation.audio_session_id == sess.id)
        ).first()
        transcript = db.exec(
            select(Transcript)
            .where(Transcript.audio_session_id == sess.id)
            .order_by(Transcript.created_at.desc())
            .limit(1)
        ).first()
        transcript_segments = (
            json.loads(transcript.segments_json)
            if (transcript and transcript.segments_json)
            else []
        )
    return templates.TemplateResponse(
        request,
        "tune_session.html",
        {
            "user": user,
            "session": sess,
            "conversation": conv,
            "transcript": transcript,
            "transcript_segments": transcript_segments,
            "current": _current_vad_defaults(),
        },
    )


@router.post("/tune/{session_id}/analyze", response_class=HTMLResponse)
async def tune_analyze(
    request: Request,
    user: UIUser,
    session_id: UUID,
    threshold_db: Annotated[float, Form()],
    min_silence_s: Annotated[float, Form()],
    gap_seconds: Annotated[float, Form()],
    pad_seconds: Annotated[float, Form()],
):
    """HTMX endpoint: re-run VAD on this session with given params, return a
    fragment with the timeline + segmentation result. No DB writes."""
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None or sess.user_id != user:
            raise HTTPException(404, "session not found")
        if not sess.audio_path:
            raise HTTPException(404, "session has no audio_path")
        audio_path = Path(sess.audio_path)
    if not audio_path.exists():
        raise HTTPException(404, "audio file missing")

    try:
        duration_s, silences = await vad_mod.analyse(
            audio_path,
            threshold_db=threshold_db,
            min_silence_s=min_silence_s,
        )
    except vad_mod.VADError as e:
        return templates.TemplateResponse(
            request,
            "_tune_results.html",
            {"error": str(e)},
            status_code=400,
        )

    convs = vad_mod.segment_by_silence_gaps(
        duration_s,
        silences,
        gap_threshold_s=gap_seconds,
        pad_s=pad_seconds,
    )
    speech_total = sum(end - start for start, end in convs)
    speech_pct = (speech_total / duration_s * 100) if duration_s > 0 else 0
    silence_pct = 100 - speech_pct

    return templates.TemplateResponse(
        request,
        "_tune_results.html",
        {
            "duration_s": duration_s,
            "silences": silences,
            "conversations": convs,
            "speech_total": speech_total,
            "speech_pct": speech_pct,
            "silence_pct": silence_pct,
            "gap_seconds": gap_seconds,
            "params": {
                "threshold_db": threshold_db,
                "min_silence_s": min_silence_s,
                "gap_seconds": gap_seconds,
                "pad_seconds": pad_seconds,
            },
        },
    )


@router.post("/tune/apply-defaults", response_class=HTMLResponse)
async def tune_apply_defaults(
    user: UIUser,
    threshold_db: Annotated[float, Form()],
    min_silence_s: Annotated[float, Form()],
    gap_seconds: Annotated[float, Form()],
    pad_seconds: Annotated[float, Form()],
):
    """Write the tuned values to .env, preserving comments and other settings.

    Only the four VAD keys are touched — everything else in the file is left
    untouched. Restart required to pick up changes (.env is read once).
    """
    env_path = Path(".env")
    if not env_path.exists():
        raise HTTPException(400, ".env file not found on server")

    new_values = {
        "OMILOG_VAD_THRESHOLD_DB": str(threshold_db),
        "OMILOG_VAD_MIN_SILENCE_SECONDS": str(min_silence_s),
        "OMILOG_VAD_GAP_SECONDS": str(gap_seconds),
        "OMILOG_VAD_PAD_SECONDS": str(pad_seconds),
    }

    lines = env_path.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    updated: list[str] = []
    for line in lines:
        matched = False
        for key, val in new_values.items():
            # Match `KEY=` at the start; preserve indented or commented-out
            # variants by leaving them alone (only replace the actual setter).
            if line.startswith(f"{key}="):
                updated.append(f"{key}={val}")
                seen.add(key)
                matched = True
                break
        if not matched:
            updated.append(line)
    # Append any keys that weren't already present.
    for key, val in new_values.items():
        if key not in seen:
            updated.append(f"{key}={val}")

    env_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return HTMLResponse(
        '<small style="color: var(--pico-color-green-500)">'
        "✓ Saved to <code>.env</code>. Restart the server to apply."
        "</small>"
    )


def _current_vad_defaults() -> dict[str, Any]:
    return {
        "threshold_db": settings.vad_threshold_db,
        "min_silence_s": settings.vad_min_silence_seconds,
        "gap_seconds": settings.vad_gap_seconds,
        "pad_seconds": settings.vad_pad_seconds,
    }


@router.post("/sessions/{session_id}/dismiss")
async def dismiss_session(user: UIUser, session_id: UUID):
    """Delete an AudioSession row + its audio file. Used by the pending panel
    to clear failed or stuck rows from the index view.

    Returns an empty 200 so HTMX's outerHTML swap removes the row from the DOM.
    """
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None or sess.user_id != user:
            raise HTTPException(404, "session not found")
        # Best-effort file cleanup — don't block deletion if the file is gone.
        if sess.audio_path:
            try:
                Path(sess.audio_path).unlink(missing_ok=True)
            except OSError:
                pass
        db.delete(sess)
        db.commit()
    return Response(status_code=200, content="")


@router.post("/actions/{action_id}/status", response_class=HTMLResponse)
async def action_set_status(
    request: Request,
    user: UIUser,
    action_id: UUID,
    status: Annotated[str, Form()],
):
    try:
        target = ActionItemStatus(status)
    except ValueError:
        raise HTTPException(400, "invalid status")
    with Session(engine) as db:
        action = db.get(ActionItem, action_id)
        if action is None:
            raise HTTPException(404, "action not found")
        conv = db.get(Conversation, action.conversation_id)
        if conv is None or conv.user_id != user:
            raise HTTPException(404, "action not found")
        action.status = target
        db.add(action)
        db.commit()
        # Refresh for the partial render.
        db.refresh(action)
        row = _action_row(action, conv)
    return templates.TemplateResponse(
        request,
        "_action_row.html",
        {"item": row},
    )
