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
from fastapi.responses import HTMLResponse, RedirectResponse, Response
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
    return templates.TemplateResponse(
        request,
        "events.html",
        {
            "user": user,
            "upcoming": [_event_row(e, c) for e, c in upcoming],
            "past": [_event_row(e, c) for e, c in past],
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
