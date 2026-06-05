"""Server-rendered web UI routes.

All HTML lives here. JSON-returning endpoints stay in `api/`. The split keeps
the two response shapes clean and lets API consumers (curl, future MCP server)
keep working untouched.
"""

import asyncio
import json
import shutil
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from typing import Annotated, Any
from uuid import UUID
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import func


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
    Decision,
    PersonMention,
    SessionStatus,
    Speaker,
    Transcript,
    WakeAction,
    WakeInvocation,
)
from ..pipeline import daily as daily_mod
from ..pipeline import diarize as diarize_mod
from ..pipeline import vad as vad_mod
from ..pipeline import wake as wake_mod
from ..pipeline.audio import TranscodeError, transcode_to_wav_bytes
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
            {"error": "Invalid credentials."},
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

# Default "show normal+up": hides scores below this threshold. Users can
# bump this to substantive-only or drop it to "show all" via the dropdown.
_QUALITY_FILTER_THRESHOLDS = {
    "all": 0.0,
    "normal": 0.3,
    "substantive": 0.7,
}


def _effective_quality(c: Conversation) -> float:
    """User override wins over LLM score. Defined here so the list filter
    and the detail-page badge agree on the same number."""
    return c.quality_override if c.quality_override is not None else c.quality_score


def _quality_bucket(q: float) -> str:
    """Three-bucket classifier used everywhere a badge or icon is rendered.
    Matches the LLM prompt's anchors: noise/normal/substantive."""
    if q < 0.3:
        return "noise"
    if q < 0.7:
        return "normal"
    return "substantive"


@router.get("/", response_class=HTMLResponse)
async def index(
    request: Request, user: UIUser, q: str = "normal", show_hidden: int = 0
):
    """Conversation list. `q` selects the quality threshold filter; the
    "show_hidden" toggle ignores it entirely (so you can find a conversation
    that was auto-buried as noise)."""
    threshold = _QUALITY_FILTER_THRESHOLDS.get(q, 0.3)
    if show_hidden:
        threshold = 0.0

    with Session(engine) as db:
        convs = db.exec(
            select(Conversation)
            .where(Conversation.user_id == user)
            .order_by(Conversation.started_at.desc())
            .limit(200)  # we'll filter down further in Python
        ).all()
        # Apply the effective-quality filter (override OR llm-score). Doing
        # this in Python rather than SQL because override-or-fallback isn't
        # a clean SQLite expression and the row count is bounded anyway.
        filtered = [c for c in convs if _effective_quality(c) >= threshold]
        hidden_count = len(convs) - len(filtered)

        rows = []
        for c in filtered[:100]:  # cap output regardless of filter
            n_events = db.exec(
                select(CalendarEvent).where(CalendarEvent.conversation_id == c.id)
            ).all()
            n_actions = db.exec(
                select(ActionItem)
                .where(ActionItem.conversation_id == c.id)
                .where(ActionItem.status == ActionItemStatus.open)
            ).all()
            eq = _effective_quality(c)
            rows.append(
                {
                    "id": str(c.id),
                    "title": c.title or "(untitled)",
                    "summary": c.summary or "",
                    "started_at": c.started_at,
                    "topics": json.loads(c.topics_json) if c.topics_json else [],
                    "event_count": len(n_events),
                    "open_actions": len(n_actions),
                    "extraction_repaired": c.extraction_repaired,
                    "quality": eq,
                    "quality_bucket": _quality_bucket(eq),
                    "quality_reasoning": c.quality_reasoning,
                    "quality_overridden": c.quality_override is not None,
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
            "quality_filter": q if q in _QUALITY_FILTER_THRESHOLDS else "normal",
            "hidden_count": hidden_count,
            "show_hidden": bool(show_hidden),
        },
    )


@router.post("/conversations/{conv_id}/rate")
async def conversation_rate(
    user: UIUser,
    conv_id: UUID,
    rating: Annotated[str, Form()],
):
    """Set the user's quality override.

    ``rating`` is one of:
    - ``noise``        → override = 0.0 (definitely hide)
    - ``substantive``  → override = 1.0 (definitely show)
    - ``clear``        → override = None (trust the LLM's score)
    """
    override: float | None
    if rating == "noise":
        override = 0.0
    elif rating == "substantive":
        override = 1.0
    elif rating == "clear":
        override = None
    else:
        raise HTTPException(400, "rating must be noise / substantive / clear")

    with Session(engine) as db:
        conv = db.get(Conversation, conv_id)
        if conv is None or conv.user_id != user:
            raise HTTPException(404, "conversation not found")
        conv.quality_override = override
        conv.updated_at = datetime.now(timezone.utc)
        db.add(conv)
        db.commit()
    return Response(headers={"HX-Refresh": "true"})


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
        decisions = db.exec(
            select(Decision).where(Decision.conversation_id == conv.id)
        ).all()
        people = db.exec(
            select(PersonMention).where(PersonMention.conversation_id == conv.id)
        ).all()
        audio_session = db.get(AudioSession, conv.audio_session_id)
        # Wake actions that fired on this conversation, with their action name.
        wake_rows = list(
            db.exec(
                select(WakeInvocation, WakeAction)
                .join(WakeAction, WakeInvocation.wake_action_id == WakeAction.id)
                .where(WakeInvocation.conversation_id == conv.id)
                .order_by(WakeInvocation.created_at.asc())
            ).all()
        )

        # Resolve cross-conversation Speaker rows for the segments that
        # carry a speaker_id. Segments from pre-Phase-5 transcripts just
        # have a `speaker` label and no speaker_id; those get a None entry.
        transcript_segments = (
            json.loads(transcript.segments_json)
            if transcript and transcript.segments_json
            else []
        )
        speaker_ids: set[UUID] = set()
        for s in transcript_segments:
            raw = s.get("speaker_id")
            if not raw:
                continue
            try:
                speaker_ids.add(UUID(raw) if isinstance(raw, str) else raw)
            except (ValueError, TypeError):
                # Malformed speaker_id in segments_json — skip silently rather
                # than 500 on the conversation page. The segment will fall back
                # to its per-conversation label.
                continue
        speakers_by_id: dict[str, Speaker] = {}
        if speaker_ids:
            rows = db.exec(
                select(Speaker).where(Speaker.id.in_(speaker_ids))
            ).all()
            speakers_by_id = {str(sp.id): sp for sp in rows}

        # Order labels by first appearance in the transcript so USER comes
        # first when it actually speaks first, otherwise we get an arbitrary
        # dict-iteration order that flips around between renders.
        speakers_in_conv: dict[str, dict[str, Any]] = {}
        for seg in transcript_segments:
            label = seg.get("speaker")
            if not label or label in speakers_in_conv:
                continue
            sid = seg.get("speaker_id")
            speakers_in_conv[label] = {
                "label": label,
                "speaker": speakers_by_id.get(sid) if sid else None,
            }

    eq = _effective_quality(conv)
    # Compute linked-speaker count here rather than in the template —
    # Jinja's `selectattr` on dicts has version-dependent behavior with
    # missing/None attrs that's brittle. Python's plain comprehension is
    # unambiguous: count speakers_in_conv entries whose 'speaker' is set
    # (i.e. actually linked to a Speaker row, not just a transcript label).
    linked_speaker_count = sum(
        1 for info in speakers_in_conv.values() if info.get("speaker")
    )
    return templates.TemplateResponse(
        request,
        "conversation.html",
        {
            "user": user,
            "conv": conv,
            "transcript": transcript,
            "transcript_segments": transcript_segments,
            "events": events,
            "actions": actions,
            "decisions": decisions,
            "people": people,
            "audio_session": audio_session,
            "topics": json.loads(conv.topics_json) if conv.topics_json else [],
            "wake_invocations": [
                {"inv": inv, "action": action} for inv, action in wake_rows
            ],
            "speakers_in_conv": speakers_in_conv,
            "speakers_by_id": speakers_by_id,
            "linked_speaker_count": linked_speaker_count,
            "quality": eq,
            "quality_bucket": _quality_bucket(eq),
            "quality_reasoning": conv.quality_reasoning,
            "quality_overridden": conv.quality_override is not None,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Speakers (Phase 5: cross-conversation linking)
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/speakers", response_class=HTMLResponse)
async def speakers_index(request: Request, user: UIUser):
    """List every known voice. Most-mentioned first — the user's own voice
    tends to dominate that ranking, followed by frequent contacts."""
    with Session(engine) as db:
        rows = list(
            db.exec(
                select(Speaker)
                .where(Speaker.user_id == user)
                .order_by(Speaker.mention_count.desc(), Speaker.updated_at.desc())
            ).all()
        )
    return templates.TemplateResponse(
        request,
        "speakers.html",
        {"user": user, "speakers": rows},
    )


@router.post("/speakers/{speaker_id}/rename")
async def speaker_rename(
    user: UIUser,
    speaker_id: UUID,
    name: Annotated[str, Form()] = "",
):
    """Set or clear the human name for a Speaker row.

    Returns HX-Refresh so HTMX reloads the page that triggered the rename —
    works for both the conversation page (transcript needs to re-render with
    the new name in front of every segment) and the /speakers list page.
    """
    with Session(engine) as db:
        sp = db.get(Speaker, speaker_id)
        if sp is None or sp.user_id != user:
            raise HTTPException(404, "speaker not found")
        sp.name = name.strip() or None
        sp.updated_at = datetime.now(timezone.utc)
        db.add(sp)
        db.commit()
    return Response(headers={"HX-Refresh": "true"})


@router.post("/speakers/{speaker_id}/toggle-user")
async def speaker_toggle_user(user: UIUser, speaker_id: UUID):
    """Flip the is_user flag. Used when the longest-talker heuristic
    misidentified another speaker as the wearer (e.g. you were quiet during
    a meeting), or to demote a false positive."""
    with Session(engine) as db:
        sp = db.get(Speaker, speaker_id)
        if sp is None or sp.user_id != user:
            raise HTTPException(404, "speaker not found")
        sp.is_user = not sp.is_user
        sp.updated_at = datetime.now(timezone.utc)
        db.add(sp)
        db.commit()
    return Response(headers={"HX-Refresh": "true"})


@router.post("/speakers/{speaker_id}/delete")
async def speaker_delete(user: UIUser, speaker_id: UUID):
    """Remove a Speaker row. Segments that reference it lose their link —
    they'll fall back to displaying their per-conversation label. Future
    conversations with this voice produce a fresh Speaker row (since the
    embedding is gone)."""
    with Session(engine) as db:
        sp = db.get(Speaker, speaker_id)
        if sp is None or sp.user_id != user:
            raise HTTPException(404, "speaker not found")
        db.delete(sp)
        db.commit()
    return Response(headers={"HX-Refresh": "true"})


@router.post("/speakers/merge")
async def speakers_merge(
    user: UIUser,
    speaker_ids: Annotated[list[str], Form()],
    return_to: Annotated[str, Form()] = "/speakers",
):
    """Combine N >= 2 Speaker rows into one, surviving as the primary.

    Mechanics:
    - Primary = the named-and-most-mentioned candidate (ties broken by
      oldest). User can rename / re-prioritise afterwards if the default
      wasn't right.
    - Embedding is weighted-averaged by mention_count so the merged
      centroid reflects all the audio that fed each constituent voice.
    - mention_count sums; is_user is OR'd; an empty primary name inherits
      the first non-empty secondary name.
    - Every transcript segment whose speaker_id points at a secondary
      gets rewritten to the primary's id, so the conversation pages
      stop showing the merged voices as separate labels.
    - Secondary Speaker rows are deleted at the end.

    This is what the user actually wants when they go around naming
    multiple rows 'Marie' — without merge, those stay as separate voices
    that just happen to share a display name. After merge, they're one.
    """
    if len(speaker_ids) < 2:
        raise HTTPException(400, "select at least 2 speakers to merge")

    ids: list[UUID] = []
    for raw in speaker_ids:
        try:
            ids.append(UUID(raw))
        except ValueError:
            raise HTTPException(400, f"invalid speaker id: {raw!r}")
    if len(set(ids)) != len(ids):
        raise HTTPException(400, "duplicate speaker ids in selection")

    # Lazy-import the embedding helpers from the runner module — they
    # encapsulate the float32-bytes serialisation that Speaker.embedding
    # uses, and re-defining them here would just be a maintenance hazard.
    from ..pipeline import runner as runner_mod

    with Session(engine) as db:
        rows = list(
            db.exec(
                select(Speaker)
                .where(Speaker.user_id == user)
                .where(Speaker.id.in_(ids))
            ).all()
        )
        if len(rows) != len(ids):
            raise HTTPException(
                404, "one or more speakers not found or not yours"
            )

        # Primary selection: prefer named over unnamed, then by mention
        # count (higher wins), then by created_at (older wins — stable when
        # everything else is tied). Sort descending on the composite key.
        def _primary_rank(r: Speaker) -> tuple[int, int, float]:
            created = r.created_at.timestamp() if r.created_at else 0.0
            return (1 if r.name else 0, r.mention_count, -created)

        rows.sort(key=_primary_rank, reverse=True)
        primary = rows[0]
        secondaries = rows[1:]

        # Weighted-average embedding. mention_count = 0 falls back to 1
        # so a fresh-import row doesn't get zero weight and disappear.
        total_weight = 0.0
        accumulator: list[float] | None = None
        for r in rows:
            emb = runner_mod._emb_from_bytes(r.embedding)
            if accumulator is None:
                accumulator = [0.0] * len(emb)
            w = float(r.mention_count) if r.mention_count > 0 else 1.0
            for i, v in enumerate(emb):
                accumulator[i] += v * w
            total_weight += w
        if accumulator is None or total_weight == 0:
            raise HTTPException(500, "merge: nothing to average (impossible)")
        merged_emb = [x / total_weight for x in accumulator]

        primary.embedding = runner_mod._emb_to_bytes(merged_emb)
        primary.mention_count = sum(max(1, r.mention_count) for r in rows)
        primary.is_user = any(r.is_user for r in rows)
        if not primary.name:
            for r in secondaries:
                if r.name:
                    primary.name = r.name
                    break
        primary.updated_at = datetime.now(timezone.utc)
        db.add(primary)

        # Rewrite transcript segments to point at the primary. Done in a
        # single pass over all of THIS user's transcripts (joined via the
        # AudioSession FK). Limiting to this user's data keeps a single
        # corrupted segment_json in someone else's row from blowing up.
        secondary_id_strs = {str(r.id) for r in secondaries}
        primary_str = str(primary.id)
        transcripts = list(
            db.exec(
                select(Transcript)
                .join(AudioSession, Transcript.audio_session_id == AudioSession.id)
                .where(AudioSession.user_id == user)
            ).all()
        )
        for t in transcripts:
            if not t.segments_json:
                continue
            try:
                segments = json.loads(t.segments_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(segments, list):
                continue
            changed = False
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                sid = seg.get("speaker_id")
                if sid and sid in secondary_id_strs:
                    seg["speaker_id"] = primary_str
                    changed = True
            if changed:
                t.segments_json = json.dumps(segments)
                db.add(t)

        for r in secondaries:
            db.delete(r)
        db.commit()

    # Path-only redirect; refuse anything that isn't a same-origin path
    # so a hostile form value can't bounce the user off-site.
    if not return_to.startswith("/") or return_to.startswith("//"):
        return_to = "/speakers"
    return RedirectResponse(return_to, status_code=303)


# ──────────────────────────────────────────────────────────────────────────────
# Daily summary
# ──────────────────────────────────────────────────────────────────────────────


def _parse_iso_date(s: str) -> date_cls | None:
    """``YYYY-MM-DD`` → datetime.date, else None. Used to validate path
    params without leaking the underlying ValueError up to the 500 handler."""
    try:
        return date_cls.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _today_local() -> date_cls:
    try:
        tz = ZoneInfo(settings.local_timezone)
    except Exception:
        tz = ZoneInfo("UTC")
    return datetime.now(tz).date()


@router.get("/daily", response_class=HTMLResponse)
async def daily_today(request: Request, user: UIUser):
    """Redirect /daily → /daily/<today's date in local timezone>. Means the
    nav link can be a stable, dumb URL and we don't bake "today" into the
    cached HTML."""
    return RedirectResponse(f"/daily/{_today_local().isoformat()}", status_code=303)


@router.get("/daily/{date_str}", response_class=HTMLResponse)
async def daily_show(
    request: Request,
    user: UIUser,
    date_str: str,
):
    d = _parse_iso_date(date_str)
    if d is None:
        raise HTTPException(404, "invalid date — expected YYYY-MM-DD")

    cached = daily_mod.get_cached(user, d)
    threshold = (
        cached.quality_threshold
        if cached is not None
        else settings.daily_summary_threshold
    )

    # We pull the conversations that COULD feed a fresh summary at the
    # current threshold so the user sees what would be included if they
    # regenerate, plus we can render them as clickable rows below the
    # narrative. Doing this here (rather than in the template) keeps Jinja
    # free of DB calls.
    eligible = daily_mod._fetch_eligible(user, d, threshold)

    # If we have a cached summary but the eligible-now list differs from
    # what got cached, flag a "regenerate suggested" hint.
    cached_ids: set[str] = set()
    if cached is not None:
        cached_ids = {str(cid) for cid in daily_mod.conversation_ids_for(cached)}
    current_ids = {str(c.id) for c in eligible}
    drift = bool(cached_ids and cached_ids != current_ids)

    prev_day = d - timedelta(days=1)
    next_day = d + timedelta(days=1)
    today = _today_local()

    return templates.TemplateResponse(
        request,
        "daily.html",
        {
            "user": user,
            "date": d,
            "date_str": d.isoformat(),
            "summary": cached,
            "narrative": cached.narrative if cached and cached.narrative else None,
            "eligible": eligible,
            "threshold": threshold,
            "drift": drift,
            "prev_day_url": f"/daily/{prev_day.isoformat()}",
            "next_day_url": f"/daily/{next_day.isoformat()}",
            "prev_day_label": prev_day.isoformat(),
            "next_day_label": next_day.isoformat(),
            "today": today,
            "is_today": d == today,
            "llm_configured": bool(settings.llm_base_url),
        },
    )


@router.post("/daily/{date_str}/generate")
async def daily_generate(
    user: UIUser,
    date_str: str,
    threshold: Annotated[float | None, Form()] = None,
):
    """Fire the LLM call and cache the result. Threshold defaults to the
    system-wide setting; UI may override per-request via form field so the
    user can preview a stricter or looser day-summary without changing
    .env."""
    d = _parse_iso_date(date_str)
    if d is None:
        raise HTTPException(404, "invalid date — expected YYYY-MM-DD")

    cutoff = (
        threshold if threshold is not None else settings.daily_summary_threshold
    )
    cutoff = max(0.0, min(1.0, cutoff))

    try:
        result = await daily_mod.generate(user, d, quality_threshold=cutoff)
    except Exception as e:  # noqa: BLE001 — surface anything; details in toast
        raise HTTPException(500, f"daily summary failed: {e}") from e

    daily_mod.store(user, d, result)
    return RedirectResponse(f"/daily/{date_str}", status_code=303)


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
        "conversation_title": c.title or "(untitled)",
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
        "conversation_title": c.title or "(untitled)",
        "text": a.text,
        "owner": a.owner,
        "due_at": a.due_at,
        "status": a.status.value,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Wake-word actions
# ──────────────────────────────────────────────────────────────────────────────


def _parse_phrases(raw: str) -> list[str]:
    """Accept newline- or comma-separated phrase list, strip empties."""
    out: list[str] = []
    for chunk in raw.replace("\r\n", "\n").split("\n"):
        for sub in chunk.split(","):
            phrase = sub.strip()
            if phrase:
                out.append(phrase)
    return out


@router.get("/wake-actions", response_class=HTMLResponse)
async def wake_actions_index(request: Request, user: UIUser):
    with Session(engine) as db:
        actions = list(
            db.exec(
                select(WakeAction)
                .where(WakeAction.user_id == user)
                .order_by(WakeAction.created_at.desc())
            ).all()
        )
        rows = []
        for a in actions:
            try:
                phrases = json.loads(a.phrases_json)
            except json.JSONDecodeError:
                phrases = []
            stop_phrases = []
            if a.stop_phrases_json:
                try:
                    stop_phrases = json.loads(a.stop_phrases_json) or []
                except json.JSONDecodeError:
                    stop_phrases = []
            recent_invocations = list(
                db.exec(
                    select(WakeInvocation)
                    .where(WakeInvocation.wake_action_id == a.id)
                    .order_by(WakeInvocation.created_at.desc())
                    .limit(3)
                ).all()
            )
            rows.append(
                {
                    "action": a,
                    "phrases": phrases,
                    "stop_phrases": stop_phrases,
                    "recent": recent_invocations,
                }
            )
    return templates.TemplateResponse(
        request,
        "wake_actions_index.html",
        {"user": user, "rows": rows},
    )


@router.get("/wake-actions/new", response_class=HTMLResponse)
async def wake_action_new_form(request: Request, user: UIUser):
    return templates.TemplateResponse(
        request,
        "wake_actions_edit.html",
        {
            "user": user,
            "action": None,
            "phrases_text": "",
            "stop_phrases_text": "",
            "form_error": None,
        },
    )


@router.post("/wake-actions/new")
async def wake_action_create(
    user: UIUser,
    name: Annotated[str, Form()],
    phrases: Annotated[str, Form()],
    command: Annotated[str, Form()],
    timeout_seconds: Annotated[float, Form()] = 30.0,
    enabled: Annotated[str, Form()] = "",
    stop_phrases: Annotated[str, Form()] = "",
):
    phrases_list = _parse_phrases(phrases)
    stop_phrases_list = _parse_phrases(stop_phrases)
    if not name.strip() or not command.strip() or not phrases_list:
        raise HTTPException(400, "name, phrases, and command are all required")
    with Session(engine) as db:
        db.add(
            WakeAction(
                user_id=user,
                name=name.strip(),
                phrases_json=json.dumps(phrases_list),
                stop_phrases_json=(
                    json.dumps(stop_phrases_list) if stop_phrases_list else None
                ),
                command=command,
                enabled=bool(enabled),
                timeout_seconds=max(1.0, min(timeout_seconds, 300.0)),
            )
        )
        db.commit()
    return RedirectResponse("/wake-actions", status_code=303)


@router.get("/wake-actions/{action_id}/edit", response_class=HTMLResponse)
async def wake_action_edit_form(
    request: Request, user: UIUser, action_id: UUID
):
    with Session(engine) as db:
        action = db.get(WakeAction, action_id)
        if action is None or action.user_id != user:
            raise HTTPException(404, "action not found")
        try:
            phrases = json.loads(action.phrases_json)
        except json.JSONDecodeError:
            phrases = []
        stop_phrases = []
        if action.stop_phrases_json:
            try:
                stop_phrases = json.loads(action.stop_phrases_json) or []
            except json.JSONDecodeError:
                stop_phrases = []
    return templates.TemplateResponse(
        request,
        "wake_actions_edit.html",
        {
            "user": user,
            "action": action,
            "phrases_text": "\n".join(phrases),
            "stop_phrases_text": "\n".join(stop_phrases),
            "form_error": None,
        },
    )


@router.post("/wake-actions/{action_id}/edit")
async def wake_action_update(
    user: UIUser,
    action_id: UUID,
    name: Annotated[str, Form()],
    phrases: Annotated[str, Form()],
    command: Annotated[str, Form()],
    timeout_seconds: Annotated[float, Form()] = 30.0,
    enabled: Annotated[str, Form()] = "",
    stop_phrases: Annotated[str, Form()] = "",
):
    phrases_list = _parse_phrases(phrases)
    stop_phrases_list = _parse_phrases(stop_phrases)
    if not name.strip() or not command.strip() or not phrases_list:
        raise HTTPException(400, "name, phrases, and command are all required")
    with Session(engine) as db:
        action = db.get(WakeAction, action_id)
        if action is None or action.user_id != user:
            raise HTTPException(404, "action not found")
        action.name = name.strip()
        action.phrases_json = json.dumps(phrases_list)
        action.stop_phrases_json = (
            json.dumps(stop_phrases_list) if stop_phrases_list else None
        )
        action.command = command
        action.enabled = bool(enabled)
        action.timeout_seconds = max(1.0, min(timeout_seconds, 300.0))
        action.updated_at = datetime.now(timezone.utc)
        db.add(action)
        db.commit()
    return RedirectResponse("/wake-actions", status_code=303)


@router.post("/wake-actions/{action_id}/delete")
async def wake_action_delete(user: UIUser, action_id: UUID):
    with Session(engine) as db:
        action = db.get(WakeAction, action_id)
        if action is None or action.user_id != user:
            raise HTTPException(404, "action not found")
        # Delete dependent invocation rows first (no cascade on SQLite by default).
        invocations = list(
            db.exec(
                select(WakeInvocation).where(WakeInvocation.wake_action_id == action.id)
            ).all()
        )
        for inv in invocations:
            db.delete(inv)
        db.delete(action)
        db.commit()
    return RedirectResponse("/wake-actions", status_code=303)


@router.post("/wake-actions/{action_id}/test", response_class=HTMLResponse)
async def wake_action_test(
    request: Request,
    user: UIUser,
    action_id: UUID,
    test_input: Annotated[str, Form()] = "",
):
    """Manually fire an action with arbitrary input, for debugging from the UI.

    Stores a WakeInvocation with conversation_id=NULL so it shows up in the
    action's history without being attached to a real conversation.
    """
    with Session(engine) as db:
        action = db.get(WakeAction, action_id)
        if action is None or action.user_id != user:
            raise HTTPException(404, "action not found")
        action_name = action.name
        timeout = action.timeout_seconds
        template = action.command

    variables = {
        "transcript": test_input,
        "transcript_full": test_input,
        "conversation_id": "(test)",
        "wake_phrase": "(test)",
    }
    resolved = wake_mod.resolve_command(template, variables)
    result = await wake_mod.execute_command(resolved, timeout_s=timeout)

    with Session(engine) as db:
        inv = WakeInvocation(
            wake_action_id=action_id,
            conversation_id=None,
            matched_phrase="(test)",
            input_text=test_input,
            command_resolved=resolved,
            exit_code=result["exit_code"],
            stdout=result["stdout"],
            stderr=result["stderr"],
            duration_ms=result["duration_ms"],
        )
        db.add(inv)
        db.commit()
        db.refresh(inv)

    return templates.TemplateResponse(
        request,
        "_wake_invocation.html",
        {"inv": inv, "action_name": action_name},
    )


@router.get("/wake-actions/{action_id}/log", response_class=HTMLResponse)
async def wake_action_log(request: Request, user: UIUser, action_id: UUID):
    with Session(engine) as db:
        action = db.get(WakeAction, action_id)
        if action is None or action.user_id != user:
            raise HTTPException(404, "action not found")
        invocations = list(
            db.exec(
                select(WakeInvocation)
                .where(WakeInvocation.wake_action_id == action_id)
                .order_by(WakeInvocation.created_at.desc())
                .limit(200)
            ).all()
        )
        try:
            phrases = json.loads(action.phrases_json)
        except json.JSONDecodeError:
            phrases = []
    return templates.TemplateResponse(
        request,
        "wake_actions_log.html",
        {
            "user": user,
            "action": action,
            "phrases": phrases,
            "invocations": invocations,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# /status dashboard
# ──────────────────────────────────────────────────────────────────────────────


async def _check_url(url: str, *, timeout_s: float = 3.0) -> dict[str, Any]:
    """Light-touch reachability ping. We don't care about the status code —
    any response means the backend is alive enough to answer. Connection
    errors and timeouts both surface as not-ok."""
    if not url:
        return {"configured": False}
    started = monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(url)
        return {
            "configured": True,
            "ok": True,
            "status": r.status_code,
            "ms": int((monotonic() - started) * 1000),
        }
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        return {
            "configured": True,
            "ok": False,
            "error": type(e).__name__,
            "ms": int((monotonic() - started) * 1000),
        }
    except Exception as e:  # noqa: BLE001 — surface anything else as "down"
        return {
            "configured": True,
            "ok": False,
            "error": str(e)[:100],
            "ms": int((monotonic() - started) * 1000),
        }


def _walk_storage_size(root: Path) -> tuple[int, int]:
    """Total bytes + file count under `root`. Defensive against weird names."""
    total = 0
    count = 0
    if not root.exists():
        return 0, 0
    for f in root.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
                count += 1
        except OSError:
            pass
    return total, count


def _humanize_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if isinstance(n, float) else f"{n} {unit}"
        n = n / 1024
    return f"{n:.1f} PB"


@router.get("/status", response_class=HTMLResponse)
async def status_page(request: Request, user: UIUser):
    # Run backend checks in parallel so a slow one doesn't block the other.
    stt_url = settings.stt_base_url
    llm_url = settings.llm_base_url + "/models" if settings.llm_base_url else ""
    stt_check, llm_check = await asyncio.gather(
        _check_url(stt_url),
        _check_url(llm_url),
    )

    with Session(engine) as db:
        def count_where(*conds) -> int:
            stmt = select(func.count(AudioSession.id))
            for c in conds:
                stmt = stmt.where(c)
            return db.exec(stmt).first() or 0

        pending_vad = count_where(AudioSession.status == SessionStatus.pending_vad)
        pending_stt = count_where(AudioSession.status == SessionStatus.pending_stt)
        pending_llm = count_where(AudioSession.status == SessionStatus.pending_llm)
        failed_total = count_where(AudioSession.status == SessionStatus.failed)

        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        recent_failures = list(
            db.exec(
                select(AudioSession)
                .where(AudioSession.status == SessionStatus.failed)
                .where(AudioSession.started_at >= yesterday)
                .order_by(AudioSession.started_at.desc())
                .limit(10)
            ).all()
        )

        last_conv = db.exec(
            select(Conversation)
            .where(Conversation.user_id == user)
            .order_by(Conversation.created_at.desc())
            .limit(1)
        ).first()

        recent_24h_conv_count = db.exec(
            select(func.count(Conversation.id))
            .where(Conversation.user_id == user)
            .where(Conversation.created_at >= yesterday)
        ).first() or 0

    audio_bytes, audio_files = _walk_storage_size(settings.storage_dir)
    db_bytes = settings.db_path.stat().st_size if settings.db_path.exists() else 0
    try:
        disk = shutil.disk_usage(settings.storage_dir)
    except OSError:
        disk = None

    return templates.TemplateResponse(
        request,
        "status.html",
        {
            "user": user,
            "stt": {"url": stt_url, **stt_check},
            "llm": {"url": settings.llm_base_url, **llm_check},
            "diarization_enabled": settings.diarization_enabled,
            "diarization_models_present": all(
                p.exists()
                for p in (
                    settings.diarization_segmentation_model,
                    settings.diarization_embedding_model,
                )
            ),
            "pipeline": {
                "pending_vad": pending_vad,
                "pending_stt": pending_stt,
                "pending_llm": pending_llm,
                "failed_total": failed_total,
            },
            "storage": {
                "audio_bytes": audio_bytes,
                "audio_files": audio_files,
                "audio_human": _humanize_bytes(audio_bytes),
                "db_bytes": db_bytes,
                "db_human": _humanize_bytes(db_bytes),
                "disk_total": _humanize_bytes(disk.total) if disk else "?",
                "disk_free": _humanize_bytes(disk.free) if disk else "?",
                "disk_used_pct": (
                    int(100 * disk.used / disk.total) if disk and disk.total else 0
                ),
            },
            "activity": {
                "last_conv": last_conv,
                "recent_24h_count": recent_24h_conv_count,
            },
            "recent_failures": recent_failures,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Audio archive (exempt from retention rotation)
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/conversations/{conv_id}/archive")
async def conversation_archive(user: UIUser, conv_id: UUID):
    """Toggle the ``archived`` flag on this conversation's AudioSession.

    Archived sessions are exempt from the periodic audio-retention sweep
    no matter how old — useful for conversations you want to keep the
    audio for indefinitely (memorable moments, evidence, family history).
    The transcript/extraction is always kept regardless of this flag;
    the only thing it protects is the .opus file on disk.
    """
    with Session(engine) as db:
        conv = db.get(Conversation, conv_id)
        if conv is None or conv.user_id != user:
            raise HTTPException(404, "conversation not found")
        sess = db.get(AudioSession, conv.audio_session_id)
        if sess is None:
            raise HTTPException(404, "audio session not found")
        sess.archived = not sess.archived
        db.add(sess)
        db.commit()
    return RedirectResponse(f"/conversations/{conv_id}", status_code=303)


# ──────────────────────────────────────────────────────────────────────────────
# Conversation deletion
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/conversations/{conv_id}/delete")
async def conversation_delete(user: UIUser, conv_id: UUID):
    """Delete a conversation, all its dependent rows, AND the source audio
    file. Destructive; the UI confirms before firing.

    What gets removed:
      - PersonMention, ActionItem, CalendarEvent (children of conversation)
      - WakeInvocation rows referencing this conversation
      - Conversation row
      - Transcript rows tied to the source AudioSession
      - The AudioSession row itself
      - The .opus file on disk

    Leaves alone:
      - Any other conversations sharing a parent AudioSession (children of a
        VAD-segmented parent each have their own AudioSession, so this case
        only matters if you somehow had a single AudioSession with multiple
        conversations — which we never produce).
    """
    with Session(engine) as db:
        conv = db.get(Conversation, conv_id)
        if conv is None or conv.user_id != user:
            raise HTTPException(404, "conversation not found")
        audio_session_id = conv.audio_session_id

        # Children first (FK).
        for inv in db.exec(
            select(WakeInvocation).where(WakeInvocation.conversation_id == conv.id)
        ).all():
            db.delete(inv)
        for mention in db.exec(
            select(PersonMention).where(PersonMention.conversation_id == conv.id)
        ).all():
            db.delete(mention)
        for item in db.exec(
            select(ActionItem).where(ActionItem.conversation_id == conv.id)
        ).all():
            db.delete(item)
        for event in db.exec(
            select(CalendarEvent).where(CalendarEvent.conversation_id == conv.id)
        ).all():
            db.delete(event)
        db.delete(conv)

        # Transcript + source audio.
        for t in db.exec(
            select(Transcript).where(Transcript.audio_session_id == audio_session_id)
        ).all():
            db.delete(t)
        sess = db.get(AudioSession, audio_session_id)
        audio_path_str = sess.audio_path if sess else None
        if sess is not None:
            db.delete(sess)
        db.commit()

    if audio_path_str:
        try:
            p = Path(audio_path_str).resolve()
            storage_root = settings.storage_dir.resolve()
            p.relative_to(storage_root)  # path-traversal guard
            p.unlink(missing_ok=True)
        except (ValueError, OSError):
            # Path outside storage or unlink failed — DB rows are gone either way.
            pass

    return RedirectResponse("/", status_code=303)


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


# ──────────────────────────────────────────────────────────────────────────────
# /config — editable settings page
# ──────────────────────────────────────────────────────────────────────────────

# Fields the UI lets you edit. Things deliberately left off: username,
# password hash, JWT secret (rotate via CLI), storage paths (changing breaks
# data), and host/port (server-level, restart-only).
_CONFIG_SECTIONS: list[tuple[str, list[dict[str, Any]]]] = [
    ("STT (whisper.cpp)", [
        {"key": "OMILOG_STT_BASE_URL", "label": "Server URL", "kind": "text",
         "help": "Tailnet URL like http://gpu-host.tailnet:8080. Empty disables STT."},
        {"key": "OMILOG_STT_INFERENCE_PATH", "label": "Inference path", "kind": "text",
         "help": "Usually /inference."},
        {"key": "OMILOG_STT_LANGUAGE", "label": "Language", "kind": "text",
         "help": "ISO code (fr, en) or 'auto' for detection."},
        {"key": "OMILOG_STT_TIMEOUT_S", "label": "Timeout (s)", "kind": "number",
         "step": 5, "min": 10, "max": 600},
        {"key": "OMILOG_STT_MODEL_NAME", "label": "Model label", "kind": "text",
         "help": "Stamped on each transcript row. The server uses its actually-loaded model regardless."},
        {"key": "OMILOG_STT_INITIAL_PROMPT", "label": "Initial prompt", "kind": "text",
         "help": "Optional short text that biases Whisper toward your vocabulary (proper nouns, technical terms, dominant language). Short and concrete is better than long and abstract."},
        {"key": "OMILOG_STT_TEMPERATURE", "label": "Temperature", "kind": "number",
         "step": 0.1, "min": 0.0, "max": 1.0,
         "help": "0 = deterministic. Bump to 0.2 if Whisper hallucinates on noisy or low-volume audio."},
    ]),
    ("LLM (llama.cpp)", [
        {"key": "OMILOG_LLM_BASE_URL", "label": "Server URL", "kind": "text",
         "help": "OpenAI-compatible endpoint, e.g. http://gpu-host.tailnet:8081/v1. Empty disables LLM."},
        {"key": "OMILOG_LLM_MODEL", "label": "Model name", "kind": "text",
         "help": "Sent in the request; llama-server ignores it but other backends may not."},
        {"key": "OMILOG_LLM_TEMPERATURE", "label": "Temperature", "kind": "number",
         "step": 0.05, "min": 0, "max": 2},
        {"key": "OMILOG_LLM_MAX_TOKENS", "label": "Max tokens", "kind": "number",
         "step": 512, "min": 256, "max": 32768,
         "help": "Hit truncations? Bump. 4096 fits most conversations."},
        {"key": "OMILOG_LLM_TIMEOUT_S", "label": "Timeout (s)", "kind": "number",
         "step": 10, "min": 10, "max": 900},
        {"key": "OMILOG_LLM_PRIMARY_LANGUAGE", "label": "Primary language hint", "kind": "text",
         "help": "Optional hint baked into the LLM prompt, e.g. 'French' or 'Spanish'. Empty = language-neutral. Whisper handles actual language detection from audio independently."},
    ]),
    ("VAD (segmentation)", [
        {"key": "OMILOG_VAD_ENABLED", "label": "Enabled", "kind": "checkbox"},
        {"key": "OMILOG_VAD_THRESHOLD_DB", "label": "Silence threshold (dB)", "kind": "number",
         "step": 1, "min": -80, "max": -10,
         "help": "Lower catches quieter speech."},
        {"key": "OMILOG_VAD_GAP_SECONDS", "label": "Conversation gap (s)", "kind": "number",
         "step": 5, "min": 5, "max": 600,
         "help": "Silence ≥ this becomes a new conversation."},
        {"key": "OMILOG_VAD_MIN_SILENCE_SECONDS", "label": "Min silence (s)", "kind": "number",
         "step": 0.1, "min": 0.1, "max": 10},
        {"key": "OMILOG_VAD_PAD_SECONDS", "label": "Pad (s)", "kind": "number",
         "step": 0.1, "min": 0, "max": 3,
         "help": "Symmetric padding so the first/last word isn't clipped."},
    ]),
    ("WS rollover", [
        {"key": "OMILOG_WS_ROLLOVER_SECONDS", "label": "Rollover interval (s)", "kind": "number",
         "step": 60, "min": 0, "max": 7200,
         "help": "Close & process current segment every N seconds without dropping the WS. 0 disables."},
        {"key": "OMILOG_WS_RECEIVE_TIMEOUT_SECONDS", "label": "Receive timeout (s)", "kind": "number",
         "step": 1, "min": 1, "max": 30,
         "help": "How often the WS loop wakes to check rollover. Default 5s is fine."},
    ]),
    ("Diarization (sherpa-onnx)", [
        {"key": "OMILOG_DIARIZATION_ENABLED", "label": "Enabled", "kind": "checkbox",
         "help": "Requires the diarization extra installed + models downloaded."},
        {"key": "OMILOG_DIARIZATION_SEGMENTATION_MODEL", "label": "Segmentation model path", "kind": "text"},
        {"key": "OMILOG_DIARIZATION_EMBEDDING_MODEL", "label": "Embedding model path", "kind": "text"},
        {"key": "OMILOG_DIARIZATION_NUM_CLUSTERS", "label": "Force speaker count", "kind": "number",
         "step": 1, "min": -1, "max": 32,
         "help": "-1 = auto. Set to a positive number to pin the clusterer to exactly that many speakers per conversation. Use when you know the typical count (2-4 people) and over-segmentation is breaking things."},
        {"key": "OMILOG_DIARIZATION_CLUSTER_THRESHOLD", "label": "Cluster threshold (sherpa)", "kind": "number",
         "step": 0.05, "min": 0.1, "max": 0.95,
         "help": "sherpa-onnx's internal cluster cutoff. Limited effect in practice — use the post-merge threshold below for reliable folding."},
        {"key": "OMILOG_DIARIZATION_POST_MERGE_THRESHOLD", "label": "Post-merge threshold", "kind": "number",
         "step": 0.05, "min": 0.5, "max": 1.0,
         "help": "Cosine similarity above which two clusters get folded by our own in-Python second pass. 1.0 = disabled. 0.75-0.85 is the useful range for over-split conversations (e.g. 9 clusters that are really 2 people). Scales naturally with the real speaker count, unlike forcing num_clusters."},
        {"key": "OMILOG_DIARIZATION_MIN_SPEECH_SECONDS", "label": "Min speech seconds", "kind": "number",
         "step": 0.1, "min": 0.1, "max": 3.0,
         "help": "Minimum continuous speech to be considered a turn. Default 0.3 is permissive; bump to 0.7-1.0 to drop short noise / fillers from the cluster pool."},
        {"key": "OMILOG_DIARIZATION_MIN_SILENCE_SECONDS", "label": "Min silence seconds", "kind": "number",
         "step": 0.1, "min": 0.1, "max": 3.0,
         "help": "Minimum gap between speech that counts as a turn boundary."},
        {"key": "OMILOG_SPEAKER_MATCH_THRESHOLD", "label": "Cross-conversation match threshold", "kind": "number",
         "step": 0.05, "min": 0.1, "max": 0.95,
         "help": "Higher = stricter cross-conversation linking; lower = more aggressive merging across conversations. Distinct from the cluster threshold above (this one runs after diarization, against stored Speaker rows)."},
    ]),
    ("Calendar (ICS feed)", [
        {"key": "OMILOG_ICS_FEED_TOKEN", "label": "Feed token", "kind": "text",
         "help": "Empty disables. Generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))'."},
        {"key": "OMILOG_ICS_CALNAME", "label": "Calendar display name", "kind": "text"},
        {"key": "OMILOG_ICS_FEED_MIN_CONFIDENCE", "label": "Min confidence", "kind": "number",
         "step": 0.05, "min": 0, "max": 1},
    ]),
    ("Extraction categories", [
        {"key": "OMILOG_EXTRACT_CALENDAR_EVENTS", "label": "Calendar events", "kind": "checkbox",
         "help": "When off, the prompt schema drops the calendar_events section and any returned events are ignored. Existing events stay in the DB."},
        {"key": "OMILOG_EXTRACT_ACTION_ITEMS", "label": "Action items", "kind": "checkbox",
         "help": "Concrete tasks with owners. When off, /actions stops getting new entries."},
        {"key": "OMILOG_EXTRACT_DECISIONS", "label": "Decisions", "kind": "checkbox",
         "help": "Conclusions, plans agreed on, preferences settled. Surfaced on the conversation detail page."},
        {"key": "OMILOG_EXTRACT_PEOPLE_MENTIONED", "label": "People mentioned", "kind": "checkbox",
         "help": "Names of people referenced in the conversation. Disabling stops feeding the future CRM page."},
        {"key": "OMILOG_EXTRACT_TOPICS", "label": "Topics", "kind": "checkbox",
         "help": "Free-text topic tags shown under each conversation's summary."},
    ]),
    ("Storage / retention", [
        {"key": "OMILOG_AUDIO_RETENTION_DAYS", "label": "Audio retention (days)", "kind": "number",
         "step": 1, "min": 0, "max": 3650,
         "help": "After this many days, .opus files for done conversations get auto-deleted; transcripts and extractions are always kept. 0 disables rotation. Pin individual conversations with the 📌 button to exempt them."},
    ]),
    ("Other", [
        {"key": "OMILOG_LOCAL_TIMEZONE", "label": "Local timezone", "kind": "text",
         "help": "For resolving 'demain'/'next week' in extracted events."},
        {"key": "OMILOG_LOG_LEVEL", "label": "Log level", "kind": "select",
         "options": ["DEBUG", "INFO", "WARNING", "ERROR"]},
        {"key": "OMILOG_COOKIE_SECURE", "label": "Cookie Secure flag", "kind": "checkbox",
         "help": "Enable when fronted by HTTPS (Caddy / tailscale serve)."},
        {"key": "OMILOG_PIPELINE_POLL_SECONDS", "label": "Pipeline poll (s)", "kind": "number",
         "step": 0.5, "min": 0.5, "max": 30},
    ]),
]


def _current_value(key: str) -> str:
    """Read the in-memory setting matching `key`, return as a form-friendly str."""
    attr = key.removeprefix("OMILOG_").lower()
    val = getattr(settings, attr, None)
    if val is None:
        return ""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, Path):
        return str(val)
    return str(val)


def _config_sections_with_values() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for title, fields in _CONFIG_SECTIONS:
        out.append(
            {
                "title": title,
                "fields": [
                    {**f, "value": _current_value(f["key"])} for f in fields
                ],
            }
        )
    return out


@router.get("/config", response_class=HTMLResponse)
async def config_page(request: Request, user: UIUser):
    return templates.TemplateResponse(
        request,
        "config.html",
        {
            "user": user,
            "sections": _config_sections_with_values(),
            "env_path": str(Path(".env").resolve()),
        },
    )


@router.post("/config", response_class=HTMLResponse)
async def config_save(request: Request, user: UIUser):
    """Write submitted values back to .env preserving comments and unrelated
    keys. Only the keys listed in _CONFIG_SECTIONS are ever touched."""
    form = await request.form()
    known_keys: list[dict[str, Any]] = []
    for _title, fields in _CONFIG_SECTIONS:
        known_keys.extend(fields)

    updates: dict[str, str] = {}
    for field in known_keys:
        key = field["key"]
        kind = field["kind"]
        if kind == "checkbox":
            updates[key] = "true" if key in form else "false"
            continue
        raw = form.get(key, "")
        raw = str(raw).strip()
        if "\n" in raw or "\r" in raw or "\x00" in raw:
            raise HTTPException(400, f"invalid value for {key}")
        updates[key] = raw

    env_path = Path(".env")
    _write_env_updates(env_path, updates)

    return HTMLResponse(
        '<small style="color: var(--pico-color-green-500)">'
        "✓ Saved to <code>.env</code>. Restart the server (Ctrl-C + "
        "<code>./scripts/start.sh</code>) to apply."
        "</small>"
    )


@router.get("/config/prompt", response_class=HTMLResponse)
async def config_prompt_page(request: Request, user: UIUser):
    from ..pipeline.extract import render_default_system_prompt

    prompt_path = settings.llm_system_prompt_file
    is_customized = prompt_path.exists() and prompt_path.read_text(encoding="utf-8").strip() != ""
    if is_customized:
        current_prompt = prompt_path.read_text(encoding="utf-8")
    else:
        current_prompt = render_default_system_prompt(settings.llm_primary_language)
    return templates.TemplateResponse(
        request,
        "config_prompt.html",
        {
            "user": user,
            "prompt_path": str(prompt_path.resolve()),
            "current_prompt": current_prompt,
            "is_customized": is_customized,
            "default_prompt": render_default_system_prompt(settings.llm_primary_language),
            "primary_language": settings.llm_primary_language,
        },
    )


@router.post("/config/prompt", response_class=HTMLResponse)
async def config_prompt_save(
    user: UIUser,
    prompt: Annotated[str, Form()] = "",
    action: Annotated[str, Form()] = "save",
):
    prompt_path = settings.llm_system_prompt_file
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    if action == "reset":
        try:
            prompt_path.unlink(missing_ok=True)
        except OSError:
            pass
        return RedirectResponse("/config/prompt", status_code=303)
    if not prompt.strip():
        # Empty save is treated as reset to avoid storing a no-op override.
        try:
            prompt_path.unlink(missing_ok=True)
        except OSError:
            pass
        return RedirectResponse("/config/prompt", status_code=303)
    prompt_path.write_text(prompt, encoding="utf-8")
    return RedirectResponse("/config/prompt", status_code=303)


def _write_env_updates(env_path: Path, updates: dict[str, str]) -> None:
    """In-place rewrite. Empty/missing file → create with just these keys."""
    existing = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in existing:
        matched = False
        for key, val in updates.items():
            if line.startswith(f"{key}="):
                out.append(f"{key}={val}")
                seen.add(key)
                matched = True
                break
        if not matched:
            out.append(line)
    for key, val in updates.items():
        if key not in seen:
            out.append(f"{key}={val}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


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
            "current_diarize": _current_diarize_defaults(),
            "diarization_enabled": settings.diarization_enabled,
            "diarization_available": diarize_mod.DIARIZATION_AVAILABLE,
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


def _current_diarize_defaults() -> dict[str, Any]:
    """Snapshot of the diarization-tunable settings, in the shape the
    /tune/<id> form expects. Distinct from VAD defaults — diarization
    runs AFTER VAD, on the per-conversation WAV."""
    return {
        "num_clusters": settings.diarization_num_clusters,
        "cluster_threshold": settings.diarization_cluster_threshold,
        "post_merge_threshold": settings.diarization_post_merge_threshold,
        "min_speech_s": settings.diarization_min_speech_seconds,
        "min_silence_s": settings.diarization_min_silence_seconds,
    }


@router.post("/tune/{session_id}/diarize", response_class=HTMLResponse)
async def tune_diarize(
    request: Request,
    user: UIUser,
    session_id: UUID,
    num_clusters: Annotated[int, Form()],
    cluster_threshold: Annotated[float, Form()],
    min_speech_s: Annotated[float, Form()],
    min_silence_s: Annotated[float, Form()],
    post_merge_threshold: Annotated[float, Form()] = 1.0,
):
    """HTMX endpoint: re-run diarization on this session with the supplied
    params, return a fragment showing the resulting turn count, per-cluster
    talk time, and the transcript re-labeled with USER/S1/S2 from the new
    diarization. No DB writes — purely a preview.

    The model is reloaded from disk on every call because the production
    diarizer is process-cached with its own settings; that's a 5-10s cost
    per tune iteration on a Pi but lets the user iterate without
    restarting the server.
    """
    if not settings.diarization_enabled:
        raise HTTPException(
            400, "diarization is disabled in settings (OMILOG_DIARIZATION_ENABLED)"
        )
    if not diarize_mod.DIARIZATION_AVAILABLE:
        raise HTTPException(
            400,
            "sherpa-onnx isn't installed in this venv — see "
            "docs/diarization-setup.md",
        )

    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None or sess.user_id != user:
            raise HTTPException(404, "session not found")
        if not sess.audio_path:
            raise HTTPException(404, "session has no audio_path")
        audio_path = Path(sess.audio_path)
        transcript = db.exec(
            select(Transcript)
            .where(Transcript.audio_session_id == sess.id)
            .order_by(Transcript.created_at.desc())
            .limit(1)
        ).first()
    if not audio_path.exists():
        raise HTTPException(404, "audio file missing")

    started = monotonic()
    try:
        wav_bytes = await transcode_to_wav_bytes(audio_path)
    except TranscodeError as e:
        return templates.TemplateResponse(
            request,
            "_tune_diarize_results.html",
            {"error": f"ffmpeg: {e}"},
            status_code=400,
        )

    # Clamp params to the same ranges /config and .env enforce. Stops a
    # user from setting num_clusters=-2 and getting a confusing C++ error.
    num_clusters = max(-1, min(int(num_clusters), 32))
    cluster_threshold = max(0.1, min(float(cluster_threshold), 0.95))
    min_speech_s = max(0.1, min(float(min_speech_s), 3.0))
    min_silence_s = max(0.1, min(float(min_silence_s), 3.0))
    # 1.0 = no post-merge, anything below clamped to [0.5, 1.0]. Going
    # lower than 0.5 risks folding distinct voices indiscriminately.
    post_merge_threshold = max(0.5, min(float(post_merge_threshold), 1.0))

    try:
        turns = await diarize_mod.diarize_uncached(
            wav_bytes,
            seg_path=settings.diarization_segmentation_model,
            emb_path=settings.diarization_embedding_model,
            min_speech_s=min_speech_s,
            min_silence_s=min_silence_s,
            num_threads=settings.diarization_num_threads,
            num_clusters=num_clusters,
            cluster_threshold=cluster_threshold,
        )
        # Apply the same post-merge the runner uses, with the user's
        # tune-page threshold so they can preview the effect of changing
        # it without restart.
        if post_merge_threshold < 1.0:
            turns = await diarize_mod.post_merge_clusters(
                wav_bytes,
                turns,
                emb_path=settings.diarization_embedding_model,
                threshold=post_merge_threshold,
                num_threads=settings.diarization_num_threads,
            )
    except diarize_mod.DiarizationError as e:
        return templates.TemplateResponse(
            request,
            "_tune_diarize_results.html",
            {"error": str(e)},
            status_code=400,
        )
    elapsed_ms = int((monotonic() - started) * 1000)

    # Raw cluster talk-time (SPEAKER_00, SPEAKER_01…) — useful to see
    # whether the clusterer hit the expected K.
    cluster_durations: dict[str, float] = {}
    for t in turns:
        d = t["end"] - t["start"]
        cluster_durations[t["speaker"]] = cluster_durations.get(t["speaker"], 0) + d

    # Overlay on the transcript and re-label to USER/S1/… so the user can
    # eyeball the result the way the rest of the UI displays speakers.
    relabeled_segments: list[dict[str, Any]] = []
    if transcript and transcript.segments_json:
        try:
            raw_segs = json.loads(transcript.segments_json)
        except json.JSONDecodeError:
            raw_segs = []
        # Strip any existing speaker assignments so the new ones aren't
        # mixed with stale ones (assign only WRITES; it doesn't clear).
        clean = [
            {k: v for k, v in s.items() if k != "speaker"} for s in raw_segs
        ]
        clean = diarize_mod.assign_speakers_to_segments(clean, turns)
        clean = diarize_mod.relabel_user_and_others(clean)
        relabeled_segments = clean

    relabeled_durations: dict[str, float] = {}
    for s in relabeled_segments:
        sp = s.get("speaker")
        if not sp:
            continue
        start_s = float(s.get("start", 0) or 0)
        end_s = float(s.get("end", start_s) or start_s)
        if end_s > start_s:
            relabeled_durations[sp] = relabeled_durations.get(sp, 0) + (end_s - start_s)

    return templates.TemplateResponse(
        request,
        "_tune_diarize_results.html",
        {
            "turns_count": len(turns),
            "clusters_count": len(cluster_durations),
            "cluster_durations": dict(sorted(
                cluster_durations.items(), key=lambda kv: -kv[1]
            )),
            "relabeled_segments": relabeled_segments,
            "relabeled_durations": dict(sorted(
                relabeled_durations.items(), key=lambda kv: -kv[1]
            )),
            "elapsed_ms": elapsed_ms,
            "params": {
                "num_clusters": num_clusters,
                "cluster_threshold": cluster_threshold,
                "post_merge_threshold": post_merge_threshold,
                "min_speech_s": min_speech_s,
                "min_silence_s": min_silence_s,
            },
        },
    )


@router.post("/tune/apply-diarize-defaults", response_class=HTMLResponse)
async def tune_apply_diarize_defaults(
    user: UIUser,
    num_clusters: Annotated[int, Form()],
    cluster_threshold: Annotated[float, Form()],
    min_speech_s: Annotated[float, Form()],
    min_silence_s: Annotated[float, Form()],
    post_merge_threshold: Annotated[float, Form()] = 1.0,
):
    """Mirror of /tune/apply-defaults but for the diarize knobs. Restart is
    required because the diarizer is process-cached."""
    env_path = Path(".env")
    if not env_path.exists():
        raise HTTPException(400, ".env file not found on server")
    new_values = {
        "OMILOG_DIARIZATION_NUM_CLUSTERS": str(int(num_clusters)),
        "OMILOG_DIARIZATION_CLUSTER_THRESHOLD": str(float(cluster_threshold)),
        "OMILOG_DIARIZATION_POST_MERGE_THRESHOLD": str(float(post_merge_threshold)),
        "OMILOG_DIARIZATION_MIN_SPEECH_SECONDS": str(float(min_speech_s)),
        "OMILOG_DIARIZATION_MIN_SILENCE_SECONDS": str(float(min_silence_s)),
    }
    _write_env_updates(env_path, new_values)
    return HTMLResponse(
        '<small style="color: var(--pico-color-green-500)">'
        "✓ Saved to <code>.env</code>. The diarizer is process-cached, so "
        "restart the server (Ctrl-C + <code>./scripts/start.sh</code>) "
        "for the new values to take effect."
        "</small>"
    )


def _cascade_delete_session(db: Session, session_id: UUID) -> list[str]:
    """Delete an AudioSession plus every FK dependent in safe order.

    Returns the list of audio file paths that should be unlinked from disk
    after the transaction commits.

    Handles:
      - Child AudioSessions (recursive, for segmented parents)
      - Transcripts pointing to this session
      - Conversations pointing to this session AND their children
        (PersonMention, ActionItem, CalendarEvent, WakeInvocation)
      - The AudioSession itself
    """
    sess = db.get(AudioSession, session_id)
    if sess is None:
        return []

    files: list[str] = []
    if sess.audio_path:
        files.append(sess.audio_path)

    # Recurse into children first (parents reference nobody, children
    # reference parents via parent_id).
    for child in db.exec(
        select(AudioSession).where(AudioSession.parent_id == session_id)
    ).all():
        files.extend(_cascade_delete_session(db, child.id))

    # Clean up any Conversation tied to this session and its grandchildren.
    for conv in db.exec(
        select(Conversation).where(Conversation.audio_session_id == session_id)
    ).all():
        for inv in db.exec(
            select(WakeInvocation).where(WakeInvocation.conversation_id == conv.id)
        ).all():
            db.delete(inv)
        for m in db.exec(
            select(PersonMention).where(PersonMention.conversation_id == conv.id)
        ).all():
            db.delete(m)
        for a in db.exec(
            select(ActionItem).where(ActionItem.conversation_id == conv.id)
        ).all():
            db.delete(a)
        for e in db.exec(
            select(CalendarEvent).where(CalendarEvent.conversation_id == conv.id)
        ).all():
            db.delete(e)
        db.delete(conv)

    # Then transcripts.
    for t in db.exec(
        select(Transcript).where(Transcript.audio_session_id == session_id)
    ).all():
        db.delete(t)

    # Flush so dependents are gone before the session itself goes.
    db.flush()
    db.delete(sess)
    return files


@router.post("/sessions/{session_id}/dismiss")
async def dismiss_session(user: UIUser, session_id: UUID):
    """Delete an AudioSession row, every FK dependent, and the audio file(s).
    Used by the pending panel to clear failed or stuck rows from the index view.

    Sessions in the pending panel can be at any pipeline stage — failed during
    VAD (just a raw file), failed during STT (no transcript yet), failed during
    LLM (transcript exists), or even segmented parents whose children are still
    pending. Cascading covers all cases.

    Returns an empty 200 so HTMX's outerHTML swap removes the row from the DOM.
    """
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None or sess.user_id != user:
            raise HTTPException(404, "session not found")
        files = _cascade_delete_session(db, session_id)
        db.commit()

    storage_root = settings.storage_dir.resolve()
    for path_str in files:
        try:
            p = Path(path_str).resolve()
            p.relative_to(storage_root)  # path-traversal guard
            p.unlink(missing_ok=True)
        except (ValueError, OSError):
            pass

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
