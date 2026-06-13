"""/eval — browser labeling UI for the ground-truth eval set.

Front-end over the same case directories the CLI scripts use (the files
stay the source of truth; see src/omilog/evals/cases.py for the format).
The labeling pain this removes: bouncing between an audio player and two
text files with timestamps in your head — here the player, the row editor
and the per-case metrics live on one page, and clicking a row seeks the
audio to it.
"""

import shutil
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from sqlmodel import Session

from ..config import settings
from ..db import engine
from ..evals import runner as eval_runner
from ..evals.bootstrap import BootstrapError, create_case
from ..evals.cases import (
    CASE_NAME_RE,
    EvalCase,
    load_case,
    load_cases,
    update_case_meta,
    write_reference_files,
)
from ..models import Conversation
from .auth import UIUser
from .routes import templates

router = APIRouter()

_AUDIO_MEDIA_TYPES = {
    ".opus": "audio/ogg",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


def _case_dir_or_404(name: str) -> Path:
    """Validate a URL case name and resolve it inside the cases dir.
    CASE_NAME_RE alone blocks traversal; the resolve() check is belt and
    braces (symlinks, future regex edits)."""
    if not CASE_NAME_RE.match(name):
        raise HTTPException(404, "case not found")
    cases_root = settings.eval_cases_dir.resolve()
    case_dir = (settings.eval_cases_dir / name).resolve()
    try:
        case_dir.relative_to(cases_root)
    except ValueError:
        raise HTTPException(404, "case not found") from None
    if not case_dir.is_dir():
        raise HTTPException(404, "case not found")
    return case_dir


def _case_or_404(name: str) -> EvalCase:
    case = load_case(_case_dir_or_404(name))
    if case is None:
        raise HTTPException(404, "case is malformed — fix or delete its files on disk")
    return case


def _editing_rows(case: EvalCase) -> list[dict[str, Any]]:
    """Rows for the editor table. reference_turns.json is the row store
    (bootstrap writes text into it); cases hand-made without it fall back
    to one timing-less row per reference.txt line."""
    if case.reference_turns:
        return [
            {
                "start": float(t.get("start", 0) or 0),
                "end": float(t.get("end", 0) or 0),
                "speaker": str(t.get("speaker") or ""),
                "text": str(t.get("text") or ""),
            }
            for t in case.reference_turns
        ]
    return [
        {"start": 0.0, "end": 0.0, "speaker": "", "text": line.strip()}
        for line in case.reference_text.splitlines()
        if line.strip()
    ]


@router.get("/eval", response_class=HTMLResponse)
async def eval_index(request: Request, user: UIUser):
    cases = load_cases(settings.eval_cases_dir)
    latest = eval_runner.latest_metrics_by_case(settings.eval_results_dir)
    return templates.TemplateResponse(
        request,
        "eval_index.html",
        {"cases": cases, "latest": latest},
    )


@router.post("/eval/create/{conversation_id}")
async def eval_create(
    user: UIUser,
    conversation_id: UUID,
    name: Annotated[str, Form()] = "",
    hq: Annotated[bool, Form()] = False,
):
    with Session(engine) as db:
        conv = db.get(Conversation, conversation_id)
        if conv is None or conv.user_id != user:
            raise HTTPException(404, "conversation not found")
        session_id = conv.audio_session_id
    try:
        case_dir = await create_case(session_id, name=name or None, hq=hq)
    except BootstrapError as e:
        # Plain-HTML error: the button is a regular form post, so a readable
        # page with a way back beats a JSON 400.
        return HTMLResponse(
            "<main class='container'><article>"
            f"<strong>Couldn't create eval case</strong><br><small>{e}</small><br>"
            f"<a href='/conversations/{conversation_id}'>← back to conversation</a>"
            "</article></main>",
            status_code=400,
        )
    return RedirectResponse(f"/eval/{case_dir.name}", status_code=303)


@router.get("/eval/{name}", response_class=HTMLResponse)
async def eval_case_page(request: Request, user: UIUser, name: str):
    case = _case_or_404(name)
    latest = eval_runner.latest_metrics_by_case(settings.eval_results_dir).get(name)
    return templates.TemplateResponse(
        request,
        "eval_case.html",
        {
            "case": case,
            "rows": _editing_rows(case),
            "latest": latest,
            "saved": request.query_params.get("saved") == "1",
            "speaker_options": ["USER", "S1", "S2", "S3", "S4"],
        },
    )


@router.post("/eval/{name}/save")
async def eval_save(
    user: UIUser,
    name: str,
    start: Annotated[list[float], Form()] = [],
    end: Annotated[list[float], Form()] = [],
    speaker: Annotated[list[str], Form()] = [],
    text: Annotated[list[str], Form()] = [],
    verified: Annotated[bool, Form()] = False,
):
    case_dir = _case_dir_or_404(name)
    n = min(len(start), len(end), len(speaker), len(text))
    rows: list[dict[str, Any]] = []
    for i in range(n):
        row_text = text[i].strip()
        row_speaker = speaker[i].strip()
        if not row_text and not row_speaker:
            continue  # fully blank row — treat as deleted
        row: dict[str, Any] = {
            "start": round(start[i], 2),
            "end": round(end[i], 2),
            "text": row_text,
        }
        if row_speaker:
            row["speaker"] = row_speaker
        rows.append(row)
    if not rows:
        raise HTTPException(400, "nothing to save — every row is blank")
    write_reference_files(case_dir, rows)
    update_case_meta(case_dir, verified=verified)
    return RedirectResponse(f"/eval/{name}?saved=1", status_code=303)


@router.get("/eval/{name}/audio")
async def eval_audio(user: UIUser, name: str):
    case_dir = _case_dir_or_404(name)
    candidates = sorted(case_dir.glob("audio.*"))
    if not candidates:
        raise HTTPException(404, "case has no audio file")
    audio = candidates[0]
    media_type = _AUDIO_MEDIA_TYPES.get(audio.suffix.lower(), "application/octet-stream")
    return FileResponse(audio, media_type=media_type)


@router.post("/eval/{name}/run", response_class=HTMLResponse)
async def eval_run_one(request: Request, user: UIUser, name: str):
    """HTMX: score this case against the live STT/diarization config.
    Slow (real STT call, diarization on the host) — the button shows a
    spinner. Appends to the same history the CLI writes."""
    case = _case_or_404(name)
    do_diarize, reason = eval_runner.resolve_do_diarize(None)
    error: str | None = None
    row: dict[str, Any] = {}
    if not settings.stt_base_url:
        error = "OMILOG_STT_BASE_URL is not configured"
    else:
        try:
            row = await eval_runner.eval_case(
                case,
                do_diarize=do_diarize,
                reuse_stt=True,
                cache_dir=settings.eval_results_dir / "cache",
            )
        except Exception as e:  # noqa: BLE001 — render, don't 500 the fragment
            error = f"{type(e).__name__}: {e}"
    if not error:
        record = eval_runner.build_record(
            {name: row}, note=f"web:{name}", collar=0.25, with_diarization=do_diarize
        )
        eval_runner.append_history(settings.eval_results_dir, record)
    return templates.TemplateResponse(
        request,
        "_eval_run_result.html",
        {"row": row, "error": error, "diarize_skip_reason": reason},
    )


@router.post("/eval/{name}/delete")
async def eval_delete(user: UIUser, name: str):
    case_dir = _case_dir_or_404(name)
    shutil.rmtree(case_dir)
    return RedirectResponse("/eval", status_code=303)
