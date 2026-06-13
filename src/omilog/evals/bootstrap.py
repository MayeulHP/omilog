"""Create an eval case from a processed session — shared by the
scripts/eval_bootstrap.py CLI and the /eval web UI.

Copies the session audio and exports the machine transcript (+ speaker
labels when present) as editing rows for hand-correction. Optional "HQ
draft" mode re-transcribes the audio right now with quality-leaning
settings (pinned language + a vocabulary prompt built from known speaker /
people names) so the human starts from a better draft — less to fix.
Caveat documented in eval/README.md: the draft still comes from the same
model family being evaluated, so anything not actually checked against the
audio biases WER optimistically.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import UUID

from sqlmodel import Session, select

from ..config import settings
from ..db import engine
from ..models import AudioSession, PersonMention, Speaker, Transcript
from ..pipeline.audio import transcode_to_wav_bytes
from ..pipeline.diarize import assign_speakers_to_segments
from ..pipeline.stt import collapse_repeated_segments, transcribe_wav
from .cases import (
    CASE_NAME_RE,
    rows_from_segments,
    turns_from_segments,
    update_case_meta,
    write_reference_files,
)


class BootstrapError(RuntimeError):
    """User-presentable failure: missing audio, no transcript, bad name…"""


def _load_source(session_id: UUID):
    with Session(engine) as db:
        sess = db.get(AudioSession, session_id)
        if sess is None:
            raise BootstrapError(f"session {session_id} not found")
        transcript = db.exec(
            select(Transcript)
            .where(Transcript.audio_session_id == session_id)
            .order_by(Transcript.created_at.desc())  # type: ignore[attr-defined]
            .limit(1)
        ).first()
        return sess, transcript


def _known_names(user_id: str, limit: int = 15) -> list[str]:
    """Named speakers + most-mentioned people — the proper nouns Whisper is
    most likely to butcher and an initial prompt is most likely to fix."""
    with Session(engine) as db:
        speaker_names = [
            s.name
            for s in db.exec(select(Speaker).where(Speaker.user_id == user_id)).all()
            if s.name
        ]
        mentioned = [p.name for p in db.exec(select(PersonMention)).all()]
    counts: dict[str, int] = {}
    for n in mentioned:
        counts[n] = counts.get(n, 0) + 1
    ranked = sorted(counts, key=lambda n: -counts[n])
    out: list[str] = []
    for n in speaker_names + ranked:
        if n not in out:
            out.append(n)
    return out[:limit]


async def _hq_segments(
    audio_path: Path,
    stored_segments: list[dict],
    *,
    language_hint: str | None,
    user_id: str,
) -> list[dict]:
    """Re-transcribe for a better draft, then carry the stored speaker
    labels over onto the fresh segments by time overlap."""
    if not settings.stt_base_url:
        raise BootstrapError("HQ draft needs OMILOG_STT_BASE_URL configured")
    language = settings.stt_language
    if language == "auto":
        language = language_hint or "auto"
    names = _known_names(user_id)
    prompt = settings.stt_initial_prompt.strip()
    if names:
        prompt = (prompt + " " if prompt else "") + ", ".join(names) + "."

    wav_bytes = await transcode_to_wav_bytes(audio_path)
    result = await transcribe_wav(
        wav_bytes,
        base_url=settings.stt_base_url,
        inference_path=settings.stt_inference_path,
        language=language,
        timeout_s=settings.stt_timeout_s,
        initial_prompt=prompt,
        temperature=0.0,
    )
    segments = collapse_repeated_segments(list(result.segments or []))
    speaker_turns = turns_from_segments(stored_segments)
    if speaker_turns:
        segments = assign_speakers_to_segments(segments, speaker_turns)
    return segments


async def create_case(
    session_id: UUID,
    *,
    name: str | None = None,
    cases_dir: Path | None = None,
    hq: bool = False,
    force: bool = False,
) -> Path:
    """Build eval/cases/<name>/ from a session; returns the case directory.

    Raises BootstrapError with a user-presentable message on any precondition
    failure (audio rotated away, no transcript yet, name collision…).
    """
    cases_dir = cases_dir if cases_dir is not None else settings.eval_cases_dir
    sess, transcript = _load_source(session_id)

    audio_path = Path(sess.audio_path) if sess.audio_path else None
    if audio_path is None or not audio_path.exists():
        raise BootstrapError(
            "audio file is no longer on disk (rotated?) — pin (📌) sessions "
            "you plan to use for eval"
        )
    if transcript is None:
        raise BootstrapError(
            "no transcript yet — run the pipeline (or replay_session.py) first "
            "so there is a machine draft to correct"
        )

    name = (name or "").strip() or f"{sess.started_at:%Y-%m-%d}-{str(session_id)[:8]}"
    if not CASE_NAME_RE.match(name):
        raise BootstrapError(
            "case name must be letters/digits/dot/dash/underscore (≤80 chars)"
        )
    case_dir = cases_dir / name
    if case_dir.exists() and not force:
        raise BootstrapError(f"case {name!r} already exists")

    segments: list[dict] = []
    if transcript.segments_json:
        try:
            loaded = json.loads(transcript.segments_json)
            if isinstance(loaded, list):
                segments = [s for s in loaded if isinstance(s, dict)]
        except ValueError:
            pass

    if hq:
        segments = await _hq_segments(
            audio_path,
            segments,
            language_hint=transcript.language,
            user_id=sess.user_id,
        )

    rows = rows_from_segments(segments)
    if not rows and transcript.text.strip():
        # No usable segments (very old transcript?) — fall back to one
        # timing-less row per line so there's still something to correct.
        rows = [{"start": 0.0, "end": 0.0, "text": line.strip()}
                for line in transcript.text.splitlines() if line.strip()]
    if not rows:
        raise BootstrapError("transcript is empty — nothing to label")

    case_dir.mkdir(parents=True, exist_ok=True)
    suffix = audio_path.suffix or ".opus"
    shutil.copy2(audio_path, case_dir / f"audio{suffix}")
    write_reference_files(case_dir, rows)
    update_case_meta(
        case_dir,
        source_session_id=str(session_id),
        recorded_at=sess.started_at.isoformat(),
        duration_s=sess.duration_s,
        language=transcript.language,
        hq_draft=hq,
        verified=False,
        notes="",
    )
    return case_dir
