"""POST /api/audio/upload — drop an audio file straight into the pipeline.

Useful for two things:
  1. Testing the STT pipeline end-to-end without a live Omi pairing — push any
     existing WAV/MP3/Opus and watch the runner pick it up.
  2. Re-importing audio captured by other means.

We don't try to validate codec/duration here; ffmpeg in the pipeline will
reject anything it can't decode and the session ends up in `failed` with a
readable error_msg.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlmodel import Session

from ..auth import current_user
from ..config import settings
from ..db import engine
from ..models import AudioSession, SessionStatus

router = APIRouter(prefix="/api", tags=["audio"])
logger = logging.getLogger("omilog.audio_upload")

_MAX_BYTES = 500 * 1024 * 1024  # 500 MB cap — single conversation upper bound
_ALLOWED_SUFFIXES = {".wav", ".mp3", ".opus", ".ogg", ".m4a", ".aac", ".flac", ".webm"}


@router.post("/audio/upload")
async def upload_audio(
    file: UploadFile,
    user: Annotated[str, Depends(current_user)],
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix and suffix not in _ALLOWED_SUFFIXES:
        # Don't hard-fail unknown suffixes — ffmpeg may still handle them. Just
        # log so weird inputs are visible.
        logger.info("upload: unfamiliar suffix=%r (still trying)", suffix)
    suffix = suffix or ".bin"

    session_id = uuid4()
    path = settings.storage_dir / f"{session_id}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)

    bytes_written = 0
    with path.open("wb") as f:
        while True:
            chunk = await file.read(64 * 1024)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > _MAX_BYTES:
                f.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, f"file exceeds {_MAX_BYTES} bytes")
            f.write(chunk)

    if bytes_written == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "empty upload")

    now = datetime.now(timezone.utc)
    with Session(engine) as db:
        db.add(
            AudioSession(
                id=session_id,
                user_id=user,
                audio_path=str(path),
                codec=suffix.lstrip("."),
                started_at=now,
                ended_at=now,
                bytes_written=bytes_written,
                status=SessionStatus.pending_stt,
            )
        )
        db.commit()

    logger.info(
        "upload: session=%s user=%s file=%s bytes=%d",
        session_id,
        user,
        file.filename,
        bytes_written,
    )
    return {
        "session_id": str(session_id),
        "status": SessionStatus.pending_stt.value,
        "bytes": bytes_written,
        "path": str(path),
    }
