"""Audio preprocessing for STT.

whisper.cpp's HTTP server takes WAV (16-bit signed PCM). Our captures are
Ogg-Opus from the BLE necklace, so we shell out to ffmpeg to transcode +
resample to 16 kHz mono before posting. Doing this as an async subprocess
keeps the runner event loop unblocked while ffmpeg works.
"""

import asyncio
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("omilog.pipeline.audio")

WHISPER_SAMPLE_RATE = 16000


class FFmpegMissing(RuntimeError):
    pass


class TranscodeError(RuntimeError):
    pass


def assert_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise FFmpegMissing(
            "ffmpeg not found on PATH — install it (e.g. `sudo apt install ffmpeg`)."
        )


async def transcode_to_wav_bytes(
    src: Path,
    *,
    sample_rate: int = WHISPER_SAMPLE_RATE,
    channels: int = 1,
    timeout_s: float = 30.0,
) -> bytes:
    """Decode + resample any audio file to WAV bytes whisper.cpp will accept."""
    assert_ffmpeg_available()
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(src),
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-f", "wav",
        "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError as e:
        proc.kill()
        await proc.wait()
        raise TranscodeError(f"ffmpeg timed out after {timeout_s}s") from e

    if proc.returncode != 0:
        raise TranscodeError(
            f"ffmpeg exit={proc.returncode}: {stderr.decode(errors='replace')[:500]}"
        )
    if not stdout:
        raise TranscodeError("ffmpeg returned empty WAV — input may be silent or corrupt")
    return stdout
