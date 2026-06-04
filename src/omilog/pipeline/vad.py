"""Silence detection + conversation segmentation.

Uses ffmpeg's `silencedetect` filter rather than a Python ML model — pure CPU,
streams the whole file, no extra deps, and on speech-volume audio it's
indistinguishable from silero-vad for our purposes. We can swap in silero
later if we see misses on whispered/quiet speech.

The flow this module supports:
    raw long capture
        → find_silence_regions     (ffmpeg silencedetect parse)
        → segment_by_silence_gaps  (group at gap_threshold_s)
        → list of (conv_start_s, conv_end_s)
    then the runner extracts each conversation with ffmpeg trim+re-encode.
"""

import asyncio
import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger("omilog.pipeline.vad")


class VADError(RuntimeError):
    pass


# ──────────────────────────────────────────────────────────────────────────────
# ffmpeg silencedetect
# ──────────────────────────────────────────────────────────────────────────────

_RE_SILENCE_START = re.compile(r"silence_start:\s*([\-\d.]+)")
_RE_SILENCE_END = re.compile(r"silence_end:\s*([\-\d.]+)")
_RE_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):([\d.]+)")


def parse_silencedetect_output(text: str) -> list[tuple[float, float]]:
    """Pull (silence_start, silence_end) pairs out of ffmpeg's stderr.

    silencedetect prints one `silence_start: T` line, then a matching
    `silence_end: T | silence_duration: D` line. If the file ends in silence
    there will be a final silence_start with no end — we ignore it.
    """
    pending_start: float | None = None
    out: list[tuple[float, float]] = []
    for line in text.splitlines():
        m_start = _RE_SILENCE_START.search(line)
        if m_start:
            try:
                pending_start = float(m_start.group(1))
            except ValueError:
                pending_start = None
            continue
        m_end = _RE_SILENCE_END.search(line)
        if m_end and pending_start is not None:
            try:
                end = float(m_end.group(1))
                out.append((pending_start, end))
            except ValueError:
                pass
            pending_start = None
    return out


def parse_duration(text: str) -> float | None:
    m = _RE_DURATION.search(text)
    if not m:
        return None
    h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mn * 60 + s


async def analyse(
    src: Path,
    *,
    threshold_db: float,
    min_silence_s: float,
    timeout_s: float = 300.0,
) -> tuple[float, list[tuple[float, float]]]:
    """Return (duration_s, [(silence_start, silence_end), ...]).

    Raises VADError on ffmpeg failure or unparseable output.
    """
    if shutil.which("ffmpeg") is None:
        raise VADError("ffmpeg not found on PATH")
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-i", str(src),
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_silence_s}",
        "-f", "null", "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s
        )
    except asyncio.TimeoutError as e:
        proc.kill()
        await proc.wait()
        raise VADError(f"silencedetect timed out after {timeout_s}s") from e
    if proc.returncode != 0:
        raise VADError(
            f"ffmpeg silencedetect exit={proc.returncode}: "
            f"{stderr_bytes.decode(errors='replace')[:300]}"
        )
    stderr = stderr_bytes.decode(errors="replace")
    duration = parse_duration(stderr)
    if duration is None:
        raise VADError("could not parse duration from ffmpeg output")
    silences = parse_silencedetect_output(stderr)
    return duration, silences


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation
# ──────────────────────────────────────────────────────────────────────────────

def segment_by_silence_gaps(
    duration_s: float,
    silences: list[tuple[float, float]],
    *,
    gap_threshold_s: float,
    pad_s: float = 0.0,
) -> list[tuple[float, float]]:
    """Group conversation boundaries based on long silence regions.

    Silences shorter than `gap_threshold_s` are kept *inside* a conversation
    (natural pauses, turn-taking). Silences >= threshold mark conversation
    boundaries: the conversation ends just before the long silence and the
    next one starts just after.

    `pad_s` widens each conversation symmetrically so we don't clip the first
    or last word; clamped to [0, duration_s].

    Returns conversations in chronological order. Empty list means "all
    silence — drop this capture."
    """
    if duration_s <= 0:
        return []

    # Sort defensively.
    silences = sorted(silences)

    # Leading silence trim: only if the file STARTS with a silence that's also
    # long enough to be a conversation boundary. Shorter leading silences are
    # just "warm-up" inside the first conversation. The old version trimmed any
    # leading silence which could swallow real speech in low-volume captures.
    speech_start = 0.0
    if (
        silences
        and silences[0][0] <= 0.05
        and (silences[0][1] - silences[0][0]) >= gap_threshold_s
    ):
        speech_start = silences[0][1]

    # Trailing silence trim: same guard — only if the trailing silence is long
    # enough to count as its own conversation boundary. Without this, any
    # trailing low-volume speech got misclassified as "end of recording."
    speech_end = duration_s
    if silences:
        last = silences[-1]
        if (
            last[1] >= duration_s - 0.05
            and (last[1] - last[0]) >= gap_threshold_s
        ):
            speech_end = last[0]

    if speech_end <= speech_start:
        return []  # entirely silence

    # Long silences inside the [speech_start, speech_end] window become splits.
    splits: list[tuple[float, float]] = []
    for s, e in silences:
        if s <= speech_start or e >= speech_end:
            continue  # boundary trims, not interior gaps
        if e - s >= gap_threshold_s:
            splits.append((s, e))

    conversations: list[tuple[float, float]] = []
    cursor = speech_start
    for split_start, split_end in splits:
        if split_start > cursor:
            conversations.append((cursor, split_start))
        cursor = split_end
    if speech_end > cursor:
        conversations.append((cursor, speech_end))

    # Apply pad, clamped to [0, duration_s].
    padded: list[tuple[float, float]] = []
    for start, end in conversations:
        padded.append(
            (
                max(0.0, start - pad_s),
                min(duration_s, end + pad_s),
            )
        )
    return padded


# ──────────────────────────────────────────────────────────────────────────────
# Segment extraction (ffmpeg trim + re-encode to Opus)
# ──────────────────────────────────────────────────────────────────────────────

async def extract_segment_to_opus(
    src: Path,
    dst: Path,
    *,
    start_s: float,
    end_s: float,
    bitrate: str = "32k",
    timeout_s: float = 180.0,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise VADError("ffmpeg not found on PATH")
    if end_s <= start_s:
        raise VADError(f"empty segment: start={start_s} end={end_s}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # -ss before -i is fast-seek (less precise but cheap). Combined with -to
    # after, ffmpeg re-decodes the necessary frames for an accurate cut.
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-to", f"{end_s:.3f}",
        "-i", str(src),
        "-c:a", "libopus",
        "-b:a", bitrate,
        "-ar", "16000",
        "-ac", "1",
        str(dst),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s
        )
    except asyncio.TimeoutError as e:
        proc.kill()
        await proc.wait()
        raise VADError("ffmpeg segment extract timed out") from e
    if proc.returncode != 0:
        raise VADError(
            f"ffmpeg segment extract exit={proc.returncode}: "
            f"{stderr_bytes.decode(errors='replace')[:300]}"
        )
    if not dst.exists() or dst.stat().st_size == 0:
        raise VADError(f"ffmpeg produced empty segment file: {dst}")
