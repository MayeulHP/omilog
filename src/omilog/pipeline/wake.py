"""Wake-word action matching + execution.

After STT (and optional diarization), we scan the transcript for any of the
configured wake phrases. For each match, the configured shell command runs
with template variables substituted in a shell-safe way (shlex.quote, so
quotes / metacharacters in the transcript can't break out of the intended
shell context).

Failure modes are non-blocking: a misbehaving command, a timeout, or a
matcher quirk only affects the WakeInvocation row. The LLM extraction
pipeline runs to completion regardless.

Template variables exposed to the command:
  $transcript          The post-wake utterance text (from after the wake
                       phrase up to the next wake phrase or end of file).
  $transcript_full     The complete transcript text.
  $conversation_id     The Conversation UUID (so the action can call back
                       into our /api/ to read events / actions / etc.).
  $wake_phrase         The phrase that matched (useful when an action lists
                       several aliases).

Matcher: case-insensitive substring for v1. Whisper transcription is good
enough at common wake phrases that fuzzy matching isn't usually needed.
"""

import asyncio
import logging
import shlex
import time
from string import Template
from typing import Any

logger = logging.getLogger("omilog.pipeline.wake")


def find_wake_matches(
    text: str,
    phrases: list[str],
    stop_phrases: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Find each non-overlapping wake phrase occurrence in ``text``.

    Returns a list of dicts ordered by start index:

        [{"phrase": "Hey Jarvis",
          "start": 42,            # char index of the matched phrase
          "end": 52,
          "post_wake": "..."},    # text from end of match to whichever of
                                  #   {next wake match, stop phrase, EOF}
                                  #   comes first
         ...]

    `stop_phrases` is optional. When provided, any case-insensitive occurrence
    of one of them after a wake match shortens that match's `post_wake` — like
    radio "over", so a long tail of unrelated conversation doesn't all flow
    into the command's argument.
    """
    text_lower = text.lower()
    raw_hits: list[tuple[int, int, str]] = []
    for phrase in phrases:
        phrase_lower = phrase.lower().strip()
        if not phrase_lower:
            continue
        start = 0
        while True:
            idx = text_lower.find(phrase_lower, start)
            if idx < 0:
                break
            raw_hits.append((idx, idx + len(phrase_lower), phrase))
            start = idx + len(phrase_lower)

    raw_hits.sort()
    accepted: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, phrase in raw_hits:
        if start < last_end:
            continue
        accepted.append((start, end, phrase))
        last_end = end

    normalized_stops = [
        sp.lower().strip() for sp in (stop_phrases or []) if sp and sp.strip()
    ]

    matches: list[dict[str, Any]] = []
    for i, (start, end, phrase) in enumerate(accepted):
        next_match_start = accepted[i + 1][0] if i + 1 < len(accepted) else len(text)

        # Earliest stop-phrase occurrence after this wake match (if any).
        stop_cutoff: int | None = None
        for sp in normalized_stops:
            idx = text_lower.find(sp, end)
            if idx >= 0 and (stop_cutoff is None or idx < stop_cutoff):
                stop_cutoff = idx

        cutoff = next_match_start
        if stop_cutoff is not None and stop_cutoff < cutoff:
            cutoff = stop_cutoff

        post_wake = text[end:cutoff].strip()
        matches.append(
            {
                "phrase": phrase,
                "start": start,
                "end": end,
                "post_wake": post_wake,
            }
        )
    return matches


def resolve_command(template: str, variables: dict[str, str]) -> str:
    """Substitute ``$VAR`` in the command template with shell-safe values.

    Uses ``string.Template.safe_substitute`` so unknown variables stay literal
    instead of raising. Values are escaped with ``shlex.quote`` so a transcript
    containing quotes / pipes / dollar signs can't escape its intended
    argument position.
    """
    safe_vars = {k: shlex.quote(v) for k, v in variables.items()}
    return Template(template).safe_substitute(safe_vars)


async def execute_command(
    command: str, *, timeout_s: float = 30.0
) -> dict[str, Any]:
    """Run ``command`` via /bin/sh -c, capture output, enforce a timeout.

    Returns a dict ready to drop into a WakeInvocation row:

        {"exit_code": int | None,
         "stdout": str,          # capped at 4 KB
         "stderr": str,          # capped at 4 KB
         "duration_ms": int}
    """
    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        return {
            "exit_code": None,
            "stdout": "",
            "stderr": f"failed to spawn shell: {e}",
            "duration_ms": int((time.monotonic() - started) * 1000),
        }

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        # uvloop on Linux races: the process can exit between our timeout
        # firing and our kill() call, raising ProcessLookupError. Don't let
        # either kill() or wait() propagate — the user just wanted to know
        # we timed out, not how the runtime tore down the process.
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await proc.wait()
        except ProcessLookupError:
            pass
        return {
            "exit_code": None,
            "stdout": "",
            "stderr": f"timed out after {timeout_s:.0f}s",
            "duration_ms": int((time.monotonic() - started) * 1000),
        }

    def _decode_and_cap(b: bytes, limit: int = 4000) -> str:
        s = b.decode("utf-8", errors="replace")
        if len(s) > limit:
            s = s[:limit] + f"\n…(truncated, total {len(s)} chars)"
        return s

    return {
        "exit_code": proc.returncode,
        "stdout": _decode_and_cap(stdout_bytes),
        "stderr": _decode_and_cap(stderr_bytes),
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
