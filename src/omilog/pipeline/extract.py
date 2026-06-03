"""Build the LLM extraction prompt and parse its structured output.

The model is asked for a single JSON object matching the schema below. We
favour conservative extraction (false-positives are worse than misses) — the
spec explicitly calls out "let's grab lunch sometime" as a forbidden fake
calendar event.

Conversations are mostly French, sometimes mixed with English. The system
prompt is English (better tool-following) but explicitly notes the input
language so the model writes the title/summary in the dominant language.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("omilog.pipeline.extract")  # noqa: E402


SYSTEM_PROMPT = """\
/no_think

You analyze conversation transcripts captured passively from a wearable microphone worn by the user. Conversations are most often in French, sometimes mixed with English.

Speakers may be labeled. When labels are present:
- [USER] is the wearer of the necklace.
- [S1], [S2], … are other speakers, anonymous unless named in the transcript.
- For action items: prefer owner="user" when [USER] is the one committing, owner="<name>" when a named other person commits, owner=null otherwise.
- For calendar events: only the wearer's intentions and commitments are firm; unattributed mentions are lower-confidence.
- If labels are absent (older captures), use your best judgment from the dialogue.

Be CONSERVATIVE. Only extract things that are clearly stated. Do not invent details, names, or times. False positives are worse than missing real items — "on devrait se voir bientôt" or "let's grab lunch sometime" is NOT a calendar event; an unspecific "I should call my mom" is NOT an action item.

Output STRICT JSON matching the schema below. No prose, no markdown fences, no commentary, no <think> tags. Just the JSON object.

Schema:
{
  "title": "string, <= 80 chars, in the conversation's dominant language",
  "summary": "string, 2-4 sentences, in the dominant language",
  "calendar_events": [
    {
      "title": "string",
      "starts_at": "ISO 8601 with timezone (e.g. 2026-06-15T14:00:00+02:00) or null if unclear",
      "ends_at": "ISO 8601 with timezone or null",
      "location": "string or null",
      "attendees": ["string"],
      "confidence": 0.0
    }
  ],
  "action_items": [
    {
      "text": "string",
      "owner": "user | other-person-name | null",
      "due_at": "ISO 8601 with timezone or null"
    }
  ],
  "people_mentioned": [
    {"name": "string", "context": "brief description, max 100 chars"}
  ],
  "topics": ["string"]
}

When resolving relative time expressions ("demain", "ce soir", "next week", "vendredi"), use the Date and timezone given in the user message. If the time is ambiguous (e.g. "à 7h" without AM/PM in a French context where 19h is more likely), pick the more plausible interpretation and reflect that in confidence.

If the transcript is trivial small talk with nothing extractable, return arrays empty and a one-sentence summary."""


@dataclass
class Extraction:
    title: str | None
    summary: str | None
    topics: list[str] = field(default_factory=list)
    calendar_events: list[dict[str, Any]] = field(default_factory=list)
    action_items: list[dict[str, Any]] = field(default_factory=list)
    people_mentioned: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    # True when json_repair had to step in — usually means max_tokens truncation
    # and the extraction may be partial. Surfaced in the UI so the user knows.
    was_repaired: bool = False


# Soft cap on input we hand the model. Qwen3 has plenty of context, but
# extraction quality drops on very long transcripts. ~6k tokens worth.
_MAX_TRANSCRIPT_CHARS = 24_000


def _format_segments(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    used = 0
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0) or 0)
        mm = int(start // 60)
        ss = int(start % 60)
        speaker = seg.get("speaker")
        if speaker:
            line = f"[{mm:02d}:{ss:02d}] [{speaker}] {text}"
        else:
            line = f"[{mm:02d}:{ss:02d}] {text}"
        if used + len(line) + 1 > _MAX_TRANSCRIPT_CHARS:
            lines.append("... (truncated)")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def build_messages(
    *,
    transcript_text: str,
    transcript_segments: list[dict[str, Any]] | None,
    now: datetime,
    timezone_label: str,
) -> list[dict[str, str]]:
    body = _format_segments(transcript_segments or []) or transcript_text[:_MAX_TRANSCRIPT_CHARS]
    user_msg = (
        f"Date: {now.strftime('%Y-%m-%d %H:%M')} ({timezone_label}).\n\n"
        f"Transcript:\n{body}\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Parser — tolerant of common LLM output quirks
# ──────────────────────────────────────────────────────────────────────────────

def _strip_think_block(text: str) -> str:
    # Some Qwen3 setups emit <think>...</think> even when asked not to.
    # Drop everything up to the last </think>.
    end = text.rfind("</think>")
    if end >= 0:
        return text[end + len("</think>") :]
    return text


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if not s.startswith("```"):
        return s
    nl = s.find("\n")
    if nl < 0:
        return s
    s = s[nl + 1 :]
    if s.rstrip().endswith("```"):
        s = s.rstrip()[: -3]
    return s.strip()


def _extract_first_json_object(text: str) -> str:
    """Find the outermost {...} block. Cheap recovery if the model wrapped its
    JSON in extra prose despite the system prompt."""
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def parse(text: str) -> Extraction:
    cleaned = _strip_think_block(text)
    cleaned = _strip_code_fences(cleaned)
    cleaned = _extract_first_json_object(cleaned)

    obj: dict | None = None
    primary_error: Exception | None = None
    was_repaired = False
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        primary_error = e
        # Fallback: json_repair handles common LLM JSON breakage — truncation
        # at max_tokens, hanging strings/arrays, unbalanced braces, smart
        # quotes etc. Better to recover whatever was complete than throw away
        # the whole extraction over a missing brace.
        try:
            import json_repair  # type: ignore[import-untyped]

            repaired = json_repair.loads(cleaned)
            if isinstance(repaired, dict):
                obj = repaired
                was_repaired = True
                logger.warning(
                    "LLM output had invalid JSON (%s); recovered via json_repair. "
                    "Likely hit max_tokens — consider bumping OMILOG_LLM_MAX_TOKENS.",
                    e,
                )
        except Exception as repair_error:  # noqa: BLE001
            logger.debug("json_repair also failed: %s", repair_error)

    if obj is None:
        raise ValueError(
            f"LLM output is not valid JSON ({primary_error}); "
            f"preview={cleaned[:200]!r}"
        ) from primary_error
    if not isinstance(obj, dict):
        raise ValueError(
            f"LLM output is not a JSON object: type={type(obj).__name__}"
        )

    return Extraction(
        title=_string_or_none(obj.get("title")),
        summary=_string_or_none(obj.get("summary")),
        topics=[s for s in (obj.get("topics") or []) if isinstance(s, str)],
        calendar_events=[
            e for e in (obj.get("calendar_events") or []) if isinstance(e, dict)
        ],
        action_items=[
            a for a in (obj.get("action_items") or []) if isinstance(a, dict)
        ],
        people_mentioned=[
            p for p in (obj.get("people_mentioned") or []) if isinstance(p, dict)
        ],
        raw_text=text,
        was_repaired=was_repaired,
    )


def _string_or_none(v: Any) -> str | None:
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def parse_iso8601(s: Any) -> datetime | None:
    """Tolerant ISO 8601 → datetime. Accepts 'Z' as +00:00. Returns None on
    anything we can't parse, so a slightly-malformed date doesn't fail the
    whole conversation."""
    if not isinstance(s, str) or not s.strip():
        return None
    candidate = s.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None
