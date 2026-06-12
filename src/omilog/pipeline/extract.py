"""Build the LLM extraction prompt and parse its structured output.

The model is asked for a single JSON object matching the schema below. We
favour conservative extraction (false-positives are worse than misses) — the
spec explicitly calls out "let's grab lunch sometime" as a forbidden fake
calendar event.

Conversations are mostly French, sometimes mixed with English. The system
prompt is English (better tool-following) but explicitly notes the input
language so the model writes the title/summary in the dominant language.

The schema is built dynamically from a set of enabled categories so users
who don't care about, say, calendar events can disable that section and
save the output tokens it would otherwise consume.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("omilog.pipeline.extract")  # noqa: E402


# Default flags when no override is passed in. Mirrors the settings keys
# (extract_*) used in production. Kept here so tests and ad-hoc callers
# can build a prompt without dragging settings in.
DEFAULT_ENABLED: dict[str, bool] = {
    "calendar_events": True,
    "action_items": True,
    "decisions": True,
    "people_mentioned": True,
    "topics": True,
}


# Per-category schema fragments. Each one is the raw text that goes inside
# the top-level JSON object the LLM is asked to produce. Keeping them
# separate strings means we can drop entire categories from the prompt by
# omitting their fragment, without touching the surrounding scaffolding.
_SCHEMA_FRAGMENTS: dict[str, str] = {
    "calendar_events": (
        '  "calendar_events": [\n'
        '    {\n'
        '      "title": "string",\n'
        '      "starts_at": "ISO 8601 with timezone (e.g. 2026-06-15T14:00:00+02:00) or null if unclear",\n'
        '      "ends_at": "ISO 8601 with timezone or null",\n'
        '      "location": "string or null",\n'
        '      "attendees": ["string"],\n'
        '      "confidence": 0.0\n'
        '    }\n'
        '  ]'
    ),
    "action_items": (
        '  "action_items": [\n'
        '    {\n'
        '      "text": "string",\n'
        '      "owner": "user | other-person-name | null",\n'
        '      "due_at": "ISO 8601 with timezone or null"\n'
        '    }\n'
        '  ]'
    ),
    "decisions": (
        '  "decisions": [\n'
        '    {\n'
        '      "text": "string — the concrete decision, choice, or commitment expressed (e.g. \\"we will use Postgres\\", \\"on choisit l\'option B\\")",\n'
        '      "made_by": "user | other-person-name | null",\n'
        '      "confidence": 0.0\n'
        '    }\n'
        '  ]'
    ),
    "people_mentioned": (
        '  "people_mentioned": [\n'
        '    {"name": "string", "context": "brief description, max 100 chars"}\n'
        '  ]'
    ),
    "topics": '  "topics": ["string"]',
}


def _build_schema(enabled: dict[str, bool]) -> str:
    """Assemble the JSON schema block from enabled category fragments.

    Always includes title / summary / quality_score / quality_reasoning since
    those are core (not category-scoped). Categories are appended in the
    fixed order they appear in DEFAULT_ENABLED so the prompt is stable across
    config changes.
    """
    parts: list[str] = [
        '  "title": "string, <= 80 chars, in the conversation\'s dominant language"',
        '  "summary": "string, 2-4 sentences, in the dominant language"',
        '  "quality_score": 0.0',
        '  "quality_reasoning": "string, one short sentence"',
    ]
    for key in DEFAULT_ENABLED:
        if enabled.get(key, False):
            parts.append(_SCHEMA_FRAGMENTS[key])
    return "{\n" + ",\n".join(parts) + "\n}"


_PROMPT_HEADER = """\
/no_think

You analyze conversation transcripts captured passively from a wearable microphone worn by the user. {language_hint}Transcripts may include code-switching between languages.

Speakers may be labeled. When labels are present:
- [USER] is the wearer of the necklace.
- [S1], [S2], … are other speakers, anonymous unless named in the transcript.
- For action items: prefer owner="user" when [USER] is the one committing, owner="<name>" when a named other person commits, owner=null otherwise.
- For decisions: same attribution rules (made_by="user" / "<name>" / null).
- For calendar events: only the wearer's intentions and commitments are firm; unattributed mentions are lower-confidence.
- If labels are absent (older captures), use your best judgment from the dialogue.

Be CONSERVATIVE. Only extract things that are clearly stated. Do not invent details, names, or times. False positives are worse than missing real items — vague phrases like "we should meet sometime", "on devrait se voir bientôt", or "let's grab lunch sometime" are NOT calendar events; an unspecific "I should call my mom" is NOT an action item; "we should probably think about that" is NOT a decision.

Decisions vs action items: a DECISION is a conclusion or choice that was settled (architecture, plan, preference, opinion). An ACTION ITEM is a specific task with an owner that can be checked off later. When something qualifies as both ("we decided I'll send the deck Friday"), prefer the action_item — it's the more specific category. A pure preference choice ("we'll use Postgres", "j'arrête le café") is a decision and not an action.

Output STRICT JSON matching the schema below. No prose, no markdown fences, no commentary, no <think> tags. Just the JSON object.

Schema:
"""


_PROMPT_FOOTER = """

quality_score is a 0.0 to 1.0 judgment of how substantive the conversation was. Use these anchors:
- 0.0: pure noise (background TV transcribed, ambient sound, untargeted mumbling, single-word fragments, transcript that doesn't reflect a real interaction)
- 0.2: real-but-trivial (greetings, brief logistics, "ok bye", weather chatter)
- 0.5: ordinary daily conversation, some content but nothing memorable
- 0.7: clear conversation with concrete content worth remembering (a plan made, news shared, a real decision, a meaningful exchange)
- 1.0: substantive multi-party discussion with decisions made, important personal news, or memorable content

Be conservative. When in doubt, pick the lower of two adjacent anchors. A transcript with no real participants (likely captured ambient audio) is always 0.0. A transcript dominated by one-sided fragmentary text is at most 0.2.

quality_reasoning is one short sentence explaining the score, shown to the user. Examples: "Brief logistics about picking up groceries.", "Multi-person discussion of project timeline with concrete next steps.", "Single-speaker fragmentary text, likely ambient TV.".

When resolving relative time expressions ("tomorrow", "demain", "next week", "vendredi", "ce soir"), use the Date and timezone given in the user message. If the time is ambiguous in 12h-vs-24h terms (e.g. "at 7" without AM/PM), pick the more plausible interpretation given typical context (evening meals around 19h, morning meetings around 7am) and reflect uncertainty in confidence.

If the transcript is trivial small talk with nothing extractable, return arrays empty and a one-sentence summary."""


def render_default_system_prompt(
    primary_language: str = "",
    *,
    enabled: dict[str, bool] | None = None,
) -> str:
    """Render the built-in default prompt.

    ``primary_language``: optional 'most often in X' hint. Empty / 'any' /
    'auto' / 'none' all collapse to the language-neutral version.

    ``enabled``: optional per-category dict. Categories absent or False are
    omitted from the schema entirely — the LLM doesn't see them and can't
    waste tokens generating empty arrays for them. Defaults to all-on.
    """
    hint = (primary_language or "").strip()
    if hint and hint.lower() not in ("any", "auto", "none"):
        language_clause = f"Conversations are most often in {hint}. "
    else:
        language_clause = ""
    header = _PROMPT_HEADER.replace("{language_hint}", language_clause)
    flags = enabled if enabled is not None else DEFAULT_ENABLED
    return header + _build_schema(flags) + _PROMPT_FOOTER


def build_system_prompt(
    primary_language: str = "",
    override_path = None,  # noqa: ANN001 — pathlib.Path | None, kept untyped to avoid imports
    *,
    enabled: dict[str, bool] | None = None,
) -> str:
    """Return the system prompt to use for an extraction call.

    - If ``override_path`` is set and the file exists, return its contents
      verbatim. The language hint and enabled-categories flags are ignored —
      the user controls the whole prompt.
    - Otherwise render the default with the language hint substituted and
      the dynamic schema built from the enabled set.
    """
    if override_path is not None:
        try:
            if override_path.exists():
                contents = override_path.read_text(encoding="utf-8").strip()
                if contents:
                    return contents
        except OSError:
            pass  # fall through to default
    return render_default_system_prompt(primary_language, enabled=enabled)


# Pre-rendered with the empty hint for backward compatibility (tests that
# import SYSTEM_PROMPT directly still work) and as the default behavior when
# no override is configured.
SYSTEM_PROMPT = render_default_system_prompt("")


@dataclass
class Extraction:
    title: str | None
    summary: str | None
    topics: list[str] = field(default_factory=list)
    calendar_events: list[dict[str, Any]] = field(default_factory=list)
    action_items: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    people_mentioned: list[dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    # True when json_repair had to step in — usually means max_tokens truncation
    # and the extraction may be partial. Surfaced in the UI so the user knows.
    was_repaired: bool = False
    # LLM's self-assessment of how substantive the conversation was. None means
    # the model didn't include the field (older transcripts, prompt override
    # that doesn't ask for it, or a parse where the field was malformed). The
    # caller falls back to a stored default in that case.
    quality_score: float | None = None
    quality_reasoning: str | None = None


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
    primary_language: str = "",
    system_prompt_override_path = None,  # noqa: ANN001 — pathlib.Path | None
    enabled: dict[str, bool] | None = None,
) -> list[dict[str, str]]:
    body = _format_segments(transcript_segments or []) or transcript_text[:_MAX_TRANSCRIPT_CHARS]
    user_msg = (
        f"Date: {now.strftime('%Y-%m-%d %H:%M')} ({timezone_label}).\n\n"
        f"Transcript:\n{body}\n"
    )
    return [
        {
            "role": "system",
            "content": build_system_prompt(
                primary_language, system_prompt_override_path, enabled=enabled
            ),
        },
        {"role": "user", "content": user_msg},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Parser — tolerant of common LLM output quirks
# ──────────────────────────────────────────────────────────────────────────────

def _strip_think_block(text: str) -> str:
    # Qwen setups with reasoning enabled emit <think>...</think> before the
    # answer (qwen-3.6 uses the same tags as Qwen3). Drop everything up to
    # the last </think>. An OPENED-but-never-closed block means generation
    # was truncated mid-reasoning (max_tokens ran out before the answer
    # started) — there is no answer, so drop the reasoning rather than let
    # the parser treat it as one: reasoning prose often contains draft JSON
    # that would parse "successfully" with wrong content.
    end = text.rfind("</think>")
    if end >= 0:
        return text[end + len("</think>") :]
    start = text.find("<think>")
    if start >= 0:
        return text[:start]
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
        decisions=[
            d for d in (obj.get("decisions") or []) if isinstance(d, dict)
        ],
        people_mentioned=[
            p for p in (obj.get("people_mentioned") or []) if isinstance(p, dict)
        ],
        raw_text=text,
        was_repaired=was_repaired,
        quality_score=_clamped_float(obj.get("quality_score")),
        quality_reasoning=_string_or_none(obj.get("quality_reasoning")),
    )


def _clamped_float(v: Any) -> float | None:
    """Parse a 0..1 float from the LLM output, tolerant of strings and
    out-of-range numbers. Returns None when nothing usable is there."""
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return max(0.0, min(1.0, x))


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
