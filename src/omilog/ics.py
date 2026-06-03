"""iCalendar (RFC 5545) generation for extracted CalendarEvent rows.

Two consumers:
  - /calendar.ics?token=…  subscribable feed (the calendar app polls it)
  - /events/{id}/download.ics  one-off per-event download

We always emit UTC `Z`-suffixed times — the LLM stores ISO timestamps with TZ
offsets already, so we just convert. No VTIMEZONE block emitted; that keeps
the file simple and unambiguous.

UIDs are stable: `<event-uuid>@<domain>`. Calendar apps treat re-fetches with
the same UID as updates, so re-running extraction on a conversation doesn't
duplicate events on the user's calendar.
"""

from datetime import datetime, timedelta, timezone
from typing import Iterable

from .models import CalendarEvent


# ──────────────────────────────────────────────────────────────────────────────
# RFC 5545 primitives
# ──────────────────────────────────────────────────────────────────────────────

def _escape_text(value: str | None) -> str:
    """Escape per §3.3.11 TEXT: backslash, semicolon, comma, newline."""
    if value is None:
        return ""
    return (
        value.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\n")
    )


def _fold(line: str, max_octets: int = 75) -> str:
    """§3.1 line folding: lines > 75 octets get split with CRLF + single space.

    Splitting is by byte to respect the octet rule even for multibyte UTF-8.
    """
    raw = line.encode("utf-8")
    if len(raw) <= max_octets:
        return line
    out: list[bytes] = []
    while len(raw) > max_octets:
        # Don't split inside a UTF-8 multi-byte char: walk back from the cut
        # point until a continuation byte is past the boundary.
        cut = max_octets
        while cut > 0 and (raw[cut] & 0xC0) == 0x80:
            cut -= 1
        out.append(raw[:cut])
        raw = b" " + raw[cut:]
    out.append(raw)
    return "\r\n".join(part.decode("utf-8") for part in out)


def _fmt_utc(dt: datetime) -> str:
    """Format a datetime as RFC 5545 UTC (YYYYMMDDTHHMMSSZ)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


# ──────────────────────────────────────────────────────────────────────────────
# Event / calendar builders
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_DURATION_HOURS = 1


def build_vevent(
    event: CalendarEvent,
    *,
    conversation_id: str | None = None,
    conversation_title: str | None = None,
    domain: str = "omilog",
    now_utc: datetime | None = None,
) -> list[str]:
    """Return one VEVENT block as a list of folded lines (no CRLFs added yet)."""
    if event.starts_at is None:
        raise ValueError(f"event {event.id} has no starts_at; cannot emit ICS")

    now_utc = now_utc or datetime.now(timezone.utc)
    uid = f"{event.id}@{domain}"
    ends_at = event.ends_at or (event.starts_at + timedelta(hours=DEFAULT_DURATION_HOURS))

    # Description with provenance — easier to audit what came from where.
    desc_lines = []
    if event.description:
        desc_lines.append(event.description)
    desc_lines.append(f"confidence: {int(round(event.confidence * 100))}%")
    if conversation_title:
        desc_lines.append(f"from: {conversation_title}")
    if conversation_id:
        desc_lines.append(f"conversation: {conversation_id}")
    description = "\n".join(desc_lines)

    lines = [
        "BEGIN:VEVENT",
        _fold(f"UID:{uid}"),
        f"DTSTAMP:{_fmt_utc(now_utc)}",
        f"DTSTART:{_fmt_utc(event.starts_at)}",
        f"DTEND:{_fmt_utc(ends_at)}",
        _fold(f"SUMMARY:{_escape_text(event.title or '(untitled)')}"),
    ]
    if event.location:
        lines.append(_fold(f"LOCATION:{_escape_text(event.location)}"))
    if description:
        lines.append(_fold(f"DESCRIPTION:{_escape_text(description)}"))
    lines.append("END:VEVENT")
    return lines


def build_vcalendar(
    items: Iterable[tuple[CalendarEvent, str | None, str | None]],
    *,
    prodid: str = "-//omilog//EN",
    calname: str = "omilog",
    domain: str = "omilog",
    now_utc: datetime | None = None,
) -> str:
    """Serialise (event, conversation_id, conversation_title) triples into a
    full VCALENDAR document with proper CRLF endings.

    Events without a start time are skipped silently — we have nothing useful
    to put on a calendar for them.
    """
    out: list[str] = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        _fold(f"PRODID:{prodid}"),
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        _fold(f"X-WR-CALNAME:{_escape_text(calname)}"),
    ]
    for event, conv_id, conv_title in items:
        if event.starts_at is None:
            continue
        out.extend(
            build_vevent(
                event,
                conversation_id=conv_id,
                conversation_title=conv_title,
                domain=domain,
                now_utc=now_utc,
            )
        )
    out.append("END:VCALENDAR")
    return "\r\n".join(out) + "\r\n"


def build_single_event_calendar(
    event: CalendarEvent,
    *,
    conversation_id: str | None = None,
    conversation_title: str | None = None,
    prodid: str = "-//omilog//EN",
    calname: str = "omilog",
    domain: str = "omilog",
) -> str:
    """Convenience wrapper for the per-event download endpoint."""
    return build_vcalendar(
        [(event, conversation_id, conversation_title)],
        prodid=prodid,
        calname=calname,
        domain=domain,
    )
