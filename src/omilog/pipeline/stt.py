"""whisper.cpp HTTP client.

The server (see docs/whisper-server.md) speaks one endpoint:
  POST {STT_BASE_URL}{STT_INFERENCE_PATH}
  multipart fields: file=<wav>, language=<code|auto>, response_format=verbose_json
  → JSON { "text": str, "segments": [...], "language": str }

Single-flight by design: whisper-server processes one request at a time. We
don't try to parallelise here — the runner is sequential.
"""

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("omilog.pipeline.stt")


@dataclass
class STTResult:
    text: str
    segments: list[dict[str, Any]]
    language: str | None
    raw: dict[str, Any]


class STTError(RuntimeError):
    pass


async def transcribe_wav(
    wav_bytes: bytes,
    *,
    base_url: str,
    inference_path: str = "/inference",
    language: str = "auto",
    timeout_s: float = 120.0,
) -> STTResult:
    if not base_url:
        raise STTError("STT_BASE_URL not configured")
    url = base_url.rstrip("/") + inference_path
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {"language": language, "response_format": "verbose_json"}
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        try:
            r = await client.post(url, files=files, data=data)
        except httpx.HTTPError as e:
            raise STTError(f"whisper.cpp request failed: {e}") from e
    if r.status_code != 200:
        snippet = r.text[:300].replace("\n", " ")
        raise STTError(f"whisper.cpp status={r.status_code} body={snippet!r}")
    try:
        payload = r.json()
    except ValueError as e:
        raise STTError(f"whisper.cpp returned non-JSON: {r.text[:200]!r}") from e

    text = (payload.get("text") or "").strip()
    segments = payload.get("segments") or []
    language_detected = payload.get("language") or payload.get("detected_language")
    if not text:
        # Server returned something but no transcript — count as a real error so
        # we don't silently store empty rows.
        raise STTError(f"whisper.cpp returned no text: {payload!r}")
    return STTResult(
        text=text,
        segments=segments,
        language=language_detected,
        raw=payload,
    )
