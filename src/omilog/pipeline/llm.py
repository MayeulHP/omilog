"""llama.cpp HTTP client (OpenAI-compatible /v1/chat/completions).

We use `response_format={"type": "json_object"}` to coerce JSON output.
llama-server respects this and constrains generation accordingly. We do NOT
rely on the JSON Schema variant — it's newer and not universally supported.
The schema is enforced via the system prompt + a tolerant parser.

Qwen3 has "thinking mode" (model emits a <think>...</think> block before its
real answer). For structured extraction we don't want that — see extract.py
where `/no_think` is prepended to the system message.
"""

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger("omilog.pipeline.llm")


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str
    finish_reason: str | None
    raw: dict[str, Any]


async def chat_json(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 2048,
    timeout_s: float = 180.0,
) -> ChatResult:
    if not base_url:
        raise LLMError("LLM_BASE_URL not configured")
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        try:
            r = await client.post(url, json=body)
        except httpx.HTTPError as e:
            raise LLMError(f"llama-server request failed: {e}") from e
    if r.status_code != 200:
        snippet = r.text[:300].replace("\n", " ")
        raise LLMError(f"llama-server status={r.status_code} body={snippet!r}")
    try:
        payload = r.json()
    except ValueError as e:
        raise LLMError(f"llama-server returned non-JSON: {r.text[:200]!r}") from e

    choices = payload.get("choices") or []
    if not choices:
        raise LLMError(f"no choices in response: {payload!r}")
    msg = choices[0].get("message") or {}
    text = msg.get("content") or ""
    finish = choices[0].get("finish_reason")
    if not text.strip():
        # Surface finish_reason — "length" means we hit max_tokens (likely
        # the model burnt the whole budget on internal thinking and didn't
        # emit any actual output). "stop" with no content is a model quirk
        # but the fix is the same: more headroom or a less ambitious prompt.
        raise LLMError(
            f"llama-server returned empty content (finish_reason={finish})"
        )
    return ChatResult(
        text=text,
        finish_reason=finish,
        raw=payload,
    )
