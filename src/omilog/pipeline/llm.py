"""llama.cpp HTTP client (OpenAI-compatible /v1/chat/completions).

We use `response_format={"type": "json_object"}` to coerce JSON output.
llama-server respects this and constrains generation accordingly. We do NOT
rely on the JSON Schema variant — it's newer and not universally supported.
The schema is enforced via the system prompt + a tolerant parser.

Qwen has "thinking mode" (model emits a <think>...</think> block before its
real answer). For structured extraction we don't want that. Two layers of
defense, because the server is shared and runs with thinking enabled:

- `disable_thinking=True` sends `chat_template_kwargs: {"enable_thinking":
  false}`, which llama.cpp applies per-request over the server's template
  defaults — reasoning is skipped entirely for our calls. (The legacy
  `/no_think` soft switch is also still prepended in extract.py.)
- If thinking happens anyway (flag off, or a template that ignores the
  kwarg), reasoning tokens count toward max_tokens before any answer
  appears — callers pass a budget sized for think-block + answer, and
  extract.py's parser strips think tags, including unclosed ones from
  mid-reasoning truncation.

Caveat observed with qwen-3.6 on llama.cpp: with enable_thinking=false the
`response_format` JSON grammar may not engage and the model can wrap its
answer in ```json fences — the tolerant parser in extract.py handles that.
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
    disable_thinking: bool = False,
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
    if disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
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
