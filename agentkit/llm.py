"""A thin client for any OpenAI-compatible chat endpoint.

Defaults to a local Ollama model, so the demo runs offline. Point LLM_BASE_URL / LLM_MODEL /
LLM_API_KEY at OpenAI, Azure OpenAI (v1 endpoint), Groq or another provider to swap models
without code changes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

from openai import BadRequestError, OpenAI

from .config import get_settings

log = logging.getLogger("agentkit.llm")

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0,
                 reasoning_effort: str | None = None):
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
        # Only reasoning models accept this (e.g. gpt-oss); leave it unset for others.
        self._extra: dict[str, Any] = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int | None = None) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, **self._extra
            )
        except Exception as exc:  # the provider's own error text can hold URLs; keep it in the log
            log.warning("LLM call failed: %s", exc)
            raise LLMError("The language model is not reachable right now.") from exc
        return _THINK_RE.sub("", resp.choices[0].message.content or "").strip()

    def json(self, messages: list[dict[str, Any]], temperature: float = 0.0) -> dict[str, Any]:
        """Ask for a JSON object and parse it, tolerating fences or text around it."""
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                **self._extra,
            )
            text = resp.choices[0].message.content or ""
        except Exception as exc:
            log.warning("LLM JSON call failed: %s", exc)
            raise LLMError("The language model is not reachable right now.") from exc
        return parse_json_object(_THINK_RE.sub("", text))

    def tool_call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any], str]:
        """Return (tool_name, arguments, text). tool_name is None when the model answered in text."""
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, tool_choice="auto", temperature=0.0, **self._extra
            )
        except BadRequestError as exc:
            # Some providers reject a call to a tool name the model made up, but return what it tried;
            # hand that back so the caller can map it to a real tool or refuse it.
            failed = _failed_generation(exc.body)
            if failed:
                try:
                    attempt = parse_json_object(failed)
                    args = attempt.get("arguments") or attempt.get("parameters") or {}
                    return str(attempt.get("name", "")), args if isinstance(args, dict) else {}, ""
                except LLMError:
                    pass
            log.warning("LLM tool call rejected: %s", exc)
            raise LLMError("The language model could not handle that request.") from exc
        except Exception as exc:
            log.warning("LLM tool call failed: %s", exc)
            raise LLMError("The language model is not reachable right now.") from exc
        msg = resp.choices[0].message
        text = _THINK_RE.sub("", msg.content or "").strip()
        if msg.tool_calls:
            call = msg.tool_calls[0]
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = parse_json_object(call.function.arguments or "")
            return call.function.name, args if isinstance(args, dict) else {}, text
        return None, {}, text


def _failed_generation(body: Any) -> str | None:
    """The rejected tool call a provider returns with a 400 (the SDK may or may not unwrap 'error')."""
    if not isinstance(body, dict):
        return None
    inner = body.get("error") if isinstance(body.get("error"), dict) else body
    value = inner.get("failed_generation")
    return value if isinstance(value, str) and value.strip() else None


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise LLMError("The language model returned an answer I could not read.")


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    s = get_settings()
    return LLM(s.llm_base_url, s.llm_api_key, s.llm_model, reasoning_effort=os.environ.get("LLM_REASONING_EFFORT") or None)
