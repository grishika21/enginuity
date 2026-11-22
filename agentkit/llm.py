"""A thin client for any OpenAI-compatible chat endpoint.

Defaults to a local Ollama model, so the demo runs offline. Point LLM_BASE_URL / LLM_MODEL /
LLM_API_KEY at OpenAI, Azure OpenAI (v1 endpoint), Groq or another provider to swap models
without code changes. LLM_FALLBACK_MODELS (comma-separated, same endpoint) takes over when the
main model is rate-limited, as free tiers are; a limited model is skipped until its limit resets.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from functools import lru_cache
from typing import Any

from openai import BadRequestError, OpenAI, RateLimitError

from .config import get_settings

log = logging.getLogger("agentkit.llm")

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
# Models like to typeset: "KC\u201150\u2011210" (non-breaking hyphens), "18\u202fNm" (narrow no-break
# space). Codes, part numbers and values are matched in code, so every reply is mapped back to
# plain ASCII hyphens and spaces before anyone reads it.
_TYPOGRAPHY = str.maketrans({**{c: "-" for c in "\u2010\u2011\u2012\u2013\u2212\ufe63\uff0d"},
                             **{c: " " for c in "\u00a0\u2007\u2009\u200a\u202f"}})


_CITATION = re.compile(r"\u3010(\d+)(?:\u2020[^\u3011]*)?\u3011")  # gpt-oss style "【1†L7】" -> "[1]"


def plain(value: Any) -> Any:
    """Normalise typographic hyphens and spaces, and citation markers, in a model reply (strings,
    inside dicts and lists)."""
    if isinstance(value, str):
        return _CITATION.sub(r"[\1]", value.translate(_TYPOGRAPHY))
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


class LLMError(RuntimeError):
    pass


class RateLimited(LLMError):
    """Every configured model is over its usage limit right now."""


BUSY = "The AI model has reached its usage limit for now. Please try again in a few minutes."
_COOLDOWN: dict[str, float] = {}  # model -> time its rate limit should be over, shared by all clients
_COOLDOWN_LOCK = threading.Lock()


SHORT_WAIT_S = 20.0  # a per-minute limit that clears this soon is waited out once, not reported


def _retry_after(exc: RateLimitError) -> float:
    """Seconds until the provider expects the limit to clear (120 when it doesn't say)."""
    try:
        return max(float(exc.response.headers.get("retry-after", "120")), 0.0)
    except (TypeError, ValueError):
        return 120.0


class LLM:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0,
                 reasoning_effort: str | None = None, fallback_models: list[str] | None = None):
        self.model = model
        self.models = [model, *[m for m in (fallback_models or []) if m and m != model]]
        # With a fallback, a rate limit moves straight on to the next model instead of retrying.
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout,
                              max_retries=0 if len(self.models) > 1 else 2)
        # Only reasoning models accept this (e.g. gpt-oss); leave it unset for others.
        self._extra: dict[str, Any] = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    def _create(self, **kwargs: Any):
        """One completion from the first model that is not rate-limited (limited ones go last, the
        one whose limit clears soonest first).

        When every model is limited and the shortest limit clears within SHORT_WAIT_S (a
        per-minute cap), wait that long and try once more instead of failing the request."""
        last: RateLimitError | None = None
        for attempt in range(2):
            now = time.time()
            ordered = sorted(self.models, key=lambda m: max(_COOLDOWN.get(m, 0.0), now))  # soonest free first
            waits = []
            for model in ordered:
                try:
                    return self._client.chat.completions.create(model=model, **kwargs, **self._extra)
                except RateLimitError as exc:
                    wait = _retry_after(exc)
                    waits.append(wait)
                    with _COOLDOWN_LOCK:  # skip this model until then (at least 30 s, at most 15 min)
                        _COOLDOWN[model] = time.time() + min(max(wait, 30.0), 900.0)
                    log.warning("model %s is rate-limited for about %.0f s", model, wait)
                    last = exc
            if attempt == 0 and min(waits) <= SHORT_WAIT_S:
                time.sleep(min(waits) + 0.5)
                continue
            break
        raise RateLimited(BUSY) from last

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int | None = None) -> str:
        try:
            resp = self._create(messages=messages, temperature=temperature, max_tokens=max_tokens)
        except LLMError:
            raise
        except Exception as exc:  # the provider's own error text can hold URLs; keep it in the log
            log.warning("LLM call failed: %s", exc)
            raise LLMError("The language model is not reachable right now.") from exc
        return plain(_THINK_RE.sub("", resp.choices[0].message.content or "").strip())

    def json(self, messages: list[dict[str, Any]], temperature: float = 0.0) -> dict[str, Any]:
        """Ask for a JSON object and parse it, tolerating fences or text around it."""
        try:
            resp = self._create(messages=messages, temperature=temperature, response_format={"type": "json_object"})
            text = resp.choices[0].message.content or ""
        except LLMError:
            raise
        except Exception as exc:
            log.warning("LLM JSON call failed: %s", exc)
            raise LLMError("The language model is not reachable right now.") from exc
        return plain(parse_json_object(_THINK_RE.sub("", text)))

    def tool_call(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any], str]:
        """Return (tool_name, arguments, text). tool_name is None when the model answered in text."""
        try:
            resp = self._create(messages=messages, tools=tools, tool_choice="auto", temperature=0.0)
        except BadRequestError as exc:
            # Some providers reject a call to a tool name the model made up, but return what it tried;
            # hand that back so the caller can map it to a real tool or refuse it.
            failed = _failed_generation(exc.body)
            if failed:
                try:
                    attempt = parse_json_object(failed)
                    args = attempt.get("arguments") or attempt.get("parameters") or {}
                    return str(attempt.get("name", "")), plain(args) if isinstance(args, dict) else {}, ""
                except LLMError:
                    pass
            log.warning("LLM tool call rejected: %s", exc)
            raise LLMError("The language model could not handle that request.") from exc
        except LLMError:
            raise
        except Exception as exc:
            log.warning("LLM tool call failed: %s", exc)
            raise LLMError("The language model is not reachable right now.") from exc
        msg = resp.choices[0].message
        text = plain(_THINK_RE.sub("", msg.content or "").strip())
        if msg.tool_calls:
            call = msg.tool_calls[0]
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = parse_json_object(call.function.arguments or "")
            return call.function.name, plain(args) if isinstance(args, dict) else {}, text
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
    fallbacks = [m.strip() for m in os.environ.get("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
    return LLM(s.llm_base_url, s.llm_api_key, s.llm_model, reasoning_effort=os.environ.get("LLM_REASONING_EFFORT") or None,
               fallback_models=fallbacks)
