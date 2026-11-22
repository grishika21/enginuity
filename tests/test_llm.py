import httpx
import pytest
from openai import BadRequestError, RateLimitError

from agentkit import llm as llm_module
from agentkit.llm import LLM, LLMError, RateLimited, parse_json_object

FAILED = ('{"name": "authors_a_new_document", "arguments": {"request": "Write a procedure"}}')


def rejected(body):
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"error": body})
    return BadRequestError("tool_use_failed", response=response, body=body)


class FakeCompletions:
    def __init__(self, exc):
        self.exc = exc

    def create(self, **_kwargs):
        raise self.exc


class FakeClient:
    def __init__(self, exc):
        self.chat = type("Chat", (), {"completions": FakeCompletions(exc)})()


@pytest.mark.parametrize("body", [
    {"message": "x", "code": "tool_use_failed", "failed_generation": FAILED},             # as the SDK unwraps it
    {"error": {"message": "x", "code": "tool_use_failed", "failed_generation": FAILED}},  # as sent on the wire
])
def test_a_rejected_made_up_tool_call_is_handed_back(body):
    llm = LLM("http://localhost", "k", "m")
    llm._client = FakeClient(rejected(body))
    name, args, text = llm.tool_call([], [])
    assert name == "authors_a_new_document" and args == {"request": "Write a procedure"}


def test_other_bad_requests_raise_a_safe_error():
    llm = LLM("http://localhost", "k", "m")
    llm._client = FakeClient(rejected({"message": "context too long"}))
    with pytest.raises(LLMError):
        llm.tool_call([], [])


def test_parse_json_object_tolerates_text_around_it():
    assert parse_json_object('Sure: ```json\n{"skill": "x"}\n```') == {"skill": "x"}
    with pytest.raises(LLMError):
        parse_json_object("no json here")


def test_typeset_hyphens_and_spaces_become_plain_so_codes_still_match():
    from agentkit.llm import plain
    from agents.docs_agent import tools

    text = plain("Replace the filter (KC\u201150\u2011210) and torque to 18\u202fNm.")
    assert text == "Replace the filter (KC-50-210) and torque to 18 Nm."
    assert tools.check_parts(text, "KC-50")[1] == ["KC-50-210"]
    assert plain({"sections": [{"title": "4.1 Seal\u00a0replacement"}], "n": 3}) == \
        {"sections": [{"title": "4.1 Seal replacement"}], "n": 3}


def limited(seconds="600"):
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={"retry-after": seconds})
    return RateLimitError("rate limited", response=response, body=None)


class ScriptedCompletions:
    """Per model: raise a rate limit (always, or the first n times with a short wait), or answer."""

    def __init__(self, limited_models, short_limits=None):
        self.limited_models, self.calls = set(limited_models), []
        self.short_limits = dict(short_limits or {})

    def create(self, model, **_kwargs):
        self.calls.append(model)
        if model in self.limited_models:
            raise limited()
        if self.short_limits.get(model):
            self.short_limits[model] -= 1
            raise limited("3")
        message = type("M", (), {"content": f"answer from {model}", "tool_calls": None})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


@pytest.fixture
def no_cooldown():
    llm_module._COOLDOWN.clear()
    yield
    llm_module._COOLDOWN.clear()


def scripted(limited_models, fallbacks=("backup",), short_limits=None):
    llm = LLM("http://localhost", "k", "main", fallback_models=list(fallbacks))
    api = ScriptedCompletions(limited_models, short_limits)
    llm._client = type("Client", (), {"chat": type("Chat", (), {"completions": api})()})()
    return llm, api


def test_a_rate_limited_model_hands_over_to_the_fallback_and_is_skipped_after(no_cooldown):
    llm, api = scripted({"main"})
    assert llm.chat([]) == "answer from backup"
    assert llm.chat([]) == "answer from backup"
    assert api.calls == ["main", "backup", "backup"]  # the limited model is not retried every call


def test_when_every_model_is_limited_the_user_gets_a_clear_message(no_cooldown):
    llm, _api = scripted({"main", "backup"})
    with pytest.raises(RateLimited, match="usage limit"):
        llm.tool_call([], [])
    with pytest.raises(RateLimited):
        llm.json([])


def test_gpt_oss_citation_markers_become_the_numbered_citations_the_code_checks():
    from agentkit.llm import plain

    assert plain("Torque to 18 Nm\u30101\u2020L7\u3011\u30102\u2020L8\u3011.") == "Torque to 18 Nm[1][2]."
    assert plain("see \u30103\u3011") == "see [3]"


def test_a_per_minute_limit_that_clears_in_seconds_is_waited_out(no_cooldown, monkeypatch):
    slept = []
    monkeypatch.setattr(llm_module.time, "sleep", slept.append)
    llm, api = scripted({"main"}, short_limits={"backup": 1})  # main is out for the day, backup for 3 s
    assert llm.chat([]) == "answer from backup"
    assert slept == [3.5] and api.calls == ["main", "backup", "backup"]  # the day-limited model is not retried
