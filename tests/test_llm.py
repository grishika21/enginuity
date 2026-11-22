import httpx
import pytest
from openai import BadRequestError

from agentkit.llm import LLM, LLMError, parse_json_object

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
