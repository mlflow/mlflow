import json
import sys
from types import ModuleType, SimpleNamespace

import openai
import pytest
from packaging.version import Version
from pydantic import BaseModel

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

from tests.tracing.helper import get_traces

_USES_REAL_DECISIONS = Version(openai.__version__) >= Version("3.26.0")
if _USES_REAL_DECISIONS:
    import httpx2


class _InputTokenDetails(BaseModel):
    cached_tokens: int
    cache_write_tokens: int


class _Usage(BaseModel):
    input_tokens: int
    input_tokens_details: _InputTokenDetails
    output_tokens: int
    output_tokens_details: dict
    total_tokens: int


class _Answer(BaseModel):
    type: str
    name: str
    probability: float


class _Decision(BaseModel):
    model: str
    answers: list[_Answer]
    usage: _Usage


class _Decisions:
    def __init__(self, handler):
        self.handler = handler

    def create(self, **kwargs):
        return self.handler(kwargs)


class _AsyncDecisions:
    def __init__(self, handler):
        self.handler = handler

    async def create(self, **kwargs):
        return self.handler(kwargs)


@pytest.fixture(scope="module")
def mock_openai():
    if _USES_REAL_DECISIONS:
        yield  # The real SDK tests use MockTransport instead of the shared server.
        return

    resources = ModuleType("openai.resources.decisions")
    resources.Decisions = _Decisions
    resources.AsyncDecisions = _AsyncDecisions
    types = ModuleType("openai.types.decision")
    types.Decision = _Decision
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, resources.__name__, resources)
        patch.setitem(sys.modules, types.__name__, types)
        yield
        mlflow.openai.autolog(disable=True)


_QUESTIONS = [
    {
        "type": "predicate",
        "name": "answer_is_correct",
        "instructions": "Is the proposed answer correct?",
    }
]
_RESPONSE = {
    "model": "gpt-6-luna-2026-09-30",
    "answers": [{"type": "predicate", "name": "answer_is_correct", "probability": 0.93}],
    "usage": {
        "input_tokens": 172,
        "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 12},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 172,
    },
}


def _client(is_async, handler):
    if not _USES_REAL_DECISIONS:
        if is_async:

            async def close():
                pass

            return SimpleNamespace(decisions=_AsyncDecisions(handler), close=close)
        return SimpleNamespace(decisions=_Decisions(handler), close=lambda: None)

    transport = httpx2.MockTransport(handler)
    http_client = httpx2.AsyncClient if is_async else httpx2.Client
    openai_client = openai.AsyncOpenAI if is_async else openai.OpenAI
    return openai_client(
        api_key="test",
        base_url="https://api.openai.com/v1",
        http_client=http_client(transport=transport),
        max_retries=0,
    )


def _response(status, payload):
    if _USES_REAL_DECISIONS:
        return httpx2.Response(status, json=payload)
    if status >= 400:
        raise ValueError(payload["error"]["message"])
    return _Decision.model_validate(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_decisions_autolog(is_async):
    mlflow.openai.autolog()
    requests = []

    def handler(request):
        requests.append(request)
        return _response(200, _RESPONSE)

    client = _client(is_async, handler)
    try:
        result = client.decisions.create(
            model="gpt-6-luna",
            input="Question: What is 2 + 2? Proposed answer: 4.",
            questions=_QUESTIONS,
        )
        if is_async:
            result = await result
    finally:
        if is_async:
            await client.close()
        else:
            client.close()

    assert result.answers[0].probability == 0.93
    assert len(requests) == 1
    if _USES_REAL_DECISIONS:
        assert requests[0].url.path == "/v1/decisions"
        assert json.loads(requests[0].content)["questions"] == _QUESTIONS
    else:
        assert requests[0]["questions"] == _QUESTIONS

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 172,
        TokenUsageKey.OUTPUT_TOKENS: 0,
        TokenUsageKey.TOTAL_TOKENS: 172,
        TokenUsageKey.CACHE_READ_INPUT_TOKENS: 40,
        TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: 12,
    }
    span = traces[0].data.spans[0]
    assert span.span_type == SpanType.LLM
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "openai_decisions"
    assert span.get_attribute(SpanAttributeKey.MODEL) == _RESPONSE["model"]
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == traces[0].info.token_usage
    assert span.inputs == {
        "model": "gpt-6-luna",
        "input": "Question: What is 2 + 2? Proposed answer: 4.",
        "questions": _QUESTIONS,
    }
    assert span.outputs["answers"] == _RESPONSE["answers"]
    assert span.outputs["usage"] == _RESPONSE["usage"]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_decisions_error_autolog(is_async):
    mlflow.openai.autolog()

    def handler(request):
        return _response(
            400,
            {"error": {"message": "Invalid question", "type": "invalid_request_error"}},
        )

    async def invoke(client):
        result = client.decisions.create(model="gpt-6-luna", input="Hello", questions=_QUESTIONS)
        if is_async:
            await result

    client = _client(is_async, handler)
    try:
        error_type = openai.BadRequestError if _USES_REAL_DECISIONS else ValueError
        with pytest.raises(error_type, match="Invalid question"):
            await invoke(client)
    finally:
        if is_async:
            await client.close()
        else:
            client.close()

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    span = traces[0].data.spans[0]
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "openai_decisions"
    assert span.inputs["questions"] == _QUESTIONS
    assert span.outputs is None
