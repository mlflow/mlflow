import json

import openai
import pytest

import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

from tests.tracing.helper import get_traces

pytest.importorskip("openai.resources.decisions")
httpx2 = pytest.importorskip("httpx2")


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
    transport = httpx2.MockTransport(handler)
    http_client = httpx2.AsyncClient if is_async else httpx2.Client
    openai_client = openai.AsyncOpenAI if is_async else openai.OpenAI
    return openai_client(
        api_key="test",
        base_url="https://api.openai.com/v1",
        http_client=http_client(transport=transport),
        max_retries=0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
async def test_decisions_autolog(is_async):
    mlflow.openai.autolog()
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=_RESPONSE)

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
    assert requests[0].url.path == "/v1/decisions"
    assert json.loads(requests[0].content)["questions"] == _QUESTIONS

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
        return httpx2.Response(
            400,
            json={"error": {"message": "Invalid question", "type": "invalid_request_error"}},
        )

    async def invoke(client):
        result = client.decisions.create(model="gpt-6-luna", input="Hello", questions=_QUESTIONS)
        if is_async:
            await result

    client = _client(is_async, handler)
    try:
        with pytest.raises(openai.BadRequestError, match="Invalid question"):
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
