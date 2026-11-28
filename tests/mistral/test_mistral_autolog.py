import asyncio
import gc
import json
from importlib.metadata import version
from typing import Any
from unittest.mock import patch

import pytest
from packaging.version import Version

if Version(version("mistralai")).major >= 3:
    import httpx2 as httpx
else:
    import httpx

try:
    from mistralai.client import Mistral  # mistralai >= 2.0
    from mistralai.client.models import (
        AssistantMessage,
        ChatCompletionChoice,
        ChatCompletionResponse,
        FunctionCall,
        ThinkChunk,
        Tool,
        ToolCall,
        UsageInfo,
    )

    CHAT_DO_REQUEST_PATH = "mistralai.client.chat.Chat.do_request"
    CHAT_DO_REQUEST_ASYNC_PATH = "mistralai.client.chat.Chat.do_request_async"
except ImportError:
    from mistralai import Mistral  # mistralai < 2.0
    from mistralai.models import (
        AssistantMessage,
        ChatCompletionChoice,
        ChatCompletionResponse,
        FunctionCall,
        ThinkChunk,
        Tool,
        ToolCall,
        UsageInfo,
    )

    CHAT_DO_REQUEST_PATH = "mistralai.chat.Chat.do_request"
    CHAT_DO_REQUEST_ASYNC_PATH = "mistralai.chat.Chat.do_request_async"
from pydantic import BaseModel

import mlflow.mistral
from mlflow.entities import SpanLogLevel
from mlflow.entities.span import SpanType
from mlflow.mistral.autolog import _warn_untraceable_stream
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.distributed import (
    _get_tracing_headers_from_span,
    set_tracing_context_from_http_request_headers,
)
from mlflow.tracing.utils import aggregate_usage_from_spans
from mlflow.version import IS_TRACING_SDK_ONLY

from tests.tracing.helper import get_traces


@pytest.fixture(autouse=True)
def reset_mistral_autolog():
    yield
    mlflow.mistral.autolog(disable=True)


DUMMY_CHAT_COMPLETION_REQUEST = {
    "model": "test_model",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "test message"}],
}

DUMMY_CHAT_COMPLETION_RESPONSE = ChatCompletionResponse(
    id="test_id",
    object="chat.completion",
    model="test_model",
    usage=UsageInfo(prompt_tokens=10, completion_tokens=18, total_tokens=28),
    created=1736200000,
    choices=[
        ChatCompletionChoice(
            index=0,
            message=AssistantMessage(
                role="assistant",
                content="test answer",
                prefix=False,
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ],
)

# Ref: https://docs.mistral.ai/capabilities/function_calling/
DUMMY_CHAT_COMPLETION_WITH_TOOLS_REQUEST = {
    "model": "test_model",
    "max_tokens": 1024,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_unit",
                "description": "Get the temperature unit commonly used in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g., San Francisco, CA",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g., San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": 'The unit of temperature, "celsius" or "fahrenheit"',
                        },
                    },
                    "required": ["location", "unit"],
                },
            },
        },
    ],
    "messages": [
        {"role": "user", "content": "What's the weather like in San Francisco?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {"name": "get_unit", "arguments": '{"location": "San Francisco"}'},
                    "id": "tool_123",
                    "type": "function",
                }
            ],
            "prefix": False,
        },
        {"role": "tool", "name": "get_unit", "content": "celsius", "tool_call_id": "tool_123"},
    ],
}

DUMMY_CHAT_COMPLETION_WITH_TOOLS_RESPONSE = ChatCompletionResponse(
    id="test_id",
    object="chat.completion",
    model="test_model",
    usage=UsageInfo(prompt_tokens=11, completion_tokens=19, total_tokens=30),
    created=1736300000,
    choices=[
        ChatCompletionChoice(
            index=0,
            message=AssistantMessage(
                role="assistant",
                content="",
                prefix=False,
                tool_calls=[
                    ToolCall(
                        function=FunctionCall(
                            name="get_weather",
                            arguments='{"location": "San Francisco", "unit": "celsius"}',
                        ),
                        id="tool_456",
                        type="function",
                    ),
                ],
            ),
            finish_reason="tool_calls",
        )
    ],
)


def _make_httpx_response(response: BaseModel, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers={"Content-Type": "application/json"},
        text=response.model_dump_json(),
    )


def test_chat_complete_autolog(mock_litellm_cost):
    with patch(
        CHAT_DO_REQUEST_PATH,
        return_value=_make_httpx_response(DUMMY_CHAT_COMPLETION_RESPONSE),
    ):
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        client.chat.complete(**DUMMY_CHAT_COMPLETION_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Chat.complete"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.log_level == SpanLogLevel.INFO
    assert span.inputs == DUMMY_CHAT_COMPLETION_REQUEST
    # Only keep input_tokens / output_tokens fields in usage dict.
    span.outputs["usage"] = {
        key: span.outputs["usage"][key]
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }
    assert span.outputs == DUMMY_CHAT_COMPLETION_RESPONSE.model_dump()
    assert span.model_name == "test_model"
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "mistral"
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 10,
        TokenUsageKey.OUTPUT_TOKENS: 18,
        TokenUsageKey.TOTAL_TOKENS: 28,
    }
    if not IS_TRACING_SDK_ONLY:
        # Verify cost is calculated (10 input tokens * 1.0 + 18 output tokens * 2.0)
        assert span.llm_cost == {
            "input_cost": 10.0,
            "output_cost": 36.0,
            "total_cost": 46.0,
        }

    with patch(
        CHAT_DO_REQUEST_PATH,
        return_value=_make_httpx_response(DUMMY_CHAT_COMPLETION_RESPONSE),
    ):
        mlflow.mistral.autolog(disable=True)
        client = Mistral(api_key="test_key")
        client.chat.complete(**DUMMY_CHAT_COMPLETION_REQUEST)

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1


@pytest.mark.parametrize("sdk_tool_models", [False, True])
def test_chat_complete_autolog_tool_calling(sdk_tool_models):
    request = DUMMY_CHAT_COMPLETION_WITH_TOOLS_REQUEST
    if sdk_tool_models:
        request = {**request, "tools": [Tool.model_validate(t) for t in request["tools"]]}
    with patch(
        CHAT_DO_REQUEST_PATH,
        return_value=_make_httpx_response(DUMMY_CHAT_COMPLETION_WITH_TOOLS_RESPONSE),
    ):
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        client.chat.complete(**request)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Chat.complete"
    assert span.span_type == SpanType.CHAT_MODEL
    expected_inputs = DUMMY_CHAT_COMPLETION_WITH_TOOLS_REQUEST
    if sdk_tool_models:
        # mistralai < 2.0 serializes unset optional fields (e.g. `strict`) as None
        expected_inputs = {**request, "tools": [t.model_dump() for t in request["tools"]]}
    assert span.inputs == expected_inputs
    assert span.outputs == DUMMY_CHAT_COMPLETION_WITH_TOOLS_RESPONSE.model_dump()
    assert span.model_name == "test_model"

    assert span.get_attribute(SpanAttributeKey.CHAT_TOOLS) == [
        {
            "type": "function",
            "function": {
                "name": "get_unit",
                "description": "Get the temperature unit commonly used in a given location",
                "parameters": {
                    "properties": {
                        "location": {
                            "description": "The city and state, e.g., San Francisco, CA",
                            "type": "string",
                        },
                    },
                    "required": ["location"],
                    "type": "object",
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "properties": {
                        "location": {
                            "description": "The city and state, e.g., San Francisco, CA",
                            "type": "string",
                        },
                        "unit": {
                            "description": 'The unit of temperature, "celsius" or "fahrenheit"',
                            "enum": ["celsius", "fahrenheit"],
                            "type": "string",
                        },
                    },
                    "required": ["location", "unit"],
                    "type": "object",
                },
            },
        },
    ]
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "mistral"
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 11,
        TokenUsageKey.OUTPUT_TOKENS: 19,
        TokenUsageKey.TOTAL_TOKENS: 30,
    }


@pytest.mark.asyncio
async def test_chat_complete_async_autolog():
    with patch(
        CHAT_DO_REQUEST_ASYNC_PATH,
        return_value=_make_httpx_response(DUMMY_CHAT_COMPLETION_RESPONSE),
    ):
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        await client.chat.complete_async(**DUMMY_CHAT_COMPLETION_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Chat.complete_async"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == DUMMY_CHAT_COMPLETION_REQUEST
    span.outputs["usage"] = {
        key: span.outputs["usage"][key]
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }
    assert span.outputs == DUMMY_CHAT_COMPLETION_RESPONSE.model_dump()
    assert span.model_name == "test_model"
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "mistral"
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 10,
        TokenUsageKey.OUTPUT_TOKENS: 18,
        TokenUsageKey.TOTAL_TOKENS: 28,
    }


def _make_sse_response(chunks: list[dict[str, Any] | str]) -> httpx.Response:
    lines = [c if isinstance(c, str) else json.dumps(c) for c in chunks]
    body = "".join(f"data: {line}\n\n" for line in [*lines, "[DONE]"])
    return httpx.Response(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        content=body.encode(),
    )


def _chunk(delta: dict[str, Any], finish_reason: str | None = None, **kwargs) -> dict[str, Any]:
    return {
        "id": "test_id",
        "object": "chat.completion.chunk",
        "created": 1736200000,
        "model": "test_model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        **kwargs,
    }


DUMMY_STREAM_CHUNKS = [
    _chunk({"role": "assistant", "content": ""}),
    _chunk({"content": "test"}),
    _chunk(
        {"content": " answer"},
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 18, "total_tokens": 28},
    ),
]

DUMMY_STREAM_OUTPUT = {
    "id": "test_id",
    "object": "chat.completion",
    "model": "test_model",
    "created": 1736200000,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "test answer"},
            "finish_reason": "stop",
        }
    ],
}


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("is_stream", [False, True])
@pytest.mark.parametrize("user_headers", [None, {"X-Custom": "my-value"}])
@pytest.mark.asyncio
async def test_tracing_headers_parent_gateway_span_without_double_counting(
    is_async, is_stream, user_headers
):
    mlflow.mistral.autolog()
    request = DUMMY_CHAT_COMPLETION_REQUEST.copy()
    if user_headers is not None:
        request["http_headers"] = user_headers
    request_path = CHAT_DO_REQUEST_ASYNC_PATH if is_async else CHAT_DO_REQUEST_PATH
    response = (
        _make_sse_response(DUMMY_STREAM_CHUNKS)
        if is_stream
        else _make_httpx_response(DUMMY_CHAT_COMPLETION_RESPONSE)
    )
    with patch(request_path, return_value=response) as mock_request:
        client = Mistral(api_key="test_key")
        match is_async, is_stream:
            case True, True:
                [event async for event in await client.chat.stream_async(**request)]
            case True, False:
                await client.chat.complete_async(**request)
            case False, True:
                list(client.chat.stream(**request))
            case False, False:
                client.chat.complete(**request)

    span = get_traces()[0].data.spans[0]
    sent_headers = mock_request.call_args.kwargs["request"].headers
    # Stored spans do not retain the live sampling flag, so compare the trace and span IDs.
    assert (
        sent_headers["traceparent"].split("-")[:3]
        == _get_tracing_headers_from_span(span)["traceparent"].split("-")[:3]
    )
    if user_headers is not None:
        assert sent_headers["X-Custom"] == "my-value"
    assert span.inputs.get("http_headers") == user_headers
    assert "traceparent" not in (span.inputs.get("http_headers") or {})

    # A gateway span using the propagated context is a child, so its repeated usage is skipped.
    with set_tracing_context_from_http_request_headers(sent_headers):
        with mlflow.start_span("gateway") as gateway_span:
            gateway_span.set_attribute(
                SpanAttributeKey.CHAT_USAGE, span.get_attribute(SpanAttributeKey.CHAT_USAGE)
            )

    assert gateway_span.trace_id == span.trace_id
    assert gateway_span.parent_id == span.span_id
    assert aggregate_usage_from_spans([span, gateway_span]) == span.get_attribute(
        SpanAttributeKey.CHAT_USAGE
    )


def _assert_stream_trace(span_name: str):
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == span_name
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == DUMMY_CHAT_COMPLETION_REQUEST
    assert span.outputs == DUMMY_STREAM_OUTPUT
    assert span.model_name == "test_model"
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "mistral"
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 10,
        TokenUsageKey.OUTPUT_TOKENS: 18,
        TokenUsageKey.TOTAL_TOKENS: 28,
    }
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 10,
        TokenUsageKey.OUTPUT_TOKENS: 18,
        TokenUsageKey.TOTAL_TOKENS: 28,
    }


def test_chat_stream_autolog():
    with patch(
        CHAT_DO_REQUEST_PATH, return_value=_make_sse_response(DUMMY_STREAM_CHUNKS)
    ) as mock_request:
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        stream = client.chat.stream(**DUMMY_CHAT_COMPLETION_REQUEST)
        # The span stays open until the stream is consumed.
        assert get_traces() == []
        text = "".join(event.data.choices[0].delta.content for event in stream)

    mock_request.assert_called_once()
    assert text == "test answer"
    _assert_stream_trace("Chat.stream")


@pytest.mark.asyncio
async def test_chat_stream_async_autolog():
    with patch(
        CHAT_DO_REQUEST_ASYNC_PATH, return_value=_make_sse_response(DUMMY_STREAM_CHUNKS)
    ) as mock_request:
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        stream = await client.chat.stream_async(**DUMMY_CHAT_COMPLETION_REQUEST)
        text = "".join([event.data.choices[0].delta.content async for event in stream])

    mock_request.assert_called_once()
    assert text == "test answer"
    _assert_stream_trace("Chat.stream_async")


def test_chat_stream_autolog_tool_calling():
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_unit", "arguments": '{"location": "Paris"}'},
        "index": 0,
    }
    parallel_call = {
        "id": "call_2",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"location": "Paris", '},
        "index": 1,
    }
    # A follow-up chunk without an ID continues the arguments of the previous tool call.
    continuation = {"function": {"name": "", "arguments": '"unit": "celsius"}'}, "index": 1}
    chunks = [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"tool_calls": [tool_call, parallel_call]}),
        _chunk({"tool_calls": [continuation]}, finish_reason="tool_calls"),
    ]
    with patch(CHAT_DO_REQUEST_PATH, return_value=_make_sse_response(chunks)) as mock_request:
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        list(client.chat.stream(**DUMMY_CHAT_COMPLETION_WITH_TOOLS_REQUEST))

    mock_request.assert_called_once()
    span = get_traces()[0].data.spans[0]
    choice = span.outputs["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_unit", "arguments": '{"location": "Paris"}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris", "unit": "celsius"}',
            },
        },
    ]
    assert [
        tool["function"]["name"] for tool in span.get_attribute(SpanAttributeKey.CHAT_TOOLS)
    ] == [
        "get_unit",
        "get_weather",
    ]


def test_chat_stream_autolog_reasoning_content():
    def thinking(text):
        return {"type": "thinking", "thinking": [{"type": "text", "text": text}]}

    chunks = [
        _chunk({"role": "assistant", "content": [thinking("Let me ")]}),
        _chunk({"content": [thinking("think.")]}),
        _chunk({"content": [{"type": "text", "text": "The answer"}]}),
        _chunk({"content": " is 42."}, finish_reason="stop"),
    ]
    with patch(CHAT_DO_REQUEST_PATH, return_value=_make_sse_response(chunks)) as mock_request:
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        list(client.chat.stream(**DUMMY_CHAT_COMPLETION_REQUEST))

    mock_request.assert_called_once()
    message = get_traces()[0].data.spans[0].outputs["choices"][0]["message"]
    assert message["content"] == [
        thinking("Let me think."),
        {"type": "text", "text": "The answer is 42."},
    ]


def test_chat_stream_autolog_closed_early():
    with patch(
        CHAT_DO_REQUEST_PATH, return_value=_make_sse_response(DUMMY_STREAM_CHUNKS)
    ) as mock_request:
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        # Leaving the ``with`` block closes the HTTP response before the stream is exhausted.
        with client.chat.stream(**DUMMY_CHAT_COMPLETION_REQUEST) as stream:
            next(stream)
            next(stream)

    mock_request.assert_called_once()
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    message = traces[0].data.spans[0].outputs["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "test"}


@pytest.mark.asyncio
async def test_chat_stream_async_autolog_closed_early():
    with patch(
        CHAT_DO_REQUEST_ASYNC_PATH, return_value=_make_sse_response(DUMMY_STREAM_CHUNKS)
    ) as mock_request:
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        async with await client.chat.stream_async(**DUMMY_CHAT_COMPLETION_REQUEST) as stream:
            await anext(stream)
            await anext(stream)

    mock_request.assert_called_once()
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    message = traces[0].data.spans[0].outputs["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "test"}


def test_chat_stream_autolog_unknown_stream_internals():
    class OpaqueStream:
        def __init__(self, response, *args, **kwargs):
            self._events = iter([])

        def __iter__(self):
            return self._events

    event_stream_path = CHAT_DO_REQUEST_PATH.replace("chat.Chat.do_request", "utils.eventstreaming")
    _warn_untraceable_stream.cache_clear()
    with (
        patch(
            CHAT_DO_REQUEST_PATH, return_value=_make_sse_response(DUMMY_STREAM_CHUNKS)
        ) as mock_request,
        patch(f"{event_stream_path}.EventStream", OpaqueStream),
        patch("mlflow.mistral.autolog._logger.warning") as mock_warning,
    ):
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        stream = client.chat.stream(**DUMMY_CHAT_COMPLETION_REQUEST)
        client.chat.stream(**DUMMY_CHAT_COMPLETION_REQUEST)

    assert mock_request.call_count == 2
    assert isinstance(stream, OpaqueStream)
    # The stream is returned untouched, and the span ends when ``stream()`` returns.
    traces = get_traces()
    assert len(traces) == 2
    assert traces[0].info.status == "OK"
    assert traces[0].data.spans[0].outputs is None
    assert mock_warning.call_count == 1
    assert "Unable to trace a Mistral stream" in mock_warning.call_args[0][0]


async def _consume_stream(client, is_async):
    if is_async:
        stream = await client.chat.stream_async(**DUMMY_CHAT_COMPLETION_REQUEST)
        return [event async for event in stream]
    return list(client.chat.stream(**DUMMY_CHAT_COMPLETION_REQUEST))


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_chat_stream_autolog_error_mid_stream(is_async):
    usage = {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
    chunks = [_chunk({"role": "assistant", "content": "test"}, usage=usage), "{not json"]
    request_path = CHAT_DO_REQUEST_ASYNC_PATH if is_async else CHAT_DO_REQUEST_PATH
    with patch(request_path, return_value=_make_sse_response(chunks)) as mock_request:
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        with pytest.raises(Exception):  # noqa: PT011
            await _consume_stream(client, is_async)

    mock_request.assert_called_once()
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    span = traces[0].data.spans[0]
    assert span.status.status_code == "ERROR"
    assert span.events[0].name == "exception"
    # What was received before the failure is kept.
    assert span.outputs["choices"][0]["message"] == {"role": "assistant", "content": "test"}
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 10,
        TokenUsageKey.OUTPUT_TOKENS: 1,
        TokenUsageKey.TOTAL_TOKENS: 11,
    }


def test_chat_stream_autolog_disabled_traces():
    with patch(
        CHAT_DO_REQUEST_PATH, return_value=_make_sse_response(DUMMY_STREAM_CHUNKS)
    ) as mock_request:
        mlflow.mistral.autolog(log_traces=False)
        client = Mistral(api_key="test_key")
        text = "".join(
            event.data.choices[0].delta.content
            for event in client.chat.stream(**DUMMY_CHAT_COMPLETION_REQUEST)
        )

    mock_request.assert_called_once()
    assert text == "test answer"
    assert get_traces() == []


async def _hang_after_first_chunk():
    yield f"data: {json.dumps(DUMMY_STREAM_CHUNKS[1])}\n\n".encode()
    await asyncio.sleep(60)


@pytest.mark.asyncio
async def test_chat_complete_async_autolog_cancelled():
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(60)

    with patch(CHAT_DO_REQUEST_ASYNC_PATH, side_effect=slow_request):
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        # The timeout must reach the caller instead of the call returning None.
        with pytest.raises(asyncio.TimeoutError):  # noqa: PT011
            await asyncio.wait_for(
                client.chat.complete_async(**DUMMY_CHAT_COMPLETION_REQUEST), timeout=0.1
            )

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    span = traces[0].data.spans[0]
    assert span.events[0].name == "exception"
    assert span.events[0].attributes["exception.type"] == "CancelledError"


async def _read_with_timeout(stream, received, timeout):
    async with asyncio.timeout(timeout):
        # Append as events arrive, so the events received before the timeout are kept.
        async for event in stream:
            received.append(event.data.choices[0].delta.content)


@pytest.mark.asyncio
async def test_chat_stream_async_autolog_cancelled_mid_stream():
    response = httpx.Response(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        content=_hang_after_first_chunk(),
    )
    with patch(CHAT_DO_REQUEST_ASYNC_PATH, return_value=response):
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        stream = await client.chat.stream_async(**DUMMY_CHAT_COMPLETION_REQUEST)
        received = []
        with pytest.raises(asyncio.TimeoutError):  # noqa: PT011
            await _read_with_timeout(stream, received, timeout=0.5)

    assert received == ["test"]
    traces = get_traces()
    assert len(traces) == 1
    # An interrupted stream is an error, not a deliberate early close.
    assert traces[0].info.status == "ERROR"
    span = traces[0].data.spans[0]
    assert span.events[0].attributes["exception.type"] == "CancelledError"
    assert span.outputs["choices"][0]["message"] == {"role": "assistant", "content": "test"}


def test_chat_stream_autolog_abandoned_stream_ends_span():
    with patch(CHAT_DO_REQUEST_PATH, return_value=_make_sse_response(DUMMY_STREAM_CHUNKS)):
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        gc.disable()
        try:
            stream = client.chat.stream(**DUMMY_CHAT_COMPLETION_REQUEST)
            for _ in stream:
                break
            assert get_traces() == []
            # Dropping the last reference ends the span right away, without a GC pass.
            del stream
            traces = get_traces()
        finally:
            gc.enable()

    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert traces[0].data.spans[0].outputs["choices"][0]["message"]["content"] == ""


def test_chat_stream_autolog_multiple_choices():
    def chunk(index, delta, finish_reason=None):
        return {
            **_chunk(delta),
            "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}],
        }

    chunks = [
        chunk(0, {"role": "assistant", "content": "Hello"}),
        chunk(1, {"role": "assistant", "content": "Hi"}),
        chunk(0, {"content": " there"}, finish_reason="stop"),
        chunk(1, {"content": " you"}, finish_reason="length"),
    ]
    with patch(CHAT_DO_REQUEST_PATH, return_value=_make_sse_response(chunks)):
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        list(client.chat.stream(**DUMMY_CHAT_COMPLETION_REQUEST, n=2))

    assert get_traces()[0].data.spans[0].outputs["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there"},
            "finish_reason": "stop",
        },
        {
            "index": 1,
            "message": {"role": "assistant", "content": "Hi you"},
            "finish_reason": "length",
        },
    ]


def test_chat_stream_autolog_interleaved_tool_call_fragments():
    def call(index, arguments, call_id=None, name=""):
        tool_call = {"function": {"name": name, "arguments": arguments}, "index": index}
        if call_id:
            tool_call |= {"id": call_id, "type": "function"}
        return tool_call

    chunks = [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({
            "tool_calls": [call(0, '{"x": ', "call_a", "f"), call(1, '{"y": ', "call_b", "g")]
        }),
        _chunk({"tool_calls": [call(0, "1}")]}),
        _chunk({"tool_calls": [call(1, "2}")]}, finish_reason="tool_calls"),
    ]
    with patch(CHAT_DO_REQUEST_PATH, return_value=_make_sse_response(chunks)):
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        list(client.chat.stream(**DUMMY_CHAT_COMPLETION_WITH_TOOLS_REQUEST))

    tool_calls = get_traces()[0].data.spans[0].outputs["choices"][0]["message"]["tool_calls"]
    assert [(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in tool_calls] == [
        ("call_a", "f", '{"x": 1}'),
        ("call_b", "g", '{"y": 2}'),
    ]


def test_chat_stream_autolog_tool_calls_without_ids():
    def call(name, arguments):
        return {"function": {"name": name, "arguments": arguments}}

    chunks = [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"tool_calls": [call("get_unit", '{"location": "Paris"}')]}),
        _chunk({"tool_calls": [call("get_weather", '{"location": "Rome"}')]}),
    ]
    with patch(CHAT_DO_REQUEST_PATH, return_value=_make_sse_response(chunks)):
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        list(client.chat.stream(**DUMMY_CHAT_COMPLETION_WITH_TOOLS_REQUEST))

    tool_calls = get_traces()[0].data.spans[0].outputs["choices"][0]["message"]["tool_calls"]
    assert [(c["function"]["name"], c["function"]["arguments"]) for c in tool_calls] == [
        ("get_unit", '{"location": "Paris"}'),
        ("get_weather", '{"location": "Rome"}'),
    ]


def test_chat_stream_autolog_thinking_blocks():
    reference = {"type": "reference", "reference_ids": [1]}
    chunks = [
        _chunk({
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": [{"type": "text", "text": "First "}]}],
        }),
        _chunk({
            "content": [
                {
                    "type": "thinking",
                    "thinking": [{"type": "text", "text": "idea."}, reference],
                    "signature": "sig_1",
                    "closed": True,
                }
            ]
        }),
        _chunk({
            "content": [
                {
                    "type": "thinking",
                    "thinking": [{"type": "text", "text": "Second."}],
                    "closed": True,
                }
            ]
        }),
        _chunk({"content": "Done."}, finish_reason="stop"),
    ]
    with patch(CHAT_DO_REQUEST_PATH, return_value=_make_sse_response(chunks)):
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        list(client.chat.stream(**DUMMY_CHAT_COMPLETION_REQUEST))

    content = get_traces()[0].data.spans[0].outputs["choices"][0]["message"]["content"]
    assert [part["type"] for part in content] == ["thinking", "thinking", "text"]
    first, second, text = content
    assert first["thinking"] == [{"type": "text", "text": "First idea."}, reference]
    # ``signature`` is only part of ``ThinkChunk`` in newer mistralai versions.
    expected_signature = "sig_1" if "signature" in ThinkChunk.model_fields else None
    assert first.get("signature") == expected_signature
    assert first["closed"] is True
    assert second["thinking"] == [{"type": "text", "text": "Second."}]
    assert text == {"type": "text", "text": "Done."}


def test_chat_stream_autolog_usage_kept_when_reconstruction_fails():
    with (
        patch(CHAT_DO_REQUEST_PATH, return_value=_make_sse_response(DUMMY_STREAM_CHUNKS)),
        patch(
            "mlflow.mistral.autolog._reconstruct_completion_from_chunks",
            side_effect=KeyError("type"),
        ),
    ):
        mlflow.mistral.autolog()
        client = Mistral(api_key="test_key")
        list(client.chat.stream(**DUMMY_CHAT_COMPLETION_REQUEST))

    span = get_traces()[0].data.spans[0]
    assert span.outputs is None
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 10,
        TokenUsageKey.OUTPUT_TOKENS: 18,
        TokenUsageKey.TOTAL_TOKENS: 28,
    }
