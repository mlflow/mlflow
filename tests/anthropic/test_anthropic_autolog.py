import asyncio
import gc
import inspect
import json
import re
from contextlib import AsyncExitStack
from typing import Any
from unittest.mock import ANY, patch

import anthropic
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from packaging.version import Version

import mlflow.anthropic
from mlflow.entities import SpanLogLevel
from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.version import IS_TRACING_SDK_ONLY

from tests.tracing.helper import get_traces

# anthropic 1.0 switched its HTTP client from `httpx` to `httpx2`.
if Version(anthropic.__version__).major >= 1:
    import httpx2 as httpx
else:
    import httpx

DUMMY_CREATE_MESSAGE_REQUEST = {
    "model": "test_model",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "test message"}],
}

DUMMY_CREATE_MESSAGE_RESPONSE = Message(
    id="test_id",
    content=[TextBlock(text="test answer", type="text", citations=None)],
    model="test_model",
    role="assistant",
    stop_reason="end_turn",
    stop_sequence=None,
    type="message",
    usage=Usage(input_tokens=10, output_tokens=18),
)

# Ref: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
DUMMY_CREATE_MESSAGE_WITH_TOOLS_REQUEST = {
    "model": "test_model",
    "max_tokens": 1024,
    "tools": [
        {
            "name": "get_unit",
            "description": "Get the temperature unit commonly used in a given location",
            "input_schema": {
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
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
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
    ],
    "messages": [
        {"role": "user", "content": "What's the weather like in San Francisco?"},
        {
            "role": "assistant",
            "content": [
                {
                    "text": "<thinking>I need to use the get_unit first.</thinking>",
                    "type": "text",
                },
                {
                    "id": "tool_123",
                    "name": "get_unit",
                    "input": {"location": "San Francisco"},
                    "type": "tool_use",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "content": "celsius",
                    "type": "tool_result",
                    "tool_use_id": "tool_123",
                    "is_error": False,
                }
            ],
        },
    ],
}

DUMMY_CREATE_MESSAGE_WITH_TOOLS_RESPONSE = Message(
    id="test_id",
    content=[
        TextBlock(
            text="<thinking>Next, I need to use the get_weather</thinking>",
            type="text",
            citations=None,
        ),
        ToolUseBlock(
            id="tool_456",
            name="get_weather",
            input={"location": "San Francisco", "unit": "celsius"},
            type="tool_use",
        ),
    ],
    model="test_model",
    role="assistant",
    stop_reason="end_turn",
    stop_sequence=None,
    type="message",
    usage=Usage(
        input_tokens=10,
        output_tokens=18,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    ),
)

_is_thinking_supported = False
try:
    from anthropic.types import ThinkingBlock

    _is_thinking_supported = True

    DUMMY_CREATE_MESSAGE_WITH_THINKING_REQUEST = {
        **DUMMY_CREATE_MESSAGE_REQUEST,
        "thinking": {"type": "enabled", "budget_tokens": 512},
    }

    DUMMY_CREATE_MESSAGE_WITH_THINKING_RESPONSE = DUMMY_CREATE_MESSAGE_RESPONSE.model_copy()
    DUMMY_CREATE_MESSAGE_WITH_THINKING_RESPONSE.content = [
        ThinkingBlock(
            type="thinking",
            thinking="I need to think about this for a while.",
            signature="ABC",
        ),
        TextBlock(
            text="test answer",
            type="text",
            citations=None,
        ),
    ]
except ImportError:
    pass


@pytest.fixture(params=[True, False], ids=["async", "sync"])
def is_async(request):
    return request.param


def _call_anthropic(request: dict[str, Any], mock_response: Message, is_async: bool):
    if is_async:
        with patch("anthropic._base_client.AsyncAPIClient.post", return_value=mock_response):
            client = anthropic.AsyncAnthropic(api_key="test_key")
            return asyncio.run(client.messages.create(**request))
    else:
        with patch("anthropic._base_client.SyncAPIClient.post", return_value=mock_response):
            client = anthropic.Anthropic(api_key="test_key")
            return client.messages.create(**request)


def test_messages_autolog(is_async, mock_litellm_cost):
    mlflow.anthropic.autolog()

    _call_anthropic(DUMMY_CREATE_MESSAGE_REQUEST, DUMMY_CREATE_MESSAGE_RESPONSE, is_async)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "AsyncMessages.create" if is_async else "Messages.create"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.log_level == SpanLogLevel.INFO
    assert span.inputs == DUMMY_CREATE_MESSAGE_REQUEST
    # Only keep input_tokens / output_tokens fields in usage dict.
    span.outputs["usage"] = {
        key: span.outputs["usage"][key] for key in ["input_tokens", "output_tokens"]
    }
    # Remove 'container' key added in anthropic v0.80.0 (code execution tool metadata)
    span.outputs.pop("container", None)
    # Remove 'stop_details' key added in anthropic v0.88.0
    span.outputs.pop("stop_details", None)
    if "diagnostics" in Message.model_fields:
        assert span.outputs.pop("diagnostics") is None
    assert span.outputs == DUMMY_CREATE_MESSAGE_RESPONSE.to_dict()

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }
    assert span.model_name == "test_model"
    assert span.get_attribute(SpanAttributeKey.MODEL_PROVIDER) == "anthropic"
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "anthropic"

    if not IS_TRACING_SDK_ONLY:
        # Verify cost is calculated (10 input tokens * 1.0 + 18 output tokens * 2.0)
        assert span.llm_cost == {
            "input_cost": 10.0,
            "output_cost": 36.0,
            "total_cost": 46.0,
        }

    assert traces[0].info.token_usage == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }

    mlflow.anthropic.autolog(disable=True)
    _call_anthropic(DUMMY_CREATE_MESSAGE_REQUEST, DUMMY_CREATE_MESSAGE_RESPONSE, is_async)

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1


def test_messages_autolog_multi_modal(is_async):
    mlflow.anthropic.autolog()

    dummy_multi_modal_request = {
        "model": "test_model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What text is in this image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": ANY,
                        },
                    },
                ],
            }
        ],
        "max_tokens": 1024,
    }

    _call_anthropic(dummy_multi_modal_request, DUMMY_CREATE_MESSAGE_RESPONSE, is_async)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "AsyncMessages.create" if is_async else "Messages.create"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == dummy_multi_modal_request

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }
    assert span.model_name == "test_model"
    assert span.get_attribute(SpanAttributeKey.MODEL_PROVIDER) == "anthropic"
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "anthropic"

    assert traces[0].info.token_usage == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }


def test_messages_autolog_tool_calling(is_async, mock_litellm_cost):
    mlflow.anthropic.autolog()

    _call_anthropic(
        DUMMY_CREATE_MESSAGE_WITH_TOOLS_REQUEST, DUMMY_CREATE_MESSAGE_WITH_TOOLS_RESPONSE, is_async
    )

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "AsyncMessages.create" if is_async else "Messages.create"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == DUMMY_CREATE_MESSAGE_WITH_TOOLS_REQUEST
    assert span.outputs == DUMMY_CREATE_MESSAGE_WITH_TOOLS_RESPONSE.to_dict(exclude_unset=False)

    if not IS_TRACING_SDK_ONLY:
        # Verify cost is calculated (10 input tokens * 1.0 + 18 output tokens * 2.0)
        assert span.llm_cost == {
            "input_cost": 10.0,
            "output_cost": 36.0,
            "total_cost": 46.0,
        }

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

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }
    assert span.model_name == "test_model"

    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "anthropic"

    assert traces[0].info.token_usage == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }


@pytest.mark.skipif(not _is_thinking_supported, reason="Thinking block is not supported")
def test_messages_autolog_with_thinking(is_async, mock_litellm_cost):
    mlflow.anthropic.autolog()

    _call_anthropic(
        DUMMY_CREATE_MESSAGE_WITH_THINKING_REQUEST,
        DUMMY_CREATE_MESSAGE_WITH_THINKING_RESPONSE,
        is_async,
    )

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "AsyncMessages.create" if is_async else "Messages.create"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == DUMMY_CREATE_MESSAGE_WITH_THINKING_REQUEST
    # Only keep input_tokens / output_tokens fields in usage dict.
    span.outputs["usage"] = {
        key: span.outputs["usage"][key] for key in ["input_tokens", "output_tokens"]
    }
    # Remove 'container' key added in anthropic v0.80.0 (code execution tool metadata)
    span.outputs.pop("container", None)
    # Remove 'stop_details' key added in anthropic v0.88.0
    span.outputs.pop("stop_details", None)
    if "diagnostics" in Message.model_fields:
        assert span.outputs.pop("diagnostics") is None
    assert span.outputs == DUMMY_CREATE_MESSAGE_WITH_THINKING_RESPONSE.to_dict()

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }
    assert span.model_name == "test_model"
    assert span.get_attribute(SpanAttributeKey.MODEL_PROVIDER) == "anthropic"
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "anthropic"

    if not IS_TRACING_SDK_ONLY:
        # Verify cost is calculated (10 input tokens * 1.0 + 18 output tokens * 2.0)
        assert span.llm_cost == {
            "input_cost": 10.0,
            "output_cost": 36.0,
            "total_cost": 46.0,
        }

    assert traces[0].info.token_usage == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }


def test_messages_autolog_with_diagnostics(is_async):
    if "diagnostics" not in Message.model_fields:
        pytest.skip("anthropic SDK does not support message diagnostics")
    from anthropic.types import Diagnostics

    diagnostics = Diagnostics(cache_miss_reason=None)
    response = DUMMY_CREATE_MESSAGE_RESPONSE.model_copy(update={"diagnostics": diagnostics})
    mlflow.anthropic.autolog()

    _call_anthropic(DUMMY_CREATE_MESSAGE_REQUEST, response, is_async)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].data.spans[0].outputs["diagnostics"] == diagnostics.model_dump()


DUMMY_CREATE_MESSAGE_WITH_CACHE_RESPONSE = Message(
    id="test_id",
    content=[TextBlock(text="cached answer", type="text", citations=None)],
    model="test_model",
    role="assistant",
    stop_reason="end_turn",
    stop_sequence=None,
    type="message",
    usage=Usage(
        input_tokens=50,
        output_tokens=20,
        cache_creation_input_tokens=15,
        cache_read_input_tokens=25,
    ),
)


def test_messages_autolog_with_cached_tokens(is_async, mock_litellm_cost):
    mlflow.anthropic.autolog()

    _call_anthropic(
        DUMMY_CREATE_MESSAGE_REQUEST, DUMMY_CREATE_MESSAGE_WITH_CACHE_RESPONSE, is_async
    )

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    span = traces[0].data.spans[0]

    # Anthropic's input_tokens (50) excludes cache tokens. After normalization:
    # input_tokens = 50 + 25 (cache_read) + 15 (cache_creation) = 90
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 90,
        TokenUsageKey.OUTPUT_TOKENS: 20,
        TokenUsageKey.TOTAL_TOKENS: 110,
        TokenUsageKey.CACHE_READ_INPUT_TOKENS: 25,
        TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: 15,
    }

    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 90,
        TokenUsageKey.OUTPUT_TOKENS: 20,
        TokenUsageKey.TOTAL_TOKENS: 110,
        TokenUsageKey.CACHE_READ_INPUT_TOKENS: 25,
        TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: 15,
    }


def test_messages_autolog_captures_1hr_cache_creation_breakdown(is_async, mock_litellm_cost):
    # The per-TTL cache_creation breakdown was added in a later Anthropic SDK version.
    if "cache_creation" not in Usage.model_fields:
        pytest.skip("anthropic SDK does not report the cache_creation TTL breakdown")
    from anthropic.types import CacheCreation

    response = Message(
        id="test_id",
        content=[TextBlock(text="cached answer", type="text", citations=None)],
        model="test_model",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=50,
            output_tokens=20,
            cache_creation_input_tokens=15,
            cache_read_input_tokens=25,
            cache_creation=CacheCreation(
                ephemeral_5m_input_tokens=5,
                ephemeral_1h_input_tokens=10,
            ),
        ),
    )

    mlflow.anthropic.autolog()

    _call_anthropic(DUMMY_CREATE_MESSAGE_REQUEST, response, is_async)

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]

    # The 1-hour portion (10) is a subset of cache_creation_input_tokens (15), so it does not
    # affect input-token normalization: input_tokens = 50 + 25 + 15 = 90.
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 90,
        TokenUsageKey.OUTPUT_TOKENS: 20,
        TokenUsageKey.TOTAL_TOKENS: 110,
        TokenUsageKey.CACHE_READ_INPUT_TOKENS: 25,
        TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: 15,
        TokenUsageKey.CACHE_CREATION_INPUT_TOKENS_ABOVE_1HR: 10,
    }


def test_tracing_headers_injected(is_async):
    mlflow.anthropic.autolog()

    _call_anthropic(DUMMY_CREATE_MESSAGE_REQUEST, DUMMY_CREATE_MESSAGE_RESPONSE, is_async)

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    span_ctx = span._span.get_span_context()
    expected_trace_id = format(span_ctx.trace_id, "032x")
    expected_span_id = format(span_ctx.span_id, "016x")

    # Verify the span has valid IDs that would produce a correct traceparent header
    from mlflow.tracing.distributed import _get_tracing_headers_from_span

    headers = _get_tracing_headers_from_span(span)
    assert "traceparent" in headers
    traceparent = headers["traceparent"]
    assert re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}", traceparent)
    assert traceparent.startswith(f"00-{expected_trace_id}-{expected_span_id}-")


def test_tracing_headers_preserve_user_extra_headers(is_async):
    mlflow.anthropic.autolog()

    request = {**DUMMY_CREATE_MESSAGE_REQUEST, "extra_headers": {"X-Custom": "my-value"}}
    _call_anthropic(request, DUMMY_CREATE_MESSAGE_RESPONSE, is_async)

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    # User-provided extra_headers should be recorded in span inputs and take precedence
    assert span.inputs.get("extra_headers", {}).get("X-Custom") == "my-value"


STREAM_EVENTS = [
    {
        "type": "message_start",
        "message": {
            "id": "test_id",
            "type": "message",
            "role": "assistant",
            "model": "test_model",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 18},
    },
    {"type": "message_stop"},
]

STREAM_WITH_TOOLS_EVENTS = [
    {
        "type": "message_start",
        "message": {
            "id": "test_id",
            "type": "message",
            "role": "assistant",
            "model": "test_model",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 40,
                "output_tokens": 1,
                "cache_creation_input_tokens": 15,
                "cache_read_input_tokens": 25,
            },
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "I need to "},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "use get_weather."},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "ABC"},
    },
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "text_delta", "text": "Let me check."},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "content_block_start",
        "index": 2,
        "content_block": {"type": "tool_use", "id": "tool_456", "name": "get_weather", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": ""},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": '{"location": "San Fran'},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": 'cisco", "unit": "celsius"}'},
    },
    {"type": "content_block_stop", "index": 2},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use", "stop_sequence": None},
        # The final input token count replaces the one in `message_start`, and the cache
        # counts, which are left out here, keep their value.
        "usage": {"input_tokens": 50, "output_tokens": 20},
    },
    {"type": "message_stop"},
]

STREAM_ERROR_EVENT = {
    "type": "error",
    "error": {"type": "overloaded_error", "message": "Overloaded"},
}


def _sse_response(events: list[dict[str, Any]]):
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


def _mock_client(is_async: bool, handler):
    transport = httpx.MockTransport(handler)
    if is_async:
        return anthropic.AsyncAnthropic(
            api_key="test_key", max_retries=0, http_client=httpx.AsyncClient(transport=transport)
        )
    return anthropic.Anthropic(
        api_key="test_key", max_retries=0, http_client=httpx.Client(transport=transport)
    )


def _streaming_client(is_async: bool, events: list[dict[str, Any]]):
    return _mock_client(is_async, lambda request: _sse_response(events))


async def _open_stream(stack: AsyncExitStack, client, use_helper: bool):
    """Opens a stream with `stream()` or `create(stream=True)`. Leaving `stack` closes it."""
    if use_helper:
        stream = client.messages.stream(**DUMMY_CREATE_MESSAGE_REQUEST)
    else:
        stream = client.messages.create(**DUMMY_CREATE_MESSAGE_REQUEST, stream=True)
        if inspect.isawaitable(stream):
            stream = await stream
    if hasattr(stream, "__aenter__"):
        return await stack.enter_async_context(stream)
    return stack.enter_context(stream)


async def _next_event(stream):
    if hasattr(stream, "__anext__"):
        return await anext(stream)
    return next(stream)


async def _read_all(stream):
    if hasattr(stream, "__aiter__"):
        return [event async for event in stream]
    return list(stream)


async def _close(stream):
    if inspect.isawaitable(result := stream.close()):
        await result


def _get_text(outputs: dict[str, Any]) -> str:
    return "".join(block["text"] for block in outputs["content"] if block["type"] == "text")


@pytest.fixture(params=[False, True], ids=["create", "stream"])
def use_helper(request):
    return request.param


@pytest.mark.asyncio
async def test_messages_autolog_streaming(is_async, use_helper, mock_litellm_cost):
    mlflow.anthropic.autolog()
    requests = []

    def handler(request):
        requests.append(request)
        return _sse_response(STREAM_EVENTS)

    client = _mock_client(is_async, handler)
    async with AsyncExitStack() as stack:
        stream = await _open_stream(stack, client, use_helper)
        # The span stays open until the stream is consumed.
        assert get_traces() == []
        events = await _read_all(stream)
        assert len(get_traces()) == 1

    text_deltas = [e.delta.text for e in events if e.type == "content_block_delta"]
    assert text_deltas == ["Hello", " world"]

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    method = "stream" if use_helper else "create"
    assert span.name == (f"AsyncMessages.{method}" if is_async else f"Messages.{method}")
    assert span.span_type == SpanType.CHAT_MODEL
    if use_helper:
        assert span.inputs == DUMMY_CREATE_MESSAGE_REQUEST
    else:
        assert span.inputs == {**DUMMY_CREATE_MESSAGE_REQUEST, "stream": True}

    # The output has the shape of a non-streaming response.
    assert span.outputs["id"] == "test_id"
    assert span.outputs["role"] == "assistant"
    assert _get_text(span.outputs) == "Hello world"
    assert span.outputs["stop_reason"] == "end_turn"
    assert [event.name for event in span.events] == [
        f"mlflow.chunk.item.{i}" for i in range(len(STREAM_EVENTS))
    ]
    first_chunk = json.loads(span.events[0].attributes["mlflow.chunk.value"])
    assert first_chunk["type"] == "message_start"

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }
    assert span.model_name == "test_model"
    assert span.get_attribute(SpanAttributeKey.MODEL_PROVIDER) == "anthropic"
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "anthropic"

    if not IS_TRACING_SDK_ONLY:
        # Verify cost is calculated (10 input tokens * 1.0 + 18 output tokens * 2.0)
        assert span.llm_cost == {
            "input_cost": 10.0,
            "output_cost": 36.0,
            "total_cost": 46.0,
        }

    assert traces[0].info.token_usage == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }

    # The request carries the trace context of the span.
    span_ctx = span._span.get_span_context()
    trace_id = format(span_ctx.trace_id, "032x")
    span_id = format(span_ctx.span_id, "016x")
    assert requests[0].headers["traceparent"].startswith(f"00-{trace_id}-{span_id}-")

    mlflow.anthropic.autolog(disable=True)
    async with AsyncExitStack() as stack:
        stream = await _open_stream(stack, client, use_helper)
        await _read_all(stream)

    # No new trace should be created
    assert len(get_traces()) == 1


@pytest.mark.asyncio
async def test_messages_autolog_stream_helper_final_message(is_async):
    mlflow.anthropic.autolog()

    client = _streaming_client(is_async, STREAM_EVENTS)
    if is_async:
        async with client.messages.stream(**DUMMY_CREATE_MESSAGE_REQUEST) as stream:
            text = "".join([text async for text in stream.text_stream])
            message = await stream.get_final_message()
    else:
        with client.messages.stream(**DUMMY_CREATE_MESSAGE_REQUEST) as stream:
            text = "".join(stream.text_stream)
            message = stream.get_final_message()

    assert text == "Hello world"
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    span = traces[0].data.spans[0]
    assert _get_text(span.outputs) == message.content[0].text == "Hello world"
    assert span.outputs["stop_reason"] == message.stop_reason == "end_turn"
    assert span.outputs["usage"]["input_tokens"] == message.usage.input_tokens == 10
    assert span.outputs["usage"]["output_tokens"] == message.usage.output_tokens == 18


@pytest.mark.asyncio
async def test_messages_autolog_streaming_tool_use_and_thinking(is_async, use_helper):
    mlflow.anthropic.autolog()

    client = _streaming_client(is_async, STREAM_WITH_TOOLS_EVENTS)
    async with AsyncExitStack() as stack:
        stream = await _open_stream(stack, client, use_helper)
        await _read_all(stream)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    span = traces[0].data.spans[0]
    thinking, text, tool_use = span.outputs["content"]
    assert thinking["type"] == "thinking"
    assert thinking["thinking"] == "I need to use get_weather."
    assert thinking["signature"] == "ABC"
    assert text["type"] == "text"
    assert text["text"] == "Let me check."
    assert tool_use["type"] == "tool_use"
    assert tool_use["id"] == "tool_456"
    assert tool_use["name"] == "get_weather"
    assert tool_use["input"] == {"location": "San Francisco", "unit": "celsius"}
    assert span.outputs["stop_reason"] == "tool_use"

    # Anthropic's input_tokens (50) excludes cache tokens. After normalization:
    # input_tokens = 50 + 25 (cache_read) + 15 (cache_creation) = 90
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 90,
        TokenUsageKey.OUTPUT_TOKENS: 20,
        TokenUsageKey.TOTAL_TOKENS: 110,
        TokenUsageKey.CACHE_READ_INPUT_TOKENS: 25,
        TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: 15,
    }


@pytest.mark.asyncio
async def test_messages_autolog_streaming_truncated_tool_input(is_async):
    mlflow.anthropic.autolog()

    events = [
        *STREAM_WITH_TOOLS_EVENTS[:12],
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": 'cisco", "unit": "cel'},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
            "usage": {"output_tokens": 20},
        },
        {"type": "message_stop"},
    ]
    client = _streaming_client(is_async, events)
    async with AsyncExitStack() as stack:
        stream = await _open_stream(stack, client, use_helper=False)
        await _read_all(stream)

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    assert span.outputs["stop_reason"] == "max_tokens"
    # The input is recorded as far as it was received.
    assert span.outputs["content"][2]["input"] == {"location": "San Francisco"}


@pytest.mark.asyncio
@pytest.mark.parametrize("num_events", [0, 3])
async def test_messages_autolog_streaming_close(is_async, use_helper, num_events):
    mlflow.anthropic.autolog()

    client = _streaming_client(is_async, STREAM_EVENTS)
    async with AsyncExitStack() as stack:
        stream = await _open_stream(stack, client, use_helper)
        for _ in range(num_events):
            await _next_event(stream)
        assert get_traces() == []
    await _close(stream)

    assert stream.response.is_closed
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    span = traces[0].data.spans[0]
    assert len(span.events) == num_events
    if num_events == 0:
        assert span.outputs is None
    else:
        assert _get_text(span.outputs) == "Hello"
        assert span.outputs["stop_reason"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("num_events", [0, 3])
async def test_messages_autolog_streaming_dropped(is_async, num_events):
    mlflow.anthropic.autolog()

    client = _streaming_client(is_async, STREAM_EVENTS)
    stream = client.messages.create(**DUMMY_CREATE_MESSAGE_REQUEST, stream=True)
    if is_async:
        stream = await stream
    for _ in range(num_events):
        await _next_event(stream)
    assert get_traces() == []

    # The stream is neither read to the end nor closed.
    del stream
    gc.collect()

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    span = traces[0].data.spans[0]
    assert len(span.events) == num_events
    if num_events == 0:
        assert span.outputs is None
    else:
        assert _get_text(span.outputs) == "Hello"


@pytest.mark.asyncio
async def test_messages_autolog_streaming_error_mid_stream(is_async, use_helper):
    mlflow.anthropic.autolog()

    client = _streaming_client(is_async, [*STREAM_EVENTS[:3], STREAM_ERROR_EVENT])
    async with AsyncExitStack() as stack:
        stream = await _open_stream(stack, client, use_helper)
        with pytest.raises(anthropic.APIError, match="Overloaded") as exc:
            await _read_all(stream)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    span = traces[0].data.spans[0]
    assert span.status.status_code == "ERROR"
    assert [event.name for event in span.events] == [
        "mlflow.chunk.item.0",
        "mlflow.chunk.item.1",
        "mlflow.chunk.item.2",
        "exception",
    ]
    assert span.events[-1].attributes["exception.type"] == type(exc.value).__name__
    # What was received before the error is still recorded.
    assert _get_text(span.outputs) == "Hello"


@pytest.mark.asyncio
async def test_messages_autolog_streaming_request_error(is_async, use_helper):
    mlflow.anthropic.autolog()

    error = {"type": "error", "error": {"type": "invalid_request_error", "message": "Bad request"}}
    client = _mock_client(is_async, lambda request: httpx.Response(400, json=error))
    async with AsyncExitStack() as stack:
        with pytest.raises(anthropic.BadRequestError, match="Bad request"):
            await _open_stream(stack, client, use_helper)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    span = traces[0].data.spans[0]
    assert span.outputs is None
    assert [event.name for event in span.events] == ["exception"]
    assert span.events[0].attributes["exception.type"] == "BadRequestError"


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_messages_autolog_streaming_close_error(is_async, error_type):
    mlflow.anthropic.autolog()

    client = _streaming_client(is_async, STREAM_EVENTS)
    stream = client.messages.create(**DUMMY_CREATE_MESSAGE_REQUEST, stream=True)
    if is_async:
        stream = await stream
    await _next_event(stream)

    error = error_type("Close failed")
    with patch.object(stream.response, "aclose" if is_async else "close", side_effect=error):
        with pytest.raises(error_type, match="Close failed") as exc:
            await _close(stream)
    await _close(stream)

    assert exc.value is error
    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    span = traces[0].data.spans[0]
    assert span.status.status_code == "ERROR"
    assert [event.name for event in span.events] == ["mlflow.chunk.item.0", "exception"]
    assert span.events[-1].attributes["exception.type"] == error_type.__name__
    assert span.events[-1].attributes["exception.message"] == "Close failed"


@pytest.mark.asyncio
async def test_messages_autolog_streaming_cancelled():
    mlflow.anthropic.autolog()
    reading = asyncio.Event()

    async def events():
        reading.set()
        await asyncio.Future()
        yield

    client = _streaming_client(True, STREAM_EVENTS)
    with patch.object(anthropic.AsyncStream, "__stream__", return_value=events()):
        stream = await client.messages.create(**DUMMY_CREATE_MESSAGE_REQUEST, stream=True)
        task = asyncio.create_task(anext(stream))
        await asyncio.wait_for(reading.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError, match="^$"):
            await task
        await stream.close()

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"
    span = traces[0].data.spans[0]
    assert span.status.status_code == "ERROR"
    assert [event.name for event in span.events] == ["exception"]
    assert span.events[0].attributes["exception.type"] == "CancelledError"


@pytest.mark.asyncio
async def test_messages_autolog_streaming_without_traces(is_async, use_helper):
    mlflow.anthropic.autolog(log_traces=False)

    client = _streaming_client(is_async, STREAM_EVENTS)
    async with AsyncExitStack() as stack:
        stream = await _open_stream(stack, client, use_helper)
        events = await _read_all(stream)

    text_deltas = [e.delta.text for e in events if e.type == "content_block_delta"]
    assert text_deltas == ["Hello", " world"]
    assert get_traces() == []
