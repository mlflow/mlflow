import asyncio
import importlib.util
import re
from typing import Any
from unittest.mock import ANY, patch

import anthropic
import pytest
from anthropic import beta_async_tool, beta_tool
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaUsage
from packaging.version import Version
from pydantic import BaseModel

import mlflow.anthropic
from mlflow.entities import SpanLogLevel
from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.version import IS_TRACING_SDK_ONLY

from tests.tracing.helper import get_traces

# anthropic>=1.0 is built on httpx2 instead of httpx.
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


class _Unit(BaseModel):
    unit: str


_UNIT_RESPONSE = Message(
    id="test_id",
    content=[TextBlock(text='{"unit": "celsius"}', type="text", citations=None)],
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


def _mock_http_client(responses, is_async, requests=None):
    # Mock the transport rather than `post` so the SDK's own response handling runs, e.g.
    # `parse()` building the parsed output and tool runners reading response metadata.
    def handler(request):
        if requests is not None:
            requests.append(request)
        return httpx.Response(200, json=responses.pop(0).to_dict())

    transport = httpx.MockTransport(handler)
    if is_async:
        return anthropic.DefaultAsyncHttpxClient(transport=transport)
    return anthropic.DefaultHttpxClient(transport=transport)


@pytest.mark.parametrize(
    ("get_resource", "method_name"),
    [
        (lambda client: client.messages, "parse"),
        (lambda client: client.beta.messages, "create"),
        (lambda client: client.beta.messages, "parse"),
    ],
    ids=["parse", "beta-create", "beta-parse"],
)
def test_messages_autolog_parse_and_beta(is_async, get_resource, method_name, mock_litellm_cost):
    mlflow.anthropic.autolog()
    requests = []
    http_client = _mock_http_client([_UNIT_RESPONSE], is_async, requests)
    client_cls = anthropic.AsyncAnthropic if is_async else anthropic.Anthropic
    resource = get_resource(client_cls(api_key="test_key", http_client=http_client))
    # `parse` was added in anthropic 0.73.0 (beta) and 0.77.0.
    if not hasattr(resource, method_name):
        pytest.skip(f"anthropic {anthropic.__version__} has no {method_name}")
    request = DUMMY_CREATE_MESSAGE_REQUEST
    if method_name == "parse":
        request = {**request, "output_format": _Unit}
    response = getattr(resource, method_name)(**request)
    if is_async:
        response = asyncio.run(response)

    if method_name == "parse":
        assert response.parsed_output == _Unit(unit="celsius")
    traces = get_traces()
    assert len(traces) == 1
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    class_name = "AsyncMessages" if is_async else "Messages"
    assert span.name == f"{class_name}.{method_name}"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs["messages"] == DUMMY_CREATE_MESSAGE_REQUEST["messages"]
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        TokenUsageKey.INPUT_TOKENS: 90,
        TokenUsageKey.OUTPUT_TOKENS: 20,
        TokenUsageKey.TOTAL_TOKENS: 110,
        TokenUsageKey.CACHE_READ_INPUT_TOKENS: 25,
        TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: 15,
    }
    assert span.model_name == "test_model"
    assert span.get_attribute(SpanAttributeKey.MODEL_PROVIDER) == "anthropic"
    if not IS_TRACING_SDK_ONLY:
        # The litellm mock prices 1.0 per input token and 2.0 per output token.
        assert span.llm_cost == {"input_cost": 90.0, "output_cost": 40.0, "total_cost": 130.0}

    span_ctx = span._span.get_span_context()
    trace_id = format(span_ctx.trace_id, "032x")
    span_id = format(span_ctx.span_id, "016x")
    assert requests[0].headers["traceparent"].startswith(f"00-{trace_id}-{span_id}-")


@pytest.mark.parametrize(
    "make_client",
    [
        pytest.param(
            lambda is_async, http_client: (
                anthropic.AsyncAnthropicBedrock if is_async else anthropic.AnthropicBedrock
            )(
                aws_access_key="test",
                aws_secret_key="test",
                aws_region="us-east-1",
                http_client=http_client,
            ),
            id="bedrock",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("boto3") is None,
                reason="AnthropicBedrock signs requests with boto3",
            ),
        ),
        pytest.param(
            lambda is_async, http_client: (
                anthropic.AsyncAnthropicVertex if is_async else anthropic.AnthropicVertex
            )(
                region="us-east5",
                project_id="test-project",
                access_token="test-token",
                http_client=http_client,
            ),
            id="vertex",
        ),
    ],
)
@pytest.mark.parametrize("method_name", ["create", "parse"])
def test_beta_messages_autolog_on_cloud_clients(is_async, make_client, method_name):
    mlflow.anthropic.autolog()
    resource = make_client(is_async, _mock_http_client([_UNIT_RESPONSE], is_async)).beta.messages
    if not hasattr(resource, method_name):
        pytest.skip(f"anthropic {anthropic.__version__} has no {method_name} on this client")
    request = DUMMY_CREATE_MESSAGE_REQUEST
    if method_name == "parse":
        request = {**request, "output_format": _Unit}
    response = getattr(resource, method_name)(**request)
    if is_async:
        asyncio.run(response)

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]
    class_name = "AsyncMessages" if is_async else "Messages"
    assert span.name == f"{class_name}.{method_name}"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE)[TokenUsageKey.INPUT_TOKENS] == 90


def _beta_message(content, stop_reason):
    return BetaMessage(
        id="test_id",
        content=content,
        model="test_model",
        role="assistant",
        stop_reason=stop_reason,
        stop_sequence=None,
        type="message",
        usage=BetaUsage(
            input_tokens=10,
            output_tokens=18,
            cache_creation_input_tokens=15,
            cache_read_input_tokens=25,
        ),
    )


def _get_unit(location: str) -> str:
    return "celsius"


async def _get_unit_async(location: str) -> str:
    return "celsius"


def test_beta_tool_runner_autolog(is_async):
    mlflow.anthropic.autolog()
    responses = [
        _beta_message(
            [
                BetaToolUseBlock(
                    id="tool_123",
                    name="get_unit",
                    input={"location": "San Francisco"},
                    type="tool_use",
                )
            ],
            "tool_use",
        ),
        _beta_message([BetaTextBlock(text="celsius", type="text", citations=None)], "end_turn"),
    ]
    request = {
        "model": "test_model",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Which unit is used in San Francisco?"}],
    }
    description = "Get the temperature unit commonly used in a given location"
    http_client = _mock_http_client(responses, is_async)

    if is_async:

        @mlflow.trace
        async def run_agent():
            client = anthropic.AsyncAnthropic(api_key="test_key", http_client=http_client)
            tool = beta_async_tool(_get_unit_async, name="get_unit", description=description)
            runner = client.beta.messages.tool_runner(tools=[tool], **request)
            return await runner.until_done()

        asyncio.run(run_agent())
    else:

        @mlflow.trace
        def run_agent():
            client = anthropic.Anthropic(api_key="test_key", http_client=http_client)
            tool = beta_tool(_get_unit, name="get_unit", description=description)
            runner = client.beta.messages.tool_runner(tools=[tool], **request)
            return runner.until_done()

        run_agent()

    traces = get_traces()
    assert len(traces) == 1
    root, *children = traces[0].data.spans
    assert root.name == "run_agent"
    assert [span.span_type for span in children] == [SpanType.CHAT_MODEL, SpanType.CHAT_MODEL]
    assert all(span.parent_id == root.span_id for span in children)
    # Each turn reports 10 input + 25 cache-read + 15 cache-write tokens and 18 output tokens.
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 100,
        TokenUsageKey.OUTPUT_TOKENS: 36,
        TokenUsageKey.TOTAL_TOKENS: 136,
        TokenUsageKey.CACHE_READ_INPUT_TOKENS: 50,
        TokenUsageKey.CACHE_CREATION_INPUT_TOKENS: 30,
    }


def test_beta_messages_autolog_records_tool_objects():
    if Version(anthropic.__version__) < Version("1.5.0"):
        pytest.skip("The beta Messages API accepts tool objects from anthropic 1.5.0")
    mlflow.anthropic.autolog()
    client = anthropic.Anthropic(
        api_key="test_key", http_client=_mock_http_client([_UNIT_RESPONSE], is_async=False)
    )
    tool = beta_tool(_get_unit, name="get_unit", description="Get the temperature unit")
    client.beta.messages.create(tools=[tool], **DUMMY_CREATE_MESSAGE_REQUEST)

    span = get_traces()[0].data.spans[0]
    assert [tool["name"] for tool in span.inputs["tools"]] == ["get_unit"]
    tools = span.get_attribute(SpanAttributeKey.CHAT_TOOLS)
    assert [tool["function"]["name"] for tool in tools] == ["get_unit"]
