import asyncio
import base64
from pathlib import Path
from typing import Any
from unittest.mock import patch

import anthropic
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

import mlflow.anthropic
from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces

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


def test_messages_autolog(is_async):
    mlflow.anthropic.autolog()

    _call_anthropic(DUMMY_CREATE_MESSAGE_REQUEST, DUMMY_CREATE_MESSAGE_RESPONSE, is_async)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "AsyncMessages.create" if is_async else "Messages.create"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == DUMMY_CREATE_MESSAGE_REQUEST
    # Only keep input_tokens / output_tokens fields in usage dict.
    span.outputs["usage"] = {
        key: span.outputs["usage"][key] for key in ["input_tokens", "output_tokens"]
    }
    assert span.outputs == DUMMY_CREATE_MESSAGE_RESPONSE.to_dict()

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "anthropic"

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

    image_dir = Path(__file__).parent.parent / "resources" / "images"
    with open(image_dir / "test.png", "rb") as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

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
                            "data": image_base64,
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
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "anthropic"

    assert traces[0].info.token_usage == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }


def test_messages_autolog_tool_calling(is_async):
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

    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "anthropic"

    assert traces[0].info.token_usage == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }


@pytest.mark.skipif(not _is_thinking_supported, reason="Thinking block is not supported")
def test_messages_autolog_with_thinking(is_async):
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
    assert span.outputs == DUMMY_CREATE_MESSAGE_WITH_THINKING_RESPONSE.to_dict()

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "anthropic"

    assert traces[0].info.token_usage == {
        "input_tokens": 10,
        "output_tokens": 18,
        "total_tokens": 28,
    }
