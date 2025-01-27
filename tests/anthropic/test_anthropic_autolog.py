import base64
from unittest.mock import patch

import anthropic
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


@patch("anthropic._base_client.SyncAPIClient.post", return_value=DUMMY_CREATE_MESSAGE_RESPONSE)
def test_messages_autolog(mock_post):
    mlflow.anthropic.autolog()
    client = anthropic.Anthropic(api_key="test_key")
    client.messages.create(**DUMMY_CREATE_MESSAGE_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Messages.create"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == DUMMY_CREATE_MESSAGE_REQUEST
    # Only keep input_tokens / output_tokens fields in usage dict.
    span.outputs["usage"] = {
        key: span.outputs["usage"][key] for key in ["input_tokens", "output_tokens"]
    }
    assert span.outputs == DUMMY_CREATE_MESSAGE_RESPONSE.to_dict()

    assert span.get_attribute(SpanAttributeKey.CHAT_MESSAGES) == [
        {
            "role": "user",
            "content": "test message",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "text": "test answer",
                    "type": "text",
                }
            ],
        },
    ]

    mlflow.anthropic.autolog(disable=True)
    client = anthropic.Anthropic(api_key="test_key")
    client.messages.create(**DUMMY_CREATE_MESSAGE_REQUEST)

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1


@patch("anthropic._base_client.SyncAPIClient.post", return_value=DUMMY_CREATE_MESSAGE_RESPONSE)
def test_messages_autolog_multi_modal(mock_post):
    mlflow.anthropic.autolog()
    client = anthropic.Anthropic(api_key="test_key")

    with open("tests/resources/images/test.png", "rb") as f:
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

    client.messages.create(**dummy_multi_modal_request)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Messages.create"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == dummy_multi_modal_request
    assert span.get_attribute(SpanAttributeKey.CHAT_MESSAGES) == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What text is in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + image_base64,
                    },
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "text": "test answer",
                    "type": "text",
                }
            ],
        },
    ]


@patch(
    "anthropic._base_client.SyncAPIClient.post",
    return_value=DUMMY_CREATE_MESSAGE_WITH_TOOLS_RESPONSE,
)
def test_messages_autolog_tool_calling(mock_post):
    mlflow.anthropic.autolog()
    client = anthropic.Anthropic(api_key="test_key")
    client.messages.create(**DUMMY_CREATE_MESSAGE_WITH_TOOLS_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Messages.create"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == DUMMY_CREATE_MESSAGE_WITH_TOOLS_REQUEST
    assert span.outputs == DUMMY_CREATE_MESSAGE_WITH_TOOLS_RESPONSE.to_dict()

    assert span.get_attribute(SpanAttributeKey.CHAT_MESSAGES) == [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco?",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "text": "<thinking>I need to use the get_unit first.</thinking>",
                    "type": "text",
                }
            ],
            "tool_calls": [
                {
                    "id": "tool_123",
                    "type": "function",
                    "function": {
                        "name": "get_unit",
                        "arguments": '{"location": "San Francisco"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": [{"text": "celsius", "type": "text"}],
            "tool_call_id": "tool_123",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "text": "<thinking>Next, I need to use the get_weather</thinking>",
                    "type": "text",
                }
            ],
            "tool_calls": [
                {
                    "id": "tool_456",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco", "unit": "celsius"}',
                    },
                }
            ],
        },
    ]

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
