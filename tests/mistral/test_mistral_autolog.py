from unittest.mock import patch

import httpx
import mistralai
from mistralai.models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionResponse,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from pydantic import BaseModel

import mlflow.mistral
from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

from tests.tracing.helper import get_traces

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


@patch(
    "mistralai.chat.Chat.do_request",
    return_value=_make_httpx_response(DUMMY_CHAT_COMPLETION_RESPONSE),
)
def test_chat_complete_autolog(mock_complete):
    mlflow.mistral.autolog()
    client = mistralai.Mistral(api_key="test_key")
    client.chat.complete(**DUMMY_CHAT_COMPLETION_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Chat.complete"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == DUMMY_CHAT_COMPLETION_REQUEST
    # Only keep input_tokens / output_tokens fields in usage dict.
    span.outputs["usage"] = {
        key: span.outputs["usage"][key]
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }
    assert span.outputs == DUMMY_CHAT_COMPLETION_RESPONSE.model_dump()
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "mistral"
    assert traces[0].info.token_usage == {
        TokenUsageKey.INPUT_TOKENS: 10,
        TokenUsageKey.OUTPUT_TOKENS: 18,
        TokenUsageKey.TOTAL_TOKENS: 28,
    }

    mlflow.mistral.autolog(disable=True)
    client = mistralai.Mistral(api_key="test_key")
    client.chat.complete(**DUMMY_CHAT_COMPLETION_REQUEST)

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1


@patch(
    "mistralai.chat.Chat.do_request",
    return_value=_make_httpx_response(DUMMY_CHAT_COMPLETION_WITH_TOOLS_RESPONSE),
)
def test_chat_complete_autolog_tool_calling(mock_complete):
    mlflow.mistral.autolog()
    client = mistralai.Mistral(api_key="test_key")
    client.chat.complete(**DUMMY_CHAT_COMPLETION_WITH_TOOLS_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Chat.complete"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == DUMMY_CHAT_COMPLETION_WITH_TOOLS_REQUEST
    assert span.outputs == DUMMY_CHAT_COMPLETION_WITH_TOOLS_RESPONSE.model_dump()

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
