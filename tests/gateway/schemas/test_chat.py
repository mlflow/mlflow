from typing import Any

import pydantic
import pytest

from mlflow.gateway.schemas import chat


def test_chat_request():
    chat.RequestPayload(**{
        "messages": [{"role": "user", "content": "content"}],
    })
    chat.RequestPayload(**{
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "content"},
                ],
            },
            {
                "role": "system",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": "data", "format": "wav"}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "image_url", "image_url": {"url": "url", "detail": "high"}},
                ],
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "123",
                        "function": {"name": "weather_tool", "arguments": "json string"},
                        "type": "function",
                    }
                ],
            },
            {"role": "tool", "content": "tool output", "tool_call_id": "123"},
        ],
    })
    chat.RequestPayload(**{
        "messages": [{"role": "user", "content": "content"}],
        "n": 1000,
        "extra": "extra",
        "temperature": 2.0,
    })
    chat.RequestPayload(**{
        "messages": [{"role": "user", "content": "content"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
    })

    with pytest.raises(pydantic.ValidationError, match="less than or equal to 2"):
        chat.RequestPayload(**{
            "messages": [{"role": "user", "content": "content"}],
            "temperature": 3.0,
        })

    with pytest.raises(pydantic.ValidationError, match="at least 1 item"):
        chat.RequestPayload(**{
            "messages": [{"role": "user", "content": "content"}],
            "stop": [],
        })

    with pytest.raises(pydantic.ValidationError, match="at least 1 item"):
        chat.RequestPayload(**{"messages": []})

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        chat.RequestPayload(**{})


def test_chat_request_preserves_nested_array_items():
    # Regression test for https://github.com/mlflow/mlflow/issues/23040.
    # Nested array schemas (e.g. list[list[str]]) must round-trip through the
    # request model without losing the inner `items` field; otherwise the
    # gateway forwards a malformed schema and providers reject it with
    # `array schema missing items`.
    nested_items = {
        "type": "array",
        "items": {"type": "string"},
    }
    payload = chat.RequestPayload(**{
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "send",
                    "description": "Send a message.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "buttons": {
                                "type": "array",
                                "items": nested_items,
                            },
                        },
                    },
                },
            }
        ],
    })

    dumped = payload.model_dump(exclude_none=True)
    inner = dumped["tools"][0]["function"]["parameters"]["properties"]["buttons"]["items"]
    assert inner == nested_items


def test_chat_request_preserves_nested_object_properties():
    # Regression test for https://github.com/mlflow/mlflow/issues/26359.
    # Nested object schemas must round-trip through the request model without
    # losing `properties` and `required`; otherwise the gateway forwards a bare
    # `{"type": "object"}` and the model returns empty or guessed arguments for it.
    nested_object = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "method": {"type": "string", "enum": ["GET", "POST"]},
            "auth": {
                "type": "object",
                "properties": {"token": {"type": "string"}},
                "required": ["token"],
            },
        },
        "required": ["url", "method"],
    }
    payload = chat.RequestPayload(**{
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "make_request",
                    "description": "Make an HTTP request.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "request": nested_object,
                            "batch": {"type": "array", "items": nested_object},
                        },
                        "required": ["request"],
                    },
                },
            }
        ],
    })

    dumped = payload.model_dump(exclude_none=True)
    properties = dumped["tools"][0]["function"]["parameters"]["properties"]
    assert properties["request"] == nested_object
    assert properties["batch"]["items"] == nested_object


@pytest.mark.parametrize(
    "properties",
    [
        {"city": {"type": "string", "required": True}},
        {
            "request": {
                "type": "object",
                "properties": {"url": {"type": "string", "required": True}},
            }
        },
        {
            "request": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": True,
            }
        },
    ],
    ids=["flat_param", "nested_param", "nested_object"],
)
def test_chat_request_rejects_non_list_required_in_tool_params(properties: dict[str, Any]):
    # `required` is validated on every tool parameter, so the draft-03 style
    # `"required": true` is rejected instead of being silently dropped.
    with pytest.raises(pydantic.ValidationError, match="Input should be a valid list"):
        chat.RequestPayload(**{
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather.",
                        "parameters": {"type": "object", "properties": properties},
                    },
                }
            ],
        })


def test_chat_request_accepts_json_schema_response_format_envelope():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    payload = chat.RequestPayload(**{
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_schema",
                "schema": schema,
                "description": "A response with a single answer field.",
            },
        },
    })

    dumped = payload.model_dump(exclude_none=True)
    assert dumped["response_format"]["json_schema"] == {
        "name": "answer_schema",
        "schema": schema,
        "strict": True,
        "description": "A response with a single answer field.",
    }


def test_chat_request_rejects_flat_json_schema_response_format():
    with pytest.raises(pydantic.ValidationError, match="Field required") as exc_info:
        chat.RequestPayload(**{
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        })

    error_locations = {error["loc"] for error in exc_info.value.errors()}
    assert ("response_format", "json_schema", "name") in error_locations
    assert ("response_format", "json_schema", "schema") in error_locations


def test_chat_response():
    chat.ResponsePayload(**{
        "created": 100,
        "model": "gpt-4",
        "choices": [
            {
                "message": {"role": "assistant", "content": "content"},
                "index": 0,
            },
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 1,
        },
    })

    chat.ResponsePayload(**{
        "id": "foobar",
        "created": 100,
        "model": "gpt-4",
        "object": "chat.completion",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "123",
                            "function": {"name": "weather_tool", "arguments": "json string"},
                            "type": "function",
                        }
                    ],
                },
                "finish_reason": "stop",
                "index": 0,
            },
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 1,
        },
    })

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        chat.ResponsePayload(**{"usage": {}})


def test_chat_stream_response():
    # Test stream response without usage
    chat.StreamResponsePayload(**{
        "id": "chatcmpl-123",
        "created": 100,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Hello"},
                "finish_reason": None,
            },
        ],
    })

    # Test stream response with usage (final chunk with stream_options.include_usage=true)
    response = chat.StreamResponsePayload(**{
        "id": "chatcmpl-123",
        "created": 100,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    })
    assert response.usage is not None
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 20
    assert response.usage.total_tokens == 30

    # Test stream response without usage field (default)
    response_no_usage = chat.StreamResponsePayload(**{
        "id": "chatcmpl-456",
        "created": 100,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "chunk"},
                "finish_reason": None,
            },
        ],
    })
    assert response_no_usage.usage is None
