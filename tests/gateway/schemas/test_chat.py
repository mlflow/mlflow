import pydantic
import pytest

from mlflow.gateway.schemas import chat


def test_chat_request():
    chat.RequestPayload(
        **{
            "messages": [{"role": "user", "content": "content"}],
        }
    )
    chat.RequestPayload(
        **{
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
        }
    )
    chat.RequestPayload(
        **{
            "messages": [{"role": "user", "content": "content"}],
            "n": 1000,
            "extra": "extra",
            "temperature": 2.0,
        }
    )
    chat.RequestPayload(
        **{
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
        }
    )

    with pytest.raises(pydantic.ValidationError, match="less than or equal to 2"):
        chat.RequestPayload(
            **{
                "messages": [{"role": "user", "content": "content"}],
                "temperature": 3.0,
            }
        )

    with pytest.raises(pydantic.ValidationError, match="at least 1 item"):
        chat.RequestPayload(
            **{
                "messages": [{"role": "user", "content": "content"}],
                "stop": [],
            }
        )

    with pytest.raises(pydantic.ValidationError, match="at least 1 item"):
        chat.RequestPayload(**{"messages": []})

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        chat.RequestPayload(**{})


def test_chat_response():
    chat.ResponsePayload(
        **{
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
        }
    )

    chat.ResponsePayload(
        **{
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
        }
    )

    with pytest.raises(pydantic.ValidationError, match=r"(?i)field required"):
        chat.ResponsePayload(**{"usage": {}})


def test_chat_stream_response():
    # Test stream response without usage
    chat.StreamResponsePayload(
        **{
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
        }
    )

    # Test stream response with usage (final chunk with stream_options.include_usage=true)
    response = chat.StreamResponsePayload(
        **{
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
        }
    )
    assert response.usage is not None
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 20
    assert response.usage.total_tokens == 30

    # Test stream response without usage field (default)
    response_no_usage = chat.StreamResponsePayload(
        **{
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
        }
    )
    assert response_no_usage.usage is None
