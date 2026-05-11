from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import EndpointConfig, _OpenAICompatibleConfig
from mlflow.gateway.providers.base import PassthroughAction
from mlflow.gateway.providers.openai_compatible import (
    OpenAICompatibleAdapter,
    OpenAICompatibleProvider,
)
from mlflow.gateway.schemas import chat, embeddings

from tests.gateway.tools import (
    MockAsyncResponse,
    MockAsyncStreamingResponse,
    mock_http_client,
)

# --- Concrete subclass for testing ---


class _TestConfig(_OpenAICompatibleConfig):
    pass


class _TestProvider(OpenAICompatibleProvider):
    DISPLAY_NAME = "TestProvider"
    CONFIG_TYPE = _TestConfig
    DEFAULT_API_BASE = "https://api.test-provider.com/v1"


# --- fixtures ---


_TEST_PROVIDER_NAME = "_test_openai_compat"

# Register once at module load so EndpointConfig validation accepts this provider
from mlflow.gateway.provider_registry import provider_registry

if _TEST_PROVIDER_NAME not in provider_registry.keys():
    provider_registry.register(_TEST_PROVIDER_NAME, _TestProvider)


def _make_provider(*, api_base: str | None = None) -> _TestProvider:

    config_dict = {
        "api_key": "test-key",
    }
    if api_base is not None:
        config_dict["api_base"] = api_base

    endpoint_config = EndpointConfig(
        name="test-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": _TEST_PROVIDER_NAME,
            "name": "test-model",
            "config": config_dict,
        },
    )
    return _TestProvider(endpoint_config)


def _chat_response():
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "test-model",
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }


def _embeddings_response():
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0,
            }
        ],
        "model": "test-model",
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8,
        },
        "headers": {"Content-Type": "application/json"},
    }


def _make_endpoint_config():
    provider = _make_provider()
    return provider.config


# --- provider tests ---


def test_default_api_base():
    provider = _make_provider()
    assert provider._api_base == "https://api.test-provider.com/v1"


def test_custom_api_base():
    provider = _make_provider(api_base="https://custom.example.com/v1")
    assert provider._api_base == "https://custom.example.com/v1"


def test_headers():
    provider = _make_provider()
    assert provider.headers == {"Authorization": "Bearer test-key"}


def test_get_headers_merges_client_headers():
    provider = _make_provider()
    merged = provider._get_headers(headers={"X-Custom": "value", "host": "ignored"})
    assert merged == {"Authorization": "Bearer test-key", "X-Custom": "value"}
    assert "host" not in merged


def test_get_headers_strips_client_authorization():
    provider = _make_provider()
    merged = provider._get_headers(
        headers={"authorization": "Bearer client-key", "X-Custom": "value"}
    )
    assert merged["Authorization"] == "Bearer test-key"
    assert "authorization" not in merged
    assert merged["X-Custom"] == "value"


@pytest.mark.parametrize(
    "user_agent",
    [
        "claude-cli/2.0.37 (external, cli)",
        "Codex-Desktop/26.422.2437.0",
        "GeminiCLI/0.39.0/gemini-2.0-pro (darwin; x64)",
    ],
)
def test_get_headers_preserves_client_key_for_credential_agents(user_agent):
    provider = _make_provider()
    merged = provider._get_headers(
        headers={"authorization": "Bearer client-key", "user-agent": user_agent}
    )
    assert merged["authorization"] == "Bearer client-key"
    assert "Authorization" not in merged


@pytest.mark.asyncio
async def test_chat():
    provider = _make_provider()
    mock_client = mock_http_client(MockAsyncResponse(_chat_response()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Hello"}],
        )
        response = await provider.chat(payload)

    result = jsonable_encoder(response)
    assert result["id"] == "chatcmpl-abc123"
    assert result["choices"][0]["message"]["content"] == "Hello!"
    assert result["usage"]["prompt_tokens"] == 13

    call_args = mock_client.post.call_args
    assert "chat/completions" in str(call_args)


@pytest.mark.asyncio
async def test_chat_stream():
    provider = _make_provider()
    chunk_data = (
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
        b'"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant",'
        b'"content":"Hi"},"finish_reason":null}]}\n\n'
    )
    chunks = [chunk_data, b"data: [DONE]\n\n"]
    mock_client = mock_http_client(MockAsyncStreamingResponse(chunks))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Hello"}],
        )
        responses = [chunk async for chunk in provider.chat_stream(payload)]

    assert len(responses) == 1
    result = jsonable_encoder(responses[0])
    assert result["choices"][0]["delta"]["content"] == "Hi"


@pytest.mark.asyncio
async def test_embeddings():
    provider = _make_provider()
    mock_client = mock_http_client(MockAsyncResponse(_embeddings_response()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        payload = embeddings.RequestPayload(input="Test text")
        response = await provider.embeddings(payload)

    result = jsonable_encoder(response)
    assert result["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert result["usage"]["prompt_tokens"] == 8


@pytest.mark.asyncio
async def test_passthrough_non_streaming():
    provider = _make_provider()
    mock_client = mock_http_client(MockAsyncResponse(_chat_response()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        result = await provider.passthrough(
            action=PassthroughAction.OPENAI_CHAT,
            payload={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert result["id"] == "chatcmpl-abc123"


@pytest.mark.asyncio
async def test_passthrough_streaming():
    provider = _make_provider()
    chunk_data = (
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,'
        b'"model":"test-model","choices":[{"index":0,"delta":{"content":"Hi"},'
        b'"finish_reason":null}]}\n\n'
    )
    chunks = [chunk_data, b"data: [DONE]\n\n"]
    mock_client = mock_http_client(MockAsyncStreamingResponse(chunks))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        result = await provider.passthrough(
            action=PassthroughAction.OPENAI_CHAT,
            payload={
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        collected = [chunk async for chunk in result]

    assert len(collected) > 0


# --- adapter tests ---


def test_chat_to_model_adds_model_name():
    config = _make_endpoint_config()
    result = OpenAICompatibleAdapter.chat_to_model(
        {"messages": [{"role": "user", "content": "Hi"}]}, config
    )
    assert result["model"] == "test-model"
    assert result["messages"] == [{"role": "user", "content": "Hi"}]


def test_model_to_chat():
    config = _make_endpoint_config()
    resp = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    result = OpenAICompatibleAdapter.model_to_chat(resp, config)
    assert isinstance(result, chat.ResponsePayload)
    assert result.choices[0].message.content == "Hello!"


def test_model_to_embeddings():
    config = _make_endpoint_config()
    resp = {
        "data": [{"embedding": [0.1, 0.2], "index": 0}],
        "model": "test-model",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }
    result = OpenAICompatibleAdapter.model_to_embeddings(resp, config)
    assert isinstance(result, embeddings.ResponsePayload)
    assert result.data[0].embedding == [0.1, 0.2]


def test_model_to_chat_with_tool_calls():
    config = _make_endpoint_config()
    resp = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "NYC"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
                "index": 0,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    result = OpenAICompatibleAdapter.model_to_chat(resp, config)
    assert result.choices[0].message.tool_calls[0].id == "call_1"


# --- config tests ---


def test_basic_config():
    config = _OpenAICompatibleConfig(api_key="test-key")
    assert config.api_key == "test-key"
    assert config.api_base is None


def test_config_with_api_base():
    config = _OpenAICompatibleConfig(api_key="test-key", api_base="https://custom.com/v1")
    assert config.api_base == "https://custom.com/v1"
