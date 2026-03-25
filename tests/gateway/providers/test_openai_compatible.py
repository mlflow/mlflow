from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import EndpointConfig, OpenAICompatibleConfig
from mlflow.gateway.providers.base import PassthroughAction
from mlflow.gateway.providers.openai_compatible import (
    OpenAICompatibleAdapter,
    OpenAICompatibleProvider,
)
from mlflow.gateway.schemas import chat, completions, embeddings

from tests.gateway.tools import (
    MockAsyncResponse,
    MockAsyncStreamingResponse,
    mock_http_client,
)


# --- Concrete subclass for testing ---


class _TestConfig(OpenAICompatibleConfig):
    pass


class _TestProvider(OpenAICompatibleProvider):
    NAME = "TestProvider"
    CONFIG_TYPE = _TestConfig
    DEFAULT_API_BASE = "https://api.test-provider.com/v1"


# --- fixtures ---


def _make_provider(*, api_base: str | None = None) -> _TestProvider:
    from mlflow.gateway.provider_registry import provider_registry

    # Temporarily register our test provider so EndpointConfig validation works
    _TEST_PROVIDER_NAME = "_test_openai_compat"
    if _TEST_PROVIDER_NAME not in provider_registry.keys():
        provider_registry.register(_TEST_PROVIDER_NAME, _TestProvider)

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


def _completions_response():
    return {
        "id": "cmpl-abc123",
        "object": "text_completion",
        "created": 1677858242,
        "model": "test-model",
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a test!",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }


# --- tests ---


class TestOpenAICompatibleProvider:
    def test_default_api_base(self):
        provider = _make_provider()
        assert provider._api_base == "https://api.test-provider.com/v1"

    def test_custom_api_base(self):
        provider = _make_provider(api_base="https://custom.example.com/v1")
        assert provider._api_base == "https://custom.example.com/v1"

    def test_headers(self):
        provider = _make_provider()
        assert provider.headers == {"Authorization": "Bearer test-key"}

    def test_get_headers_merges_client_headers(self):
        provider = _make_provider()
        merged = provider._get_headers(headers={"X-Custom": "value", "host": "ignored"})
        assert merged == {"Authorization": "Bearer test-key", "X-Custom": "value"}
        assert "host" not in merged

    @pytest.mark.asyncio
    async def test_chat(self):
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

        # Verify correct URL and payload
        call_args = mock_client.post.call_args
        assert "chat/completions" in str(call_args)

    @pytest.mark.asyncio
    async def test_chat_stream(self):
        provider = _make_provider()
        chunks = [
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock_client = mock_http_client(MockAsyncStreamingResponse(chunks))

        with mock.patch("aiohttp.ClientSession", return_value=mock_client):
            payload = chat.RequestPayload(
                messages=[{"role": "user", "content": "Hello"}],
            )
            responses = []
            async for chunk in provider.chat_stream(payload):
                responses.append(chunk)

        assert len(responses) == 1
        result = jsonable_encoder(responses[0])
        assert result["choices"][0]["delta"]["content"] == "Hi"

    @pytest.mark.asyncio
    async def test_embeddings(self):
        provider = _make_provider()
        mock_client = mock_http_client(MockAsyncResponse(_embeddings_response()))

        with mock.patch("aiohttp.ClientSession", return_value=mock_client):
            payload = embeddings.RequestPayload(input="Test text")
            response = await provider.embeddings(payload)

        result = jsonable_encoder(response)
        assert result["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert result["usage"]["prompt_tokens"] == 8

    @pytest.mark.asyncio
    async def test_completions(self):
        provider = _make_provider()
        mock_client = mock_http_client(MockAsyncResponse(_completions_response()))

        with mock.patch("aiohttp.ClientSession", return_value=mock_client):
            payload = completions.RequestPayload(prompt="Test prompt")
            response = await provider.completions(payload)

        result = jsonable_encoder(response)
        assert result["choices"][0]["text"] == "This is a test!"
        assert result["usage"]["total_tokens"] == 12

    @pytest.mark.asyncio
    async def test_passthrough_non_streaming(self):
        provider = _make_provider()
        mock_client = mock_http_client(MockAsyncResponse(_chat_response()))

        with mock.patch("aiohttp.ClientSession", return_value=mock_client):
            result = await provider.passthrough(
                action=PassthroughAction.OPENAI_CHAT,
                payload={"messages": [{"role": "user", "content": "Hello"}]},
            )

        assert result["id"] == "chatcmpl-abc123"

    @pytest.mark.asyncio
    async def test_passthrough_streaming(self):
        provider = _make_provider()
        chunks = [
            b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"test-model","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        mock_client = mock_http_client(MockAsyncStreamingResponse(chunks))

        with mock.patch("aiohttp.ClientSession", return_value=mock_client):
            result = await provider.passthrough(
                action=PassthroughAction.OPENAI_CHAT,
                payload={
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )
            collected = []
            async for chunk in result:
                collected.append(chunk)

        assert len(collected) > 0


def _make_endpoint_config():
    """Create an EndpointConfig suitable for adapter tests."""
    provider = _make_provider()
    return provider.config


class TestOpenAICompatibleAdapter:
    def test_chat_to_model_adds_model_name(self):
        config = _make_endpoint_config()
        result = OpenAICompatibleAdapter.chat_to_model(
            {"messages": [{"role": "user", "content": "Hi"}]}, config
        )
        assert result["model"] == "test-model"
        assert result["messages"] == [{"role": "user", "content": "Hi"}]

    def test_model_to_chat(self):
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

    def test_model_to_embeddings(self):
        config = _make_endpoint_config()
        resp = {
            "data": [{"embedding": [0.1, 0.2], "index": 0}],
            "model": "test-model",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        result = OpenAICompatibleAdapter.model_to_embeddings(resp, config)
        assert isinstance(result, embeddings.ResponsePayload)
        assert result.data[0].embedding == [0.1, 0.2]

    def test_model_to_chat_with_tool_calls(self):
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


class TestOpenAICompatibleConfig:
    def test_basic_config(self):
        config = OpenAICompatibleConfig(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.api_base is None

    def test_config_with_api_base(self):
        config = OpenAICompatibleConfig(api_key="test-key", api_base="https://custom.com/v1")
        assert config.api_base == "https://custom.com/v1"
