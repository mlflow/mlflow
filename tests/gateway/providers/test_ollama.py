from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.ollama import OllamaConfig, OllamaProvider
from mlflow.gateway.schemas import chat, embeddings

from tests.gateway.tools import MockAsyncResponse, mock_http_client


def _make_provider(*, api_base: str | None = None) -> OllamaProvider:
    config = {}
    if api_base is not None:
        config["api_base"] = api_base
    endpoint_config = EndpointConfig(
        name="ollama-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "ollama",
            "name": "llama3.2",
            "config": config,
        },
    )
    return OllamaProvider(endpoint_config)


def _chat_response():
    return {
        "id": "chatcmpl-ollama-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "llama3.2",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from Ollama!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }


def _embeddings_response():
    return {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
        "model": "llama3.2",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
        "headers": {"Content-Type": "application/json"},
    }


def test_default_api_base():
    provider = _make_provider()
    assert provider._api_base == "http://localhost:11434/v1"


def test_custom_api_base():
    provider = _make_provider(api_base="http://my-server:11434/v1")
    assert provider._api_base == "http://my-server:11434/v1"


def test_default_api_key():
    provider = _make_provider()
    assert provider._api_key == "ollama"


def test_name():
    provider = _make_provider()
    assert provider.DISPLAY_NAME == "Ollama"


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
    assert result["id"] == "chatcmpl-ollama-123"
    assert result["choices"][0]["message"]["content"] == "Hello from Ollama!"


@pytest.mark.asyncio
async def test_embeddings():
    provider = _make_provider()
    mock_client = mock_http_client(MockAsyncResponse(_embeddings_response()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        payload = embeddings.RequestPayload(input="Test text")
        response = await provider.embeddings(payload)

    result = jsonable_encoder(response)
    assert result["data"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_config_default_api_key():
    config = OllamaConfig()
    assert config.api_key == "ollama"


def test_config_no_api_key_required():
    config = OllamaConfig(api_base="http://custom:11434/v1")
    assert config.api_key == "ollama"
    assert config.api_base == "http://custom:11434/v1"
