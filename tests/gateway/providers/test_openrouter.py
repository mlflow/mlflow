from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.openrouter import OpenRouterProvider
from mlflow.gateway.schemas import chat

from tests.gateway.tools import MockAsyncResponse, mock_http_client


def _make_provider() -> OpenRouterProvider:
    endpoint_config = EndpointConfig(
        name="openrouter-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "openrouter",
            "name": "anthropic/claude-3.5-sonnet",
            "config": {"api_key": "sk-or-test-key"},
        },
    )
    return OpenRouterProvider(endpoint_config)


def _chat_response():
    return {
        "id": "chatcmpl-or-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "anthropic/claude-3.5-sonnet",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from OpenRouter!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }


def test_default_api_base():
    provider = _make_provider()
    assert provider._api_base == "https://openrouter.ai/api/v1"


def test_headers():
    provider = _make_provider()
    assert provider.headers == {"Authorization": "Bearer sk-or-test-key"}


def test_name():
    provider = _make_provider()
    assert provider.NAME == "OpenRouter"


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
    assert result["id"] == "chatcmpl-or-123"
    assert result["choices"][0]["message"]["content"] == "Hello from OpenRouter!"
