from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.xai import XAIProvider
from mlflow.gateway.schemas import chat

from tests.gateway.tools import MockAsyncResponse, mock_http_client


def _make_provider() -> XAIProvider:
    endpoint_config = EndpointConfig(
        name="xai-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "xai",
            "name": "grok-3",
            "config": {"api_key": "xai-test-key"},
        },
    )
    return XAIProvider(endpoint_config)


def _chat_response():
    return {
        "id": "chatcmpl-xai-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "grok-3",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from Grok!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }


def test_default_api_base():
    provider = _make_provider()
    assert provider._api_base == "https://api.x.ai/v1"


def test_headers():
    provider = _make_provider()
    assert provider.headers == {"Authorization": "Bearer xai-test-key"}


def test_name():
    provider = _make_provider()
    assert provider.DISPLAY_NAME == "xAI"
    assert provider.get_provider_name() == "xai"


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
    assert result["id"] == "chatcmpl-xai-123"
    assert result["choices"][0]["message"]["content"] == "Hello from Grok!"
