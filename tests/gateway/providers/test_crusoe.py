from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.crusoe import CrusoeProvider
from mlflow.gateway.schemas import chat

from tests.gateway.tools import MockAsyncResponse, mock_http_client


def _make_provider() -> CrusoeProvider:
    endpoint_config = EndpointConfig(
        name="crusoe-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "crusoe",
            "name": "meta-llama/Llama-3.3-70B-Instruct",
            "config": {"api_key": "crusoe-test-key"},
        },
    )
    return CrusoeProvider(endpoint_config)


def _chat_response():
    return {
        "id": "chatcmpl-crusoe-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from Crusoe!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }


def test_default_api_base():
    provider = _make_provider()
    assert provider._api_base == "https://api.inference.crusoecloud.com/v1"


def test_headers():
    provider = _make_provider()
    assert provider.headers == {"Authorization": "Bearer crusoe-test-key"}


def test_name():
    provider = _make_provider()
    assert provider.DISPLAY_NAME == "Crusoe"


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
    assert result["id"] == "chatcmpl-crusoe-123"
    assert result["choices"][0]["message"]["content"] == "Hello from Crusoe!"


def test_custom_api_base_overrides_default():
    endpoint_config = EndpointConfig(
        name="crusoe-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "crusoe",
            "name": "meta-llama/Llama-3.3-70B-Instruct",
            "config": {
                "api_key": "crusoe-test-key",
                "api_base": "https://custom.crusoecloud.com/v1",
            },
        },
    )
    provider = CrusoeProvider(endpoint_config)
    assert provider._api_base == "https://custom.crusoecloud.com/v1"
