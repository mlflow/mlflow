from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.gateway.config import AzureFoundryConfig, EndpointConfig
from mlflow.gateway.providers.azure_foundry import AzureFoundryProvider
from mlflow.gateway.schemas import chat

from tests.gateway.tools import MockAsyncResponse, mock_http_client


def _make_provider() -> AzureFoundryProvider:
    endpoint_config = EndpointConfig(
        name="azure-foundry-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "azure_foundry",
            "name": "Llama-3.3-70B-Instruct",
            "config": {
                "azure_api_key": "test-api-key",
                "azure_api_base": "https://test-endpoint.example.com",
            },
        },
    )
    return AzureFoundryProvider(endpoint_config)


def _chat_response():
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "Llama-3.3-70B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from Azure AI Foundry!",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


def test_base_url():
    provider = _make_provider()
    assert provider._api_base == "https://test-endpoint.example.com"


def test_headers_use_api_key():
    provider = _make_provider()
    assert provider.headers == {"api-key": "test-api-key"}


def test_name():
    provider = _make_provider()
    assert provider.DISPLAY_NAME == "Azure AI Foundry"
    assert provider.get_provider_name() == "azure_foundry"


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
    assert result["choices"][0]["message"]["content"] == "Hello from Azure AI Foundry!"
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 20


def test_basic_config():
    config = AzureFoundryConfig(
        azure_api_key="my-key",
        azure_api_base="https://test-endpoint.example.com",
    )
    assert config.azure_api_key == "my-key"
    assert config.azure_api_base == "https://test-endpoint.example.com"


def test_api_key_required():
    with pytest.raises(ValidationError, match="azure_api_key"):
        AzureFoundryConfig(azure_api_base="https://test-endpoint.example.com")


def test_api_base_required():
    with pytest.raises(ValidationError, match="azure_api_base"):
        AzureFoundryConfig(azure_api_key="my-key")


def test_endpoint_url():
    provider = _make_provider()
    url = provider.get_endpoint_url("llm/v1/chat")
    assert url == "https://test-endpoint.example.com/chat/completions"


@pytest.mark.asyncio
async def test_chat_does_not_inject_model_in_payload():
    provider = _make_provider()
    mock_client = mock_http_client(MockAsyncResponse(_chat_response()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_session:
        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Hello"}],
        )
        await provider.chat(payload)

    # Verify the payload sent to the API does not contain the model field
    call_args = mock_session.return_value.post.call_args
    sent_payload = call_args.kwargs.get("json", {})
    assert "model" not in sent_payload
