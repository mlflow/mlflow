from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import EndpointConfig, PortkeyConfig
from mlflow.gateway.providers.portkey import PortkeyProvider
from mlflow.gateway.schemas import chat

from tests.gateway.tools import MockAsyncResponse, mock_http_client


def _make_provider(provider_config=None, model_name="gpt-4o") -> PortkeyProvider:
    endpoint_config = EndpointConfig(
        name="portkey-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "portkey",
            "name": model_name,
            "config": {"api_key": "pk-test-key", **(provider_config or {})},
        },
    )
    return PortkeyProvider(endpoint_config)


def _chat_response():
    return {
        "id": "chatcmpl-pk-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from Portkey!"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }


def test_name():
    provider = _make_provider()
    assert provider.DISPLAY_NAME == "Portkey"


def test_config_type():
    provider = _make_provider()
    assert isinstance(provider.config.model.config, PortkeyConfig)


def test_default_api_base():
    provider = _make_provider()
    assert provider._api_base == "https://api.portkey.ai/v1"


def test_api_base_override():
    provider = _make_provider({"api_base": "https://portkey.internal.example.com/v1"})
    assert provider._api_base == "https://portkey.internal.example.com/v1"


@pytest.mark.parametrize(
    ("provider_config", "expected_headers"),
    [
        # Model Catalog routing via "@provider/model" names needs no extra headers
        (
            {},
            {"x-portkey-api-key": "pk-test-key"},
        ),
        # Managed provider from the Portkey Model Catalog
        (
            {"portkey_provider": "@openai-prod"},
            {"x-portkey-api-key": "pk-test-key", "x-portkey-provider": "@openai-prod"},
        ),
        # Saved Portkey config ID
        (
            {"portkey_config": "pc-test-1234"},
            {"x-portkey-api-key": "pk-test-key", "x-portkey-config": "pc-test-1234"},
        ),
        # Bare provider slug with the upstream key passed through
        (
            {"portkey_provider": "openai", "provider_api_key": "sk-upstream"},
            {
                "x-portkey-api-key": "pk-test-key",
                "x-portkey-provider": "openai",
                "Authorization": "Bearer sk-upstream",
            },
        ),
    ],
)
def test_headers(provider_config, expected_headers):
    provider = _make_provider(provider_config)
    assert provider.headers == expected_headers


@pytest.mark.asyncio
async def test_chat():
    provider = _make_provider({"portkey_provider": "@openai-prod"})
    mock_client = mock_http_client(MockAsyncResponse(_chat_response()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_session:
        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Hello"}],
        )
        response = await provider.chat(payload)

    result = jsonable_encoder(response)
    assert result["id"] == "chatcmpl-pk-123"
    assert result["choices"][0]["message"]["content"] == "Hello from Portkey!"

    session_headers = mock_session.call_args.kwargs["headers"]
    assert session_headers["x-portkey-api-key"] == "pk-test-key"
    assert session_headers["x-portkey-provider"] == "@openai-prod"

    mock_client.post.assert_called_once()
    assert mock_client.post.call_args.args[0] == "https://api.portkey.ai/v1/chat/completions"


def test_configured_upstream_key_wins_over_client_auth():
    # A subscription tool (e.g. Claude Code) sends its own Authorization header
    # on a passthrough request. For Portkey, Authorization carries the upstream
    # `provider_api_key`, so a configured key must win and the client's key,
    # which targets MLflow rather than the upstream, must not be forwarded.
    provider = _make_provider({"portkey_provider": "openai", "provider_api_key": "sk-upstream"})
    client_headers = {
        "user-agent": "claude-cli/1.2.3",
        "authorization": "Bearer sk-client-token",
    }
    result = provider._get_headers(client_headers)
    auth = [v for k, v in result.items() if k.lower() == "authorization"]
    # exactly one Authorization header, and it is the configured upstream key
    assert auth == ["Bearer sk-upstream"]
    assert result["x-portkey-api-key"] == "pk-test-key"
    assert result["x-portkey-provider"] == "openai"


def test_client_auth_preserved_when_no_upstream_key():
    # Without a configured provider_api_key (e.g. Model Catalog @slug routing),
    # the base behavior is preserved: a subscription tool may pass its own
    # Authorization through.
    provider = _make_provider({"portkey_provider": "@openai-prod"})
    client_headers = {
        "user-agent": "claude-cli/1.2.3",
        "authorization": "Bearer sk-client-token",
    }
    result = provider._get_headers(client_headers)
    auth = [v for k, v in result.items() if k.lower() == "authorization"]
    assert auth == ["Bearer sk-client-token"]
    assert result["x-portkey-api-key"] == "pk-test-key"
