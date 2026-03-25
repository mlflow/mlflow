from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.databricks import (
    DatabricksConfig,
    DatabricksOAuthConfig,
    DatabricksProvider,
)
from mlflow.gateway.schemas import chat, embeddings

from tests.gateway.tools import MockAsyncResponse, mock_http_client


def _make_provider(
    *,
    api_base: str = "https://my-workspace.databricks.com",
) -> DatabricksProvider:
    endpoint_config = EndpointConfig(
        name="databricks-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "databricks",
            "name": "databricks-dbrx-instruct",
            "config": {
                "api_key": "dapi-test-key",
                "api_base": api_base,
            },
        },
    )
    return DatabricksProvider(endpoint_config)


def _make_oauth_provider() -> DatabricksProvider:
    oauth_config = DatabricksOAuthConfig(
        api_base="https://my-workspace.databricks.com",
        client_id="test-client-id",
        client_secret="test-client-secret",
    )
    endpoint_config = EndpointConfig(
        name="databricks-oauth-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "databricks",
            "name": "databricks-dbrx-instruct",
            "config": {
                "api_key": "dapi-test-key",
                "api_base": "https://my-workspace.databricks.com",
            },
        },
    )
    # Replace config with OAuth config
    endpoint_config.model.config = oauth_config
    return DatabricksProvider(endpoint_config)


def _chat_response():
    return {
        "id": "chatcmpl-db-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "databricks-dbrx-instruct",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello from Databricks!"},
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
        "model": "bge-large-en",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
        "headers": {"Content-Type": "application/json"},
    }


# --- PAT token tests ---


def test_api_base_normalization():
    provider = _make_provider(api_base="https://my-workspace.databricks.com")
    assert provider._api_base == "https://my-workspace.databricks.com/serving-endpoints"


def test_api_base_already_normalized():
    provider = _make_provider(api_base="https://my-workspace.databricks.com/serving-endpoints")
    assert provider._api_base == "https://my-workspace.databricks.com/serving-endpoints"


def test_headers():
    provider = _make_provider()
    assert provider.headers == {"Authorization": "Bearer dapi-test-key"}


def test_name():
    provider = _make_provider()
    assert provider.NAME == "Databricks"


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
    assert result["id"] == "chatcmpl-db-123"
    assert result["choices"][0]["message"]["content"] == "Hello from Databricks!"


@pytest.mark.asyncio
async def test_embeddings():
    provider = _make_provider()
    mock_client = mock_http_client(MockAsyncResponse(_embeddings_response()))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        payload = embeddings.RequestPayload(input="Test text")
        response = await provider.embeddings(payload)

    result = jsonable_encoder(response)
    assert result["data"][0]["embedding"] == [0.1, 0.2, 0.3]


# --- OAuth M2M tests ---


def test_oauth_token_exchange():
    provider = _make_oauth_provider()
    mock_response = mock.Mock()
    mock_response.json.return_value = {
        "access_token": "oauth-access-token-123",
        "token_type": "Bearer",
        "expires_in": 3600,
    }
    mock_response.raise_for_status = mock.Mock()

    with mock.patch("requests.post", return_value=mock_response) as mock_post:
        headers = provider.headers

    assert headers == {"Authorization": "Bearer oauth-access-token-123"}
    mock_post.assert_called_once_with(
        "https://my-workspace.databricks.com/oidc/v1/token",
        data={"grant_type": "client_credentials", "scope": "all-apis"},
        auth=("test-client-id", "test-client-secret"),
        timeout=30,
    )


def test_oauth_token_caching():
    provider = _make_oauth_provider()
    mock_response = mock.Mock()
    mock_response.json.return_value = {
        "access_token": "cached-token",
        "token_type": "Bearer",
        "expires_in": 3600,
    }
    mock_response.raise_for_status = mock.Mock()

    with mock.patch("requests.post", return_value=mock_response) as mock_post:
        # First call fetches token
        headers1 = provider.headers
        # Second call uses cached token
        headers2 = provider.headers

    assert headers1 == headers2 == {"Authorization": "Bearer cached-token"}
    mock_post.assert_called_once()  # Only one HTTP call


@pytest.mark.asyncio
async def test_oauth_chat():
    provider = _make_oauth_provider()
    mock_token_response = mock.Mock()
    mock_token_response.json.return_value = {
        "access_token": "oauth-token",
        "token_type": "Bearer",
        "expires_in": 3600,
    }
    mock_token_response.raise_for_status = mock.Mock()

    mock_client = mock_http_client(MockAsyncResponse(_chat_response()))

    with (
        mock.patch("requests.post", return_value=mock_token_response),
        mock.patch("aiohttp.ClientSession", return_value=mock_client),
    ):
        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Hello"}],
        )
        response = await provider.chat(payload)

    result = jsonable_encoder(response)
    assert result["choices"][0]["message"]["content"] == "Hello from Databricks!"


def test_oauth_api_base_normalization():
    config = DatabricksOAuthConfig(
        api_base="https://my-workspace.databricks.com",
        client_id="cid",
        client_secret="csecret",
    )
    assert config.api_base == "https://my-workspace.databricks.com/serving-endpoints"


# --- config tests ---


def test_url_normalization():
    config = DatabricksConfig(
        api_key="dapi-test",
        api_base="https://my-workspace.databricks.com",
    )
    assert config.api_base == "https://my-workspace.databricks.com/serving-endpoints"


def test_url_already_has_serving_endpoints():
    config = DatabricksConfig(
        api_key="dapi-test",
        api_base="https://my-workspace.databricks.com/serving-endpoints",
    )
    assert config.api_base == "https://my-workspace.databricks.com/serving-endpoints"


def test_api_base_required():
    with pytest.raises(ValidationError, match="api_base"):
        DatabricksConfig(api_key="dapi-test")
