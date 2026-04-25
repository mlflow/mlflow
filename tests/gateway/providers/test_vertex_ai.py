import json
from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.gateway.config import EndpointConfig, VertexAIConfig
from mlflow.gateway.providers.vertex_ai import VertexAIProvider
from mlflow.gateway.schemas import chat

from tests.gateway.tools import MockAsyncResponse, mock_http_client


def _mock_credentials():
    creds = mock.Mock()
    creds.token = "mock-access-token"
    creds.valid = True
    return creds


def _make_provider() -> VertexAIProvider:
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "gemini-2.0-flash",
            "config": {
                "vertex_project": "my-gcp-project",
                "vertex_location": "us-central1",
            },
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    return provider


def _chat_response():
    return {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello from Vertex AI!"}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "totalTokenCount": 30,
        },
        "headers": {"Content-Type": "application/json"},
    }


def test_base_url():
    provider = _make_provider()
    assert provider.base_url == (
        "https://us-central1-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-central1/publishers/google/models"
    )


def test_headers_use_bearer_token():
    provider = _make_provider()
    assert provider.headers == {"Authorization": "Bearer mock-access-token"}


def test_name():
    provider = _make_provider()
    assert provider.DISPLAY_NAME == "Vertex AI"
    assert provider.get_provider_name() == "vertex_ai"


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
    assert result["choices"][0]["message"]["content"] == "Hello from Vertex AI!"
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 20


def test_basic_config():
    config = VertexAIConfig(vertex_project="my-project")
    assert config.vertex_project == "my-project"
    assert config.vertex_location is None
    assert config.vertex_credentials is None


def test_custom_location():
    config = VertexAIConfig(vertex_project="my-project", vertex_location="europe-west4")
    assert config.vertex_location == "europe-west4"


def test_global_endpoint_when_no_location():
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "gemini-2.0-flash",
            "config": {"vertex_project": "my-project"},
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    assert provider.base_url == (
        "https://aiplatform.googleapis.com"
        "/v1/projects/my-project/locations/global/publishers/google/models"
    )


def test_global_location_uses_global_endpoint():
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "gemini-3-pro-preview",
            "config": {
                "vertex_project": "my-gcp-project",
                "vertex_location": "global",
            },
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    assert provider.base_url == (
        "https://aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/global/publishers/google/models"
    )


def test_with_credentials():
    creds_json = json.dumps({"type": "service_account", "project_id": "test"})
    config = VertexAIConfig(vertex_project="my-project", vertex_credentials=creds_json)
    assert config.vertex_credentials == creds_json


def test_project_required():
    with pytest.raises(ValidationError, match="vertex_project"):
        VertexAIConfig()


def test_credentials_with_adc():
    config = VertexAIConfig(vertex_project="my-project")
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "gemini-2.0-flash",
            "config": config.model_dump(),
        },
    )
    provider = VertexAIProvider(endpoint_config)

    mock_creds = _mock_credentials()
    with mock.patch("google.auth.default", return_value=(mock_creds, "my-project")) as mock_default:
        headers = provider.headers
        mock_default.assert_called_once()
        assert headers == {"Authorization": "Bearer mock-access-token"}
