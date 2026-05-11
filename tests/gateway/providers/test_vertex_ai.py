import json
from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.gateway.config import EndpointConfig, VertexAIConfig
from mlflow.gateway.constants import (
    MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
    MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS,
)
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.vertex_ai import VertexAIProvider
from mlflow.gateway.schemas import chat, completions

from tests.gateway.tools import MockAsyncResponse, MockAsyncStreamingResponse, mock_http_client


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


def test_anthropic_model_uses_anthropic_publisher():
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "claude-sonnet-4-5@20251101",
            "config": {
                "vertex_project": "my-gcp-project",
                "vertex_location": "us-east5",
            },
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    assert provider.base_url == (
        "https://us-east5-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-east5/publishers/anthropic/models"
    )


def test_anthropic_model_global_location():
    endpoint_config = EndpointConfig(
        name="vertex-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "claude-opus-4-5@20251101",
            "config": {
                "vertex_project": "my-project",
                "vertex_location": "global",
            },
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    assert provider.base_url == (
        "https://aiplatform.googleapis.com"
        "/v1/projects/my-project/locations/global/publishers/anthropic/models"
    )


def _make_claude_provider() -> VertexAIProvider:
    endpoint_config = EndpointConfig(
        name="vertex-claude-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": "claude-sonnet-4-5@20251101",
            "config": {
                "vertex_project": "my-gcp-project",
                "vertex_location": "us-east5",
            },
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    return provider


def _claude_chat_response():
    return {
        "content": [{"text": "Hello from Claude on Vertex AI!", "type": "text"}],
        "id": "msg-vertex-test",
        "model": "claude-sonnet-4-5@20251101",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 10, "output_tokens": 15},
    }


@pytest.mark.asyncio
async def test_claude_chat_uses_raw_predict_endpoint():
    provider = _make_claude_provider()
    resp = _claude_chat_response()

    with (
        mock.patch("time.time", return_value=1677858242),
        mock.patch("aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)) as mock_post,
    ):
        payload = chat.RequestPayload(messages=[{"role": "user", "content": "Hello"}])
        response = await provider.chat(payload)

    result = jsonable_encoder(response)
    assert result["choices"][0]["message"]["content"] == "Hello from Claude on Vertex AI!"
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 15

    mock_post.assert_called_once_with(
        "https://us-east5-aiplatform.googleapis.com/v1/projects/my-gcp-project"
        "/locations/us-east5/publishers/anthropic/models"
        "/claude-sonnet-4-5@20251101:rawPredict",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
            "anthropic_version": "vertex-2023-10-16",
        },
        timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
    )


@pytest.mark.asyncio
async def test_claude_chat_stream_uses_stream_raw_predict_endpoint():
    provider = _make_claude_provider()
    stream_data = [
        b"event: message_start\n",
        b'data: {"type": "message_start", "message": {"id": "msg-1", "type": "message", '
        b'"role": "assistant", "content": [], "model": "claude-sonnet-4-5@20251101", '
        b'"stop_reason": null, "stop_sequence": null, '
        b'"usage": {"input_tokens": 5, "output_tokens": 1}}}\n',
        b"\n",
        b"event: content_block_delta\n",
        b'data: {"type": "content_block_delta", "index": 0, '
        b'"delta": {"type": "text_delta", "text": "Hello"}}\n',
        b"\n",
        b"event: message_delta\n",
        b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, '
        b'"usage": {"output_tokens": 1}}\n',
        b"\n",
    ]
    mock_response = MockAsyncStreamingResponse(stream_data)
    mock_client = mock_http_client(mock_response)

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        payload = chat.RequestPayload(messages=[{"role": "user", "content": "Hello"}], stream=True)
        chunks = [chunk async for chunk in provider.chat_stream(payload)]

    assert len(chunks) > 0
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert ":streamRawPredict" in call_kwargs[0][0]
    assert call_kwargs[1]["json"]["anthropic_version"] == "vertex-2023-10-16"
    assert "model" not in call_kwargs[1]["json"]


def _make_maas_provider(model_name: str, location: str = "us-central1") -> VertexAIProvider:
    endpoint_config = EndpointConfig(
        name="vertex-maas-endpoint",
        endpoint_type="llm/v1/chat",
        model={
            "provider": "vertex_ai",
            "name": model_name,
            "config": {
                "vertex_project": "my-gcp-project",
                "vertex_location": location,
            },
        },
    )
    provider = VertexAIProvider(endpoint_config)
    provider._cached_credentials = _mock_credentials()
    return provider


@pytest.mark.parametrize(
    "model_name",
    [
        "meta/llama-3.1-405b-instruct-maas",
        "mistral-large-2411",
        "codestral-2501",
        "jamba-1.5",
        "deepseek-ai/deepseek-r1-0528-maas",
        "xai/grok-4.1-fast-reasoning",
        "qwen/qwen3-235b-a22b-instruct-2507-maas",
    ],
)
def test_maas_model_uses_openapi_endpoint(model_name):
    provider = _make_maas_provider(model_name)
    assert provider._model_type == "maas"
    assert provider._delegate is not None
    assert provider._delegate._api_base == (
        "https://us-central1-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-central1/endpoints/openapi"
    )


@pytest.mark.asyncio
async def test_maas_chat_uses_openai_format():
    provider = _make_maas_provider("meta/llama-3.1-405b-instruct-maas")
    openai_resp = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "meta/llama-3.1-405b-instruct-maas",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Llama on Vertex AI!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }

    with (
        mock.patch(
            "aiohttp.ClientSession.post", return_value=MockAsyncResponse(openai_resp)
        ) as mock_post,
    ):
        payload = chat.RequestPayload(messages=[{"role": "user", "content": "Hello"}])
        response = await provider.chat(payload)

    result = jsonable_encoder(response)
    assert result["choices"][0]["message"]["content"] == "Hello from Llama on Vertex AI!"
    assert result["usage"]["prompt_tokens"] == 10

    mock_post.assert_called_once_with(
        "https://us-central1-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-central1/endpoints/openapi/chat/completions",
        json={
            "model": "meta/llama-3.1-405b-instruct-maas",
            "n": 1,
            "messages": [{"role": "user", "content": "Hello"}],
        },
        timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
    )


def test_claude_get_endpoint_url():
    provider = _make_claude_provider()
    assert provider.get_endpoint_url("llm/v1/chat") == (
        "https://us-east5-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-east5/publishers/anthropic/models"
        "/claude-sonnet-4-5@20251101:rawPredict"
    )


def test_maas_get_endpoint_url():
    provider = _make_maas_provider("meta/llama-3.1-405b-instruct-maas")
    assert provider.get_endpoint_url("llm/v1/chat") == (
        "https://us-central1-aiplatform.googleapis.com"
        "/v1/projects/my-gcp-project/locations/us-central1/endpoints/openapi/chat/completions"
    )


@pytest.mark.asyncio
async def test_claude_completions_raises_gateway_exception():
    provider = _make_claude_provider()
    with pytest.raises(AIGatewayException, match="completions endpoint is not supported"):
        await provider.completions(completions.RequestPayload(prompt="hello", max_tokens=10))


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
