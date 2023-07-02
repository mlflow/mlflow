from unittest import mock

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
import pytest

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import OpenAIConfig
from mlflow.gateway.providers.openai import OpenAIProvider
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.config import RouteConfig
from tests.gateway.tools import MockAsyncResponse, mock_http_client


def chat_config():
    return {
        "name": "chat",
        "route_type": "llm/v1/chat",
        "model": {
            "provider": "openai",
            "name": "gpt-3.5-turbo",
            "config": {
                "openai_api_base": "https://api.openai.com/v1",
                "openai_api_key": "key",
            },
        },
    }


def chat_response():
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-3.5-turbo-0301",
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "\n\nThis is a test!",
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }


@pytest.mark.asyncio
async def test_chat():
    resp = chat_response()
    config = chat_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = OpenAIProvider(RouteConfig(**config))
        payload = {"messages": [{"role": "user", "content": "Tell me a joke"}]}
        response = await provider.chat(chat.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "candidates": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "\n\nThis is a test!",
                    },
                    "metadata": {
                        "finish_reason": "stop",
                    },
                }
            ],
            "metadata": {
                "input_tokens": 13,
                "output_tokens": 7,
                "total_tokens": 20,
                "model": "gpt-3.5-turbo-0301",
                "route_type": "llm/v1/chat",
            },
        }
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_chat_throws_if_request_payload_contains_n():
    config = chat_config()
    provider = OpenAIProvider(RouteConfig(**config))
    payload = {"messages": [{"role": "user", "content": "Tell me a joke"}], "n": 1}
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.chat(chat.RequestPayload(**payload))
    assert "Invalid parameter `n`" in e.value.detail


@pytest.mark.asyncio
async def test_chat_temperature_is_doubled():
    resp = chat_response()
    config = chat_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = OpenAIProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
            "temperature": 0.5,
        }
        await provider.chat(completions.RequestPayload(**payload))
        assert mock_post.call_args[1]["json"]["temperature"] == 0.5 * 2


def completions_config():
    return {
        "name": "completions",
        "route_type": "llm/v1/completions",
        "model": {
            "provider": "openai",
            "name": "text-davinci-003",
            "config": {
                "openai_api_base": "https://api.openai.com/v1",
                "openai_api_key": "key",
                "openai_organization": "test-organization",
            },
        },
    }


@pytest.mark.asyncio
async def test_completions():
    resp = chat_response()
    config = completions_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = OpenAIProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "candidates": [{"text": "\n\nThis is a test!", "metadata": {"finish_reason": "stop"}}],
            "metadata": {
                "input_tokens": 13,
                "output_tokens": 7,
                "total_tokens": 20,
                "model": "gpt-3.5-turbo-0301",
                "route_type": "llm/v1/completions",
            },
        }
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_completions_throws_if_request_payload_contains_n():
    config = chat_config()
    provider = OpenAIProvider(RouteConfig(**config))
    payload = {"prompt": "This is a test", "n": 1}
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "Invalid parameter `n`" in e.value.detail


@pytest.mark.asyncio
async def test_completions_temperature_is_doubled():
    resp = chat_response()
    config = completions_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = OpenAIProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
            "temperature": 0.5,
        }
        await provider.completions(completions.RequestPayload(**payload))
        assert mock_post.call_args[1]["json"]["temperature"] == 0.5 * 2


def embedding_config():
    return {
        "name": "embeddings",
        "route_type": "llm/v1/embeddings",
        "model": {
            "provider": "openai",
            "name": "text-embedding-ada-002",
            "config": {
                "openai_api_base": "https://api.openai.com/v1",
                "openai_api_key": "key",
            },
        },
    }


@pytest.mark.asyncio
async def test_embeddings():
    resp = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [
                    0.0023064255,
                    -0.009327292,
                    -0.0028842222,
                ],
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }
    config = embedding_config()

    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = OpenAIProvider(RouteConfig(**config))
        payload = {"text": "This is a test"}
        response = await provider.embeddings(embeddings.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "embeddings": [
                [
                    0.0023064255,
                    -0.009327292,
                    -0.0028842222,
                ],
            ],
            "metadata": {
                "input_tokens": 8,
                "output_tokens": 0,
                "total_tokens": 8,
                "model": "text-embedding-ada-002",
                "route_type": "llm/v1/embeddings",
            },
        }
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_embeddings_batch_input():
    resp = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [
                    0.1,
                    0.2,
                    0.3,
                ],
                "index": 0,
            },
            {
                "object": "embedding",
                "embedding": [
                    0.4,
                    0.5,
                    0.6,
                ],
                "index": 0,
            },
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }
    config = embedding_config()

    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = OpenAIProvider(RouteConfig(**config))
        payload = {"text": ["1", "2"]}
        response = await provider.embeddings(embeddings.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "embeddings": [
                [
                    0.1,
                    0.2,
                    0.3,
                ],
                [
                    0.4,
                    0.5,
                    0.6,
                ],
            ],
            "metadata": {
                "input_tokens": 8,
                "output_tokens": 0,
                "total_tokens": 8,
                "model": "text-embedding-ada-002",
                "route_type": "llm/v1/embeddings",
            },
        }
        mock_post.assert_called_once()


def azure_config(api_type: str):
    return {
        "name": "completions",
        "route_type": "llm/v1/completions",
        "model": {
            "provider": "openai",
            "name": "gpt-35-turbo",
            "config": {
                "openai_api_type": api_type,
                "openai_api_key": "key",
                "openai_api_base": "https://test-azureopenai.openai.azure.com/",
                "openai_deployment_name": "test-gpt35",
                "openai_api_version": "2023-05-15",
            },
        },
    }


@pytest.mark.asyncio
async def test_azure_openai():
    resp = chat_response()
    config = azure_config(api_type="azure")
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        provider = OpenAIProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "candidates": [{"text": "\n\nThis is a test!", "metadata": {"finish_reason": "stop"}}],
            "metadata": {
                "input_tokens": 13,
                "output_tokens": 7,
                "total_tokens": 20,
                "model": "gpt-3.5-turbo-0301",
                "route_type": "llm/v1/completions",
            },
        }
        mock_build_client.assert_called_once_with(
            headers={
                "api-key": "key",
            }
        )
        mock_client.post.assert_called_once_with(
            (
                "https://test-azureopenai.openai.azure.com/openai/deployments/test-gpt35"
                "/chat/completions?api-version=2023-05-15"
            ),
            json=mock.ANY,
        )


@pytest.mark.asyncio
async def test_azuread_openai():
    resp = chat_response()
    config = azure_config(api_type="azuread")
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        provider = OpenAIProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "candidates": [{"text": "\n\nThis is a test!", "metadata": {"finish_reason": "stop"}}],
            "metadata": {
                "input_tokens": 13,
                "output_tokens": 7,
                "total_tokens": 20,
                "model": "gpt-3.5-turbo-0301",
                "route_type": "llm/v1/completions",
            },
        }
        mock_build_client.assert_called_once_with(
            headers={
                "Authorization": "Bearer key",
            }
        )
        mock_client.post.assert_called_once_with(
            (
                "https://test-azureopenai.openai.azure.com/openai/deployments/test-gpt35"
                "/chat/completions?api-version=2023-05-15"
            ),
            json=mock.ANY,
        )


@pytest.mark.parametrize(
    ("api_type", "api_base", "deployment_name", "api_version", "exception_type"),
    [
        # Invalid API Types
        ("invalidtype", None, None, None, MlflowException),
        (0, None, None, None, MlflowException),
        # OpenAI API type
        ("openai", None, None, None, None),
        ("openai", "https://api.openai.com/v1", None, None, None),
        ("openai", "https://api.openai.com/v1", None, "2023-05-15", None),
        ("openAI", "https://api.openai.com/v1", None, "2023-05-15", None),
        # Deployment name is specified when API type is not 'azure' or 'azuread'
        ("openai", None, "mock-deployment", None, MlflowException),
        # Azure API type
        ("azure", "https://test-azureopenai.openai.azure.com", "mock-dep", "2023-05-15", None),
        ("AZURe", "https://test-azureopenai.openai.azure.com", "mock-dep", "2023-04-12", None),
        # Missing required API base, deployment name, and / or api version fields
        ("azure", None, None, None, MlflowException),
        ("azure", "https://test-azureopenai.openai.azure.com", "mock-dep", None, MlflowException),
        # AzureAD API type
        ("azuread", "https://test-azureopenai.openai.azure.com", "mock-dep", "2023-05-15", None),
        ("azureAD", "https://test-azureopenai.openai.azure.com", "mock-dep", "2023-04-12", None),
        # Missing required API base, deployment name, and / or api version fields
        ("azuread", None, None, None, MlflowException),
        ("azuread", "https://test-azureopenai.openai.azure.com", "mock-dep", None, MlflowException),
    ],
)
def test_openai_provider_validates_openai_config(
    api_type,
    api_base,
    deployment_name,
    api_version,
    exception_type,
):
    openai_config = OpenAIConfig(
        openai_api_key="mock-api-key",
        openai_api_type=api_type,
        openai_api_base=api_base,
        openai_deployment_name=deployment_name,
        openai_api_version=api_version,
    )
    route_config = RouteConfig(
        name="completions",
        route_type="llm/v1/completions",
        model={
            "provider": "openai",
            "name": "text-davinci-003",
            "config": dict(openai_config),
        },
    )
    if exception_type is not None:
        with pytest.raises(exception_type, match="OpenAI"):
            OpenAIProvider(route_config)
    else:
        provider = OpenAIProvider(route_config)
        assert provider.openai_config == openai_config
