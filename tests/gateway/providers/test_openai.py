from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import OpenAIConfig, RouteConfig
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
from mlflow.gateway.providers.openai import OpenAIProvider
from mlflow.gateway.schemas import chat, completions, embeddings

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
        "headers": {"Content-Type": "application/json"},
    }


@pytest.mark.asyncio
async def test_chat():
    resp = chat_response()
    config = chat_config()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
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
        mock_build_client.assert_called_once_with(
            headers={
                "Authorization": "Bearer key",
            }
        )
        mock_client.post.assert_called_once_with(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-3.5-turbo",
                "temperature": 0,
                "n": 1,
                **payload,
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


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
            "name": "gpt-4-32k",
            "config": {
                "openai_api_key": "key",
                "openai_organization": "test-organization",
            },
        },
    }


@pytest.mark.asyncio
async def test_completions():
    resp = chat_response()
    config = completions_config()
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
                "OpenAI-Organization": "test-organization",
            }
        )
        mock_client.post.assert_called_once_with(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4-32k",
                "temperature": 0,
                "n": 1,
                "messages": [{"role": "user", "content": "This is a test"}],
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.asyncio
async def test_completions_throws_if_request_payload_contains_n():
    config = completions_config()
    provider = OpenAIProvider(RouteConfig(**config))
    payload = {"prompt": "This is a test", "n": 1}
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "Invalid parameter `n`" in e.value.detail


@pytest.mark.parametrize("prompt", [{"set1", "set2"}, ["list1"], [1], ["list1", "list2"], [1, 2]])
@pytest.mark.asyncio
async def test_completions_throws_if_prompt_contains_non_string(prompt):
    config = completions_config()
    provider = OpenAIProvider(RouteConfig(**config))
    payload = {"prompt": prompt}
    with pytest.raises(ValidationError, match=r"prompt"):
        await provider.completions(completions.RequestPayload(**payload))


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
        "headers": {"Content-Type": "application/json"},
    }
    config = embedding_config()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
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
        mock_build_client.assert_called_once_with(
            headers={
                "Authorization": "Bearer key",
            }
        )
        mock_client.post.assert_called_once_with(
            "https://api.openai.com/v1/embeddings",
            json={"model": "text-embedding-ada-002", "input": "This is a test"},
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


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
        "headers": {"Content-Type": "application/json"},
    }
    config = embedding_config()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
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
        mock_build_client.assert_called_once_with(
            headers={
                "Authorization": "Bearer key",
            }
        )
        mock_client.post.assert_called_once_with(
            "https://api.openai.com/v1/embeddings",
            json={
                "model": "text-embedding-ada-002",
                "input": ["1", "2"],
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


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
            json={
                "temperature": 0,
                "n": 1,
                "messages": [{"role": "user", "content": "This is a test"}],
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
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
            json={
                "temperature": 0,
                "n": 1,
                "messages": [{"role": "user", "content": "This is a test"}],
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.parametrize(
    ("api_type", "api_base", "deployment_name", "api_version", "organization"),
    [
        # OpenAI API type
        ("openai", None, None, None, None),
        ("openai", "https://api.openai.com/v1", None, None, None),
        ("openai", "https://api.openai.com/v1", None, "2023-05-15", None),
        ("openAI", "https://api.openai.com/v1", None, "2023-05-15", None),
        ("openAI", "https://api.openai.com/v1", None, "2023-05-15", "test-organization"),
        # Azure API type
        ("azure", "https://test.openai.azure.com", "mock-dep", "2023-05-15", None),
        ("AZURe", "https://test.openai.azure.com", "mock-dep", "2023-04-12", None),
        # AzureAD API type
        ("azuread", "https://test.openai.azure.com", "mock-dep", "2023-05-15", None),
        ("azureAD", "https://test.openai.azure.com", "mock-dep", "2023-04-12", None),
    ],
)
def test_openai_provider_can_be_constructed_with_valid_configs(
    api_type,
    api_base,
    deployment_name,
    api_version,
    organization,
):
    openai_config = OpenAIConfig(
        openai_api_key="mock-api-key",
        openai_api_type=api_type,
        openai_api_base=api_base,
        openai_deployment_name=deployment_name,
        openai_api_version=api_version,
        openai_organization=organization,
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
    provider = OpenAIProvider(route_config)
    assert provider.openai_config == openai_config


@pytest.mark.parametrize(
    ("api_type", "api_base", "deployment_name", "api_version", "organization"),
    [
        # Invalid API Type
        ("invalidtype", None, None, None, None),
        # Deployment name is specified when API type is not 'azure' or 'azuread'
        ("openai", None, "mock-deployment", None, None),
        # Missing required API base, deployment name, and / or api version fields
        ("azure", None, None, None, None),
        ("azure", "https://test.openai.azure.com", "mock-dep", None, None),
        # Organization is specified when API type is not 'openai'
        ("azure", "https://test.openai.azure.com", "mock-dep", "2023", "test-org"),
        # Missing required API base, deployment name, and / or api version fields
        ("azuread", None, None, None, None),
        ("azuread", "https://test.openai.azure.com", "mock-dep", None, None),
        # Organization is specified when API type is not 'openai'
        ("azuread", "https://test.openai.azure.com", "mock", "2023", "test-org"),
    ],
)
def test_invalid_openai_configs_throw_on_construction(
    api_type,
    api_base,
    deployment_name,
    api_version,
    organization,
):
    with pytest.raises(MlflowException, match="OpenAI"):
        OpenAIConfig(
            openai_api_key="mock-api-key",
            openai_api_type=api_type,
            openai_api_base=api_base,
            openai_deployment_name=deployment_name,
            openai_api_version=api_version,
            openai_organization=organization,
        )


@pytest.mark.asyncio
async def test_param_model_is_not_permitted():
    config = azure_config("azuread")
    provider = OpenAIProvider(RouteConfig(**config))
    payload = {
        "prompt": "This should fail",
        "max_tokens": 5000,
        "model": "something-else",
    }
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "The parameter 'model' is not permitted" in e.value.detail
    assert e.value.status_code == 422
