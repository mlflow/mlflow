from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import EndpointConfig, OpenAIConfig
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import PassthroughAction
from mlflow.gateway.providers.openai import OpenAIProvider
from mlflow.gateway.schemas import chat, completions, embeddings

from tests.gateway.tools import (
    MockAsyncResponse,
    MockAsyncStreamingResponse,
    mock_http_client,
)


def chat_config():
    return {
        "name": "chat",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "provider": "openai",
            "name": "gpt-4o-mini",
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
        "model": "gpt-4o-mini",
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


def completions_response():
    return {
        "id": "chatcmpl-abc123",
        "object": "text.completion",
        "created": 1677858242,
        "model": "gpt-4o-mini",
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
        },
        "choices": [
            {
                "text": "\n\nThis is a test!",
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }


async def _run_test_chat(provider):
    resp = chat_response()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        payload = {"messages": [{"role": "user", "content": "Tell me a joke"}], "temperature": 0.5}
        response = await provider.chat(chat.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "\n\nThis is a test!",
                        "tool_calls": None,
                        "refusal": None,
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 7,
                "total_tokens": 20,
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
                "model": "gpt-4o-mini",
                "temperature": 0.5,
                "n": 1,
                **payload,
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.asyncio
async def test_chat():
    config = chat_config()
    provider = OpenAIProvider(EndpointConfig(**config))
    await _run_test_chat(provider)


def chat_stream_response():
    return [
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant"}}]}\n',
        b"\n",
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":null,"delta":{"content":"test"}}]}\n',
        b"\n",
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":"stop","delta":{}}]}\n',
        b"\n",
        b"data: [DONE]\n",
    ]


def chat_stream_response_incomplete():
    return [
        # contains first half of a chunk
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1,"model":"test","choi'
        # contains second half of first chunk and first half of second chunk
        b'ces":[{"index":0,"finish_reason":null,"delta":{"role":"assistant"}}]}\n\n'
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1,"model":"te',
        # contains second half of second chunk
        b'st","choices":[{"index":0,"finish_reason":null,"delta":{"content":"test"}}]}\n',
        b"\n",
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":"stop","delta":{}}]}\n',
        b"\n",
        b"data: [DONE]\n",
    ]


async def _run_test_chat_stream(resp, provider):
    mock_client = mock_http_client(MockAsyncStreamingResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        payload = {"messages": [{"role": "user", "content": "Tell me a joke"}]}
        response = provider.chat_stream(chat.RequestPayload(**payload))

        chunks = [jsonable_encoder(chunk) async for chunk in response]
        assert chunks == [
            {
                "choices": [
                    {
                        "delta": {
                            "content": None,
                            "role": "assistant",
                            "tool_calls": None,
                        },
                        "finish_reason": None,
                        "index": 0,
                    }
                ],
                "created": 1,
                "id": "test-id",
                "model": "test",
                "object": "chat.completion.chunk",
            },
            {
                "choices": [
                    {
                        "delta": {
                            "content": "test",
                            "role": None,
                            "tool_calls": None,
                        },
                        "finish_reason": None,
                        "index": 0,
                    }
                ],
                "created": 1,
                "id": "test-id",
                "model": "test",
                "object": "chat.completion.chunk",
            },
            {
                "choices": [
                    {
                        "delta": {
                            "content": None,
                            "role": None,
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
                "created": 1,
                "id": "test-id",
                "model": "test",
                "object": "chat.completion.chunk",
            },
        ]

        mock_build_client.assert_called_once_with(
            headers={
                "Authorization": "Bearer key",
            }
        )
        mock_client.post.assert_called_once_with(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "temperature": 0,
                "n": 1,
                **payload,
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.parametrize("resp", [chat_stream_response(), chat_stream_response_incomplete()])
@pytest.mark.asyncio
async def test_chat_stream(resp):
    config = chat_config()
    provider = OpenAIProvider(EndpointConfig(**config))
    await _run_test_chat_stream(resp, provider)


@pytest.mark.asyncio
async def test_chat_stream_with_function_calling():
    config = chat_config()
    provider = OpenAIProvider(EndpointConfig(**config))

    resp = [
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant",'
        b'"tool_calls":[{"index":0,"id":"call_001","function":{"name":"get_weather"},'
        b'"type":"function"}]}}]}\n',
        b"\n",
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":null,"delta":{'
        b'"tool_calls":[{"index":0,"function":{"arguments":"{\\"location\\":"'
        b"}}]}}]}\n",
        b"\n",
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":"stop","delta":{'
        b'"tool_calls":[{"index":0,"function":{"arguments":"\\"Singapore\\"}"'
        b"}}]}}]}\n",
        b"\n",
        b"data: [DONE]\n",
    ]
    mock_client = mock_http_client(MockAsyncStreamingResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        payload = {"messages": [{"role": "user", "content": "What's the weather in Singapore?"}]}
        response = provider.chat_stream(chat.RequestPayload(**payload))

        chunks = [jsonable_encoder(chunk) async for chunk in response]
        assert chunks == [
            {
                "id": "test-id",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": None,
                        "delta": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_001",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": None},
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "id": "test-id",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": None,
                        "delta": {
                            "role": None,
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": None,
                                    "type": None,
                                    "function": {"name": None, "arguments": '{"location":'},
                                }
                            ],
                        },
                    }
                ],
            },
            {
                "id": "test-id",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "delta": {
                            "role": None,
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": None,
                                    "type": None,
                                    "function": {"name": None, "arguments": '"Singapore"}'},
                                }
                            ],
                        },
                    }
                ],
            },
        ]


def completions_config():
    return {
        "name": "completions",
        "endpoint_type": "llm/v1/completions",
        "model": {
            "provider": "openai",
            "name": "gpt-4-32k",
            "config": {
                "openai_api_key": "key",
                "openai_organization": "test-organization",
            },
        },
    }


async def _run_test_completions(resp, provider):
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        payload = {
            "prompt": "This is a test",
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": "chatcmpl-abc123",
            "object": "text_completion",
            "created": 1677858242,
            "model": "gpt-4o-mini",
            "choices": [{"text": "\n\nThis is a test!", "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        }
        mock_build_client.assert_called_once_with(
            headers={
                "Authorization": "Bearer key",
                "OpenAI-Organization": "test-organization",
            }
        )
        mock_client.post.assert_called_once_with(
            "https://api.openai.com/v1/completions",
            json={
                "model": "gpt-4-32k",
                "temperature": 0,
                "n": 1,
                "prompt": "This is a test",
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.parametrize("resp", [completions_response(), chat_response()])
@pytest.mark.asyncio
async def test_completions(resp):
    config = completions_config()
    provider = OpenAIProvider(EndpointConfig(**config))
    await _run_test_completions(resp, provider)


@pytest.mark.parametrize("prompt", [{"set1", "set2"}, ["list1"], [1], ["list1", "list2"], [1, 2]])
@pytest.mark.asyncio
async def test_completions_throws_if_prompt_contains_non_string(prompt):
    config = completions_config()
    provider = OpenAIProvider(EndpointConfig(**config))
    payload = {"prompt": prompt}
    with pytest.raises(ValidationError, match=r"prompt"):
        await provider.completions(completions.RequestPayload(**payload))


def completions_stream_response():
    return [
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":null,'
        b'"delta":{"role":"assistant", "content": ""}}]}\n',
        b"\n",
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":null,'
        b'"delta":{"content":"test"}}]}\n',
        b"\n",
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":"length","delta":{}}]}\n',
        b"\n",
        b"data: [DONE]\n",
    ]


def completions_stream_response_incomplete():
    return [
        # contains first half of a chunk
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1,"model":"test","choi',
        # contains second half of first chunk and first half of second chunk
        b'ces":[{"index":0,"finish_reason":null,"delta":{"role":"assistant", '
        b'"content": ""}}]}\n\ndata: {"id":"test-id","object":"chat.comp',
        # contains second half of second chunk
        b'letion.chunk","created":1,"model":"test","choices":[{"index":0,"finish_reason":null,'
        b'"delta":{"content":"test"}}]}\n',
        b"\n",
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":"length","delta":{}}]}\n',
        b"\n",
        b"data: [DONE]\n",
    ]


async def _run_test_completions_stream(resp, provider):
    mock_client = mock_http_client(MockAsyncStreamingResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        payload = {"prompt": "This is a test"}
        response = provider.completions_stream(completions.RequestPayload(**payload))

        chunks = [jsonable_encoder(chunk) async for chunk in response]
        assert chunks == [
            {
                "choices": [
                    {
                        "text": "",
                        "finish_reason": None,
                        "index": 0,
                    }
                ],
                "created": 1,
                "id": "test-id",
                "model": "test",
                "object": "text_completion_chunk",
            },
            {
                "choices": [
                    {
                        "text": "test",
                        "finish_reason": None,
                        "index": 0,
                    }
                ],
                "created": 1,
                "id": "test-id",
                "model": "test",
                "object": "text_completion_chunk",
            },
            {
                "choices": [
                    {
                        "text": None,
                        "finish_reason": "length",
                        "index": 0,
                    }
                ],
                "created": 1,
                "id": "test-id",
                "model": "test",
                "object": "text_completion_chunk",
            },
        ]

        mock_build_client.assert_called_once_with(
            headers={
                "Authorization": "Bearer key",
                "OpenAI-Organization": "test-organization",
            }
        )
        mock_client.post.assert_called_once_with(
            "https://api.openai.com/v1/completions",
            json={
                "model": "gpt-4-32k",
                "temperature": 0,
                "n": 1,
                "prompt": "This is a test",
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.parametrize(
    "resp", [completions_stream_response(), completions_stream_response_incomplete()]
)
@pytest.mark.asyncio
async def test_completions_stream(resp):
    config = completions_config()
    provider = OpenAIProvider(EndpointConfig(**config))
    await _run_test_completions_stream(resp, provider)


def embedding_config():
    return {
        "name": "embeddings",
        "endpoint_type": "llm/v1/embeddings",
        "model": {
            "provider": "openai",
            "name": "text-embedding-ada-002",
            "config": {
                "openai_api_base": "https://api.openai.com/v1",
                "openai_api_key": "key",
            },
        },
    }


async def _run_test_embeddings(provider):
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
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        payload = {"input": "This is a test"}
        response = await provider.embeddings(embeddings.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
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
async def test_embeddings():
    config = embedding_config()
    provider = OpenAIProvider(EndpointConfig(**config))
    await _run_test_embeddings(provider)


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
                "index": 1,
            },
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
        "headers": {"Content-Type": "application/json"},
    }
    config = embedding_config()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        provider = OpenAIProvider(EndpointConfig(**config))
        payload = {"input": ["1", "2"]}
        response = await provider.embeddings(embeddings.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
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
                    "index": 1,
                },
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
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
        "endpoint_type": "llm/v1/completions",
        "model": {
            "provider": "openai",
            "name": "gpt-4o-mini",
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
        provider = OpenAIProvider(EndpointConfig(**config))
        payload = {
            "prompt": "This is a test",
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": "chatcmpl-abc123",
            "object": "text_completion",
            "created": 1677858242,
            "model": "gpt-4o-mini",
            "choices": [{"text": "\n\nThis is a test!", "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        }
        mock_build_client.assert_called_once_with(
            headers={
                "api-key": "key",
            }
        )
        mock_client.post.assert_called_once_with(
            (
                "https://test-azureopenai.openai.azure.com/openai/deployments/test-gpt35"
                "/completions?api-version=2023-05-15"
            ),
            json={
                "temperature": 0,
                "n": 1,
                "prompt": "This is a test",
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.asyncio
async def test_azuread_openai():
    resp = chat_response()
    config = azure_config(api_type="azuread")
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        provider = OpenAIProvider(EndpointConfig(**config))
        payload = {
            "prompt": "This is a test",
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": "chatcmpl-abc123",
            "object": "text_completion",
            "created": 1677858242,
            "model": "gpt-4o-mini",
            "choices": [{"text": "\n\nThis is a test!", "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20},
        }
        mock_build_client.assert_called_once_with(
            headers={
                "Authorization": "Bearer key",
            }
        )
        mock_client.post.assert_called_once_with(
            (
                "https://test-azureopenai.openai.azure.com/openai/deployments/test-gpt35"
                "/completions?api-version=2023-05-15"
            ),
            json={
                "temperature": 0,
                "n": 1,
                "prompt": "This is a test",
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
    route_config = EndpointConfig(
        name="completions",
        endpoint_type="llm/v1/completions",
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
    provider = OpenAIProvider(EndpointConfig(**config))
    payload = {
        "prompt": "This should fail",
        "max_tokens": 5000,
        "model": "something-else",
    }
    with pytest.raises(AIGatewayException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "The parameter 'model' is not permitted" in e.value.detail
    assert e.value.status_code == 422


@pytest.mark.asyncio
async def test_openai_passthrough_chat():
    config = chat_config()
    provider = OpenAIProvider(EndpointConfig(**config))

    # Mock OpenAI API response
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from passthrough!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    with mock.patch(
        "mlflow.gateway.providers.openai.send_request", return_value=mock_response
    ) as mock_send:
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        response = await provider.passthrough(PassthroughAction.OPENAI_CHAT, payload)

        # Verify send_request was called with correct parameters
        assert mock_send.called
        call_kwargs = mock_send.call_args[1]
        assert call_kwargs["path"] == "chat/completions"
        assert call_kwargs["payload"]["model"] == "gpt-4o-mini"
        assert call_kwargs["payload"]["messages"] == [{"role": "user", "content": "Hello"}]

        # Verify response is raw OpenAI format
        assert response == mock_response


@pytest.mark.asyncio
async def test_openai_passthrough_embeddings():
    embeddings_config = {
        "name": "embeddings",
        "endpoint_type": "llm/v1/embeddings",
        "model": {
            "provider": "openai",
            "name": "text-embedding-3-small",
            "config": {
                "openai_api_base": "https://api.openai.com/v1",
                "openai_api_key": "key",
            },
        },
    }
    provider = OpenAIProvider(EndpointConfig(**embeddings_config))

    # Mock OpenAI API response
    mock_response = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }

    with mock.patch(
        "mlflow.gateway.providers.openai.send_request", return_value=mock_response
    ) as mock_send:
        payload = {"input": "Test input"}
        response = await provider.passthrough(PassthroughAction.OPENAI_EMBEDDINGS, payload)

        # Verify send_request was called with correct parameters
        assert mock_send.called
        call_kwargs = mock_send.call_args[1]
        assert call_kwargs["path"] == "embeddings"
        assert call_kwargs["payload"]["model"] == "text-embedding-3-small"
        assert call_kwargs["payload"]["input"] == "Test input"

        # Verify response is raw OpenAI format
        assert response == mock_response


@pytest.mark.asyncio
async def test_openai_passthrough_responses():
    config = chat_config()
    provider = OpenAIProvider(EndpointConfig(**config))

    # Mock OpenAI Responses API response (using correct Responses API schema)
    mock_response = {
        "id": "resp-123",
        "object": "response",
        "created": 1677858242,
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": [{"type": "text", "text": "Response from Responses API"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    with mock.patch(
        "mlflow.gateway.providers.openai.send_request", return_value=mock_response
    ) as mock_send:
        # Responses API uses 'input' and 'instructions' instead of 'messages'
        payload = {
            "input": [{"type": "text", "text": "Hello"}],
            "instructions": "You are a helpful assistant",
            "response_format": {"type": "text"},
        }
        response = await provider.passthrough(PassthroughAction.OPENAI_RESPONSES, payload)

        # Verify send_request was called with correct parameters
        assert mock_send.called
        call_kwargs = mock_send.call_args[1]
        assert call_kwargs["path"] == "responses"
        assert call_kwargs["payload"]["model"] == "gpt-4o-mini"
        assert call_kwargs["payload"]["input"] == [{"type": "text", "text": "Hello"}]
        assert call_kwargs["payload"]["instructions"] == "You are a helpful assistant"

        # Verify response is raw OpenAI Responses API format
        assert response == mock_response


@pytest.mark.asyncio
async def test_azure_openai_passthrough_chat_removes_model():
    azure_chat_config = {
        "name": "chat",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "provider": "openai",
            "name": "gpt-4o-mini",
            "config": {
                "openai_api_type": "azure",
                "openai_api_base": "https://my-org.openai.azure.com/",
                "openai_deployment_name": "my-deployment",
                "openai_api_version": "2023-05-15",
                "openai_api_key": "key",
            },
        },
    }
    provider = OpenAIProvider(EndpointConfig(**azure_chat_config))

    # Mock OpenAI API response
    mock_response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Azure!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    with mock.patch(
        "mlflow.gateway.providers.openai.send_request", return_value=mock_response
    ) as mock_send:
        payload = {"messages": [{"role": "user", "content": "Hello"}]}
        response = await provider.passthrough(PassthroughAction.OPENAI_CHAT, payload)

        # Verify send_request was called
        assert mock_send.called
        call_kwargs = mock_send.call_args[1]
        assert call_kwargs["path"] == "chat/completions"
        # Azure OpenAI should NOT have model in payload
        assert "model" not in call_kwargs["payload"]
        assert call_kwargs["payload"]["messages"] == [{"role": "user", "content": "Hello"}]

        # Verify response is raw OpenAI format
        assert response == mock_response
