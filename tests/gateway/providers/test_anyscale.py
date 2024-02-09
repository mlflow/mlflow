from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
from mlflow.gateway.providers.anyscale import AnyscaleProvider
from mlflow.gateway.schemas import chat, completions, embeddings

from tests.gateway.tools import MockAsyncResponse, MockAsyncStreamingResponse, mock_http_client


def chat_config(model: str = "meta-llama/Llama-2-7b-chat-hf"):
    return {
        "name": "chat",
        "route_type": "llm/v1/chat",
        "model": {
            "provider": "anyscale",
            "name": model,
            "config": {"anyscale_api_key": "key", "anyscale_api_base": "$ANYSCALE_API_BASE"},
        },
    }


def chat_response():
    return {
        "id": "chatcmpl-abc123",
        "object": "text_completion",
        "created": 1677858242,
        "model": "meta-llama/Llama-2-7b-chat-hf",
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


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ("", "https://api.endpoints.anyscale.com/v1"),
        ("https://endpoint.com/v1", "https://endpoint.com/v1"),
    ],
)
def test_config(env, expected, monkeypatch):
    config = chat_config()
    monkeypatch.setenv("ANYSCALE_API_BASE", env)
    provider = AnyscaleProvider(RouteConfig(**config))
    assert provider.base_url == expected


@pytest.mark.asyncio
async def test_chat():
    resp = chat_response()
    config = chat_config()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        provider = AnyscaleProvider(RouteConfig(**config))
        payload = {"messages": [{"role": "user", "content": "Tell me a joke"}], "temperature": 0.5}
        response = await provider.chat(chat.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "meta-llama/Llama-2-7b-chat-hf",
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
            "https://api.endpoints.anyscale.com/v1/chat/completions",
            json={
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "temperature": 0.5,
                "n": 1,
                **payload,
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


def chat_stream_response():
    return [
        b'data: {"id":"test-id","object":"text_completion","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant"}}]}\n',
        b"\n",
        b'data: {"id":"test-id","object":"text_completion","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":null,"delta":{"content":"test"}}]}\n',
        b"\n",
        b'data: {"id":"test-id","object":"text_completion","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":"stop","delta":{}}]}\n',
        b"\n",
        b"data: [DONE]\n",
    ]


def chat_stream_response_incomplete():
    return [
        # contains first half of a chunk
        b'data: {"id":"test-id","object":"text_completion","created":1,"model":"test","choi'
        # contains second half of first chunk and first half of second chunk
        b'ces":[{"index":0,"finish_reason":null,"delta":{"role":"assistant"}}]}\n\n'
        b'data: {"id":"test-id","object":"text_completion","created":1,"model":"te',
        # contains second half of second chunk
        b'st","choices":[{"index":0,"finish_reason":null,"delta":{"content":"test"}}]}\n',
        b"\n",
        b'data: {"id":"test-id","object":"text_completion","created":1,"model":"test",'
        b'"choices":[{"index":0,"finish_reason":"stop","delta":{}}]}\n',
        b"\n",
        b"data: [DONE]\n",
    ]


@pytest.mark.parametrize("resp", [chat_stream_response(), chat_stream_response_incomplete()])
@pytest.mark.asyncio
async def test_chat_stream(resp):
    config = chat_config()
    mock_client = mock_http_client(MockAsyncStreamingResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        provider = AnyscaleProvider(RouteConfig(**config))
        payload = {"messages": [{"role": "user", "content": "Tell me a joke", "stream": True}]}
        response = provider.chat_stream(chat.RequestPayload(**payload))

        chunks = [jsonable_encoder(chunk) async for chunk in response]
        assert chunks == [
            {
                "choices": [
                    {
                        "delta": {"content": None, "role": "assistant"},
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
                    {"delta": {"content": "test", "role": None}, "finish_reason": None, "index": 0}
                ],
                "created": 1,
                "id": "test-id",
                "model": "test",
                "object": "chat.completion.chunk",
            },
            {
                "choices": [
                    {"delta": {"content": None, "role": None}, "finish_reason": "stop", "index": 0}
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
            "https://api.endpoints.anyscale.com/v1/chat/completions",
            json={
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "temperature": 0,
                "n": 1,
                **payload,
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


def embedding_config():
    return {
        "name": "embeddings",
        "route_type": "llm/v1/embeddings",
        "model": {
            "provider": "anyscale",
            "name": "BAAI/bge-large-en-v1.5",
            "config": {
                "anyscale_api_key": "key",
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
        "model": "BAAI/bge-large-en-v1.5",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
        "headers": {"Content-Type": "application/json"},
    }
    config = embedding_config()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        provider = AnyscaleProvider(RouteConfig(**config))
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
            "model": "BAAI/bge-large-en-v1.5",
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }
        mock_build_client.assert_called_once_with(
            headers={
                "Authorization": "Bearer key",
            }
        )
        mock_client.post.assert_called_once_with(
            "https://api.endpoints.anyscale.com/v1/embeddings",
            json={"model": "BAAI/bge-large-en-v1.5", "input": "This is a test"},
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
                "index": 1,
            },
        ],
        "model": "BAAI/bge-large-en-v1.5",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
        "headers": {"Content-Type": "application/json"},
    }
    config = embedding_config()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        provider = AnyscaleProvider(RouteConfig(**config))
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
            "model": "BAAI/bge-large-en-v1.5",
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }
        mock_build_client.assert_called_once_with(
            headers={
                "Authorization": "Bearer key",
            }
        )
        mock_client.post.assert_called_once_with(
            "https://api.endpoints.anyscale.com/v1/embeddings",
            json={
                "model": "BAAI/bge-large-en-v1.5",
                "input": ["1", "2"],
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.asyncio
async def test_param_model_is_not_permitted():
    config = chat_config("some-model")
    provider = AnyscaleProvider(RouteConfig(**config))
    payload = {
        "prompt": "This should fail",
        "max_tokens": 5000,
        "model": "something-else",
    }
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.chat(completions.RequestPayload(**payload))
    assert "The parameter 'model' is not permitted" in e.value.detail
    assert e.value.status_code == 422
