import math
from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.mistral import MistralProvider
from mlflow.gateway.schemas import chat, completions, embeddings

from tests.gateway.tools import MockAsyncResponse, MockAsyncStreamingResponse, mock_http_client

TEST_STRING = "This is a test"
CONTENT_TYPE = "application/json"
TARGET = "aiohttp.ClientSession.post"


def completions_config():
    return {
        "name": "completions",
        "endpoint_type": "llm/v1/completions",
        "model": {
            "provider": "mistral",
            "name": "mistral-tiny",
            "config": {
                "mistral_api_key": "key",
            },
        },
    }


def completions_response():
    return {
        "id": "string",
        "object": "string",
        "create": "integer",
        "model": "string",
        "choices": [
            {
                "index": "integer",
                "message": {"role": "user", "content": TEST_STRING},
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 9,
            "total_tokens": 18,
        },
    }


@pytest.mark.asyncio
async def test_completions():
    resp = completions_response()
    config = completions_config()
    with (
        mock.patch("time.time", return_value=1677858242),
        mock.patch(TARGET, return_value=MockAsyncResponse(resp)) as mock_post,
    ):
        provider = MistralProvider(EndpointConfig(**config))
        payload = {
            "prompt": TEST_STRING,
            "n": 1,
            "stop": ["foobar"],
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": None,
            "object": "text_completion",
            "created": 1677858242,
            "model": "mistral-tiny",
            "choices": [
                {
                    "text": TEST_STRING,
                    "index": 0,
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 9,
                "total_tokens": 18,
            },
        }
        mock_post.assert_called_once_with(
            "https://api.mistral.ai/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": TEST_STRING}],
                "model": "mistral-tiny",
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.asyncio
async def test_completions_temperature_is_scaled_correctly():
    resp = completions_response()
    config = completions_config()
    with mock.patch(TARGET, return_value=MockAsyncResponse(resp)) as mock_post:
        provider = MistralProvider(EndpointConfig(**config))
        payload = {
            "prompt": TEST_STRING,
            "temperature": 0.5,
        }
        await provider.completions(completions.RequestPayload(**payload))
        assert math.isclose(
            mock_post.call_args[1]["json"]["temperature"], 0.5 * 0.5, rel_tol=1e-09, abs_tol=1e-09
        )


def embeddings_config():
    return {
        "name": "embeddings",
        "endpoint_type": "llm/v1/embeddings",
        "model": {
            "provider": "mistral",
            "name": "mistral-embed",
            "config": {
                "mistral_api_key": "key",
            },
        },
    }


def embeddings_response():
    return {
        "id": "bc57846a-3e56-4327-8acc-588ca1a37b8a",
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [
                    3.25,
                    0.7685547,
                    2.65625,
                    -0.30126953,
                    -2.3554688,
                    1.2597656,
                ],
                "index": 0,
            }
        ],
        "model": "mistral-embed",
        "usage": {"prompt_tokens": None, "total_tokens": None},
    }


def embeddings_batch_response():
    return {
        "id": "bc57846a-3e56-4327-8acc-588ca1a37b8a",
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [
                    3.25,
                    0.7685547,
                    2.65625,
                    -0.30126953,
                    -2.3554688,
                    1.2597656,
                ],
                "index": 0,
            },
            {
                "object": "embedding",
                "embedding": [
                    7.25,
                    0.7685547,
                    4.65625,
                    -0.30126953,
                    -2.3554688,
                    8.2597656,
                ],
                "index": 1,
            },
        ],
        "model": "mistral-embed",
        "usage": {"prompt_tokens": None, "total_tokens": None},
    }


@pytest.mark.asyncio
async def test_embeddings():
    resp = embeddings_response()
    config = embeddings_config()
    with mock.patch(TARGET, return_value=MockAsyncResponse(resp)) as mock_post:
        provider = MistralProvider(EndpointConfig(**config))
        payload = {"input": TEST_STRING}
        response = await provider.embeddings(embeddings.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [
                        3.25,
                        0.7685547,
                        2.65625,
                        -0.30126953,
                        -2.3554688,
                        1.2597656,
                    ],
                    "index": 0,
                }
            ],
            "model": "mistral-embed",
            "usage": {"prompt_tokens": None, "total_tokens": None},
        }
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_batch_embeddings():
    resp = embeddings_batch_response()
    config = embeddings_config()
    with mock.patch(TARGET, return_value=MockAsyncResponse(resp)) as mock_post:
        provider = MistralProvider(EndpointConfig(**config))
        payload = {"input": ["This is a", "batch test"]}
        response = await provider.embeddings(embeddings.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [
                        3.25,
                        0.7685547,
                        2.65625,
                        -0.30126953,
                        -2.3554688,
                        1.2597656,
                    ],
                    "index": 0,
                },
                {
                    "object": "embedding",
                    "embedding": [
                        7.25,
                        0.7685547,
                        4.65625,
                        -0.30126953,
                        -2.3554688,
                        8.2597656,
                    ],
                    "index": 1,
                },
            ],
            "model": "mistral-embed",
            "usage": {"prompt_tokens": None, "total_tokens": None},
        }
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_param_model_is_not_permitted():
    config = embeddings_config()
    provider = MistralProvider(EndpointConfig(**config))
    payload = {
        "prompt": "This should fail",
        "max_tokens": 5000,
        "model": "something-else",
    }
    with pytest.raises(AIGatewayException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "The parameter 'model' is not permitted" in e.value.detail
    assert e.value.status_code == 422


@pytest.mark.parametrize("prompt", [{"set1", "set2"}, ["list1"], [1], ["list1", "list2"], [1, 2]])
@pytest.mark.asyncio
async def test_completions_throws_if_prompt_contains_non_string(prompt):
    config = completions_config()
    provider = MistralProvider(EndpointConfig(**config))
    payload = {"prompt": prompt}
    with pytest.raises(ValidationError, match=r"prompt"):
        await provider.completions(completions.RequestPayload(**payload))


def chat_config():
    return {
        "name": "chat",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "provider": "mistral",
            "name": "mistral-large-latest",
            "config": {
                "mistral_api_key": "key",
            },
        },
    }


@pytest.mark.asyncio
async def test_chat_with_structured_output():
    config = chat_config()
    provider = MistralProvider(EndpointConfig(**config))

    json_schema = {
        "name": "user_info",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
            "required": ["name", "email"],
            "additionalProperties": False,
        },
    }

    resp = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1677858242,
        "model": "mistral-large-latest",
        "usage": {
            "prompt_tokens": 13,
            "completion_tokens": 50,
            "total_tokens": 63,
        },
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"name": "John Doe", "email": "john@example.com"}',
                },
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }

    with mock.patch(TARGET, return_value=MockAsyncResponse(resp)) as mock_post:
        payload = {
            "messages": [{"role": "user", "content": "Extract user info"}],
            "response_format": {"type": "json_schema", "json_schema": json_schema},
        }
        response = await provider.chat(chat.RequestPayload(**payload))

        assert (
            response.choices[0].message.content
            == '{"name": "John Doe", "email": "john@example.com"}'
        )
        assert response.choices[0].finish_reason == "stop"

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["response_format"] == {
            "type": "json_schema",
            "json_schema": json_schema,
        }


def chat_stream_response():
    return [
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1677858242,'
        b'"model":"mistral-large-latest","choices":[{"index":0,"finish_reason":null,'
        b'"delta":{"role":"assistant"}}]}\n',
        b"\n",
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1677858242,'
        b'"model":"mistral-large-latest","choices":[{"index":0,"finish_reason":null,'
        b'"delta":{"content":"Hello"}}]}\n',
        b"\n",
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1677858242,'
        b'"model":"mistral-large-latest","choices":[{"index":0,"finish_reason":null,'
        b'"delta":{"content":" there"}}]}\n',
        b"\n",
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1677858242,'
        b'"model":"mistral-large-latest","choices":[{"index":0,"finish_reason":"stop",'
        b'"delta":{}}]}\n',
        b"\n",
        b"data: [DONE]\n",
    ]


def chat_stream_response_incomplete():
    return [
        # contains first half of a chunk
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1677858242,'
        b'"model":"mistral-large-latest","choi',
        # contains second half of first chunk and first half of second chunk
        b'ces":[{"index":0,"finish_reason":null,"delta":{"role":"assistant"}}]}\n\n'
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1677858242,'
        b'"model":"mistral-large-la',
        # contains second half of second chunk
        b'test","choices":[{"index":0,"finish_reason":null,"delta":{"content":"test"}}]}\n',
        b"\n",
        b'data: {"id":"test-id","object":"chat.completion.chunk","created":1677858242,'
        b'"model":"mistral-large-latest","choices":[{"index":0,"finish_reason":"stop",'
        b'"delta":{}}]}\n',
        b"\n",
        b"data: [DONE]\n",
    ]


async def _run_test_chat_stream(resp, provider):
    mock_client = mock_http_client(MockAsyncStreamingResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        payload = {"messages": [{"role": "user", "content": "Tell me a joke"}]}
        response = provider.chat_stream(chat.RequestPayload(**payload))

        chunks = [jsonable_encoder(chunk) async for chunk in response]

        # Verify we got the expected number of chunks (excluding [DONE])
        assert len(chunks) >= 3

        # Verify the first chunk has the assistant role
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"

        # Verify the last chunk has finish_reason "stop"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

        # Verify all chunks have the expected structure
        for chunk in chunks:
            assert "id" in chunk
            assert chunk["object"] == "chat.completion.chunk"
            assert "created" in chunk
            assert "model" in chunk
            assert "choices" in chunk
            assert len(chunk["choices"]) == 1
            assert "delta" in chunk["choices"][0]

        mock_build_client.assert_called_once_with(
            headers={
                "Authorization": "Bearer key",
            }
        )
        mock_client.post.assert_called_once_with(
            "https://api.mistral.ai/v1/chat/completions",
            json={
                "model": "mistral-large-latest",
                "n": 1,
                **payload,
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.parametrize("resp", [chat_stream_response(), chat_stream_response_incomplete()])
@pytest.mark.asyncio
async def test_chat_stream(resp):
    config = chat_config()
    provider = MistralProvider(EndpointConfig(**config))
    await _run_test_chat_stream(resp, provider)
