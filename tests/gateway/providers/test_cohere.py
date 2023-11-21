from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
from mlflow.gateway.providers.cohere import CohereProvider
from mlflow.gateway.schemas import completions, embeddings

from tests.gateway.tools import MockAsyncResponse


def completions_config():
    return {
        "name": "completions",
        "route_type": "llm/v1/completions",
        "model": {
            "provider": "cohere",
            "name": "command",
            "config": {
                "cohere_api_key": "key",
            },
        },
    }


def completions_response():
    return {
        "id": "string",
        "generations": [
            {
                "id": "string",
                "text": "This is a test",
            }
        ],
        "prompt": "string",
        "headers": {"Content-Type": "application/json"},
    }


@pytest.mark.asyncio
async def test_completions():
    resp = completions_response()
    config = completions_config()
    with mock.patch("time.time", return_value=1677858242), mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = CohereProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
            "n": 1,
            "stop": ["foobar"],
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": None,
            "object": "text_completion",
            "created": 1677858242,
            "model": "command",
            "choices": [
                {
                    "text": "This is a test",
                    "index": 0,
                    "finish_reason": None,
                }
            ],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }
        mock_post.assert_called_once_with(
            "https://api.cohere.ai/v1/generate",
            json={
                "prompt": "This is a test",
                "model": "command",
                "num_generations": 1,
                "stop_sequences": ["foobar"],
                "temperature": 0.0,
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.asyncio
async def test_completions_temperature_is_scaled_correctly():
    resp = completions_response()
    config = completions_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = CohereProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
            "temperature": 0.5,
        }
        await provider.completions(completions.RequestPayload(**payload))
        assert mock_post.call_args[1]["json"]["temperature"] == 0.5 * 2.5


def embeddings_config():
    return {
        "name": "embeddings",
        "route_type": "llm/v1/embeddings",
        "model": {
            "provider": "cohere",
            "name": "embed-english-light-v2.0",
            "config": {
                "cohere_api_key": "key",
            },
        },
    }


def embeddings_response():
    return {
        "id": "bc57846a-3e56-4327-8acc-588ca1a37b8a",
        "texts": ["hello world"],
        "embeddings": [
            [
                3.25,
                0.7685547,
                2.65625,
                -0.30126953,
                -2.3554688,
                1.2597656,
            ]
        ],
        "meta": [
            {
                "api_version": [
                    {
                        "version": "1",
                    }
                ]
            },
        ],
        "headers": {"Content-Type": "application/json"},
    }


def embeddings_batch_response():
    return {
        "id": "bc57846a-3e56-4327-8acc-588ca1a37b8a",
        "texts": ["hello world"],
        "embeddings": [
            [
                3.25,
                0.7685547,
                2.65625,
                -0.30126953,
                -2.3554688,
                1.2597656,
            ],
            [
                7.25,
                0.7685547,
                4.65625,
                -0.30126953,
                -2.3554688,
                8.2597656,
            ],
        ],
        "meta": [
            {
                "api_version": [
                    {
                        "version": "1",
                    }
                ]
            },
        ],
        "headers": {"Content-Type": "application/json"},
    }


@pytest.mark.asyncio
async def test_embeddings():
    resp = embeddings_response()
    config = embeddings_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = CohereProvider(RouteConfig(**config))
        payload = {"input": "This is a test"}
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
            "model": "embed-english-light-v2.0",
            "usage": {"prompt_tokens": None, "total_tokens": None},
        }
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_batch_embeddings():
    resp = embeddings_batch_response()
    config = embeddings_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = CohereProvider(RouteConfig(**config))
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
            "model": "embed-english-light-v2.0",
            "usage": {"prompt_tokens": None, "total_tokens": None},
        }
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_param_model_is_not_permitted():
    config = embeddings_config()
    provider = CohereProvider(RouteConfig(**config))
    payload = {
        "prompt": "This should fail",
        "max_tokens": 5000,
        "model": "something-else",
    }
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "The parameter 'model' is not permitted" in e.value.detail
    assert e.value.status_code == 422


@pytest.mark.parametrize("prompt", [{"set1", "set2"}, ["list1"], [1], ["list1", "list2"], [1, 2]])
@pytest.mark.asyncio
async def test_completions_throws_if_prompt_contains_non_string(prompt):
    config = completions_config()
    provider = CohereProvider(RouteConfig(**config))
    payload = {"prompt": prompt}
    with pytest.raises(ValidationError, match=r"prompt"):
        await provider.completions(completions.RequestPayload(**payload))
