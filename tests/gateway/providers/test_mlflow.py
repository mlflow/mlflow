from unittest import mock

import pydantic
import pytest
from aiohttp import ClientTimeout
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import MlflowModelServingConfig, RouteConfig
from mlflow.gateway.constants import (
    MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS,
    MLFLOW_SERVING_RESPONSE_KEY,
)
from mlflow.gateway.providers.mlflow import MlflowModelServingProvider
from mlflow.gateway.schemas import chat, completions, embeddings

from tests.gateway.tools import MockAsyncResponse, mock_http_client


def completions_config():
    return {
        "name": "completions",
        "route_type": "llm/v1/completions",
        "model": {
            "provider": "mlflow-model-serving",
            "name": "text2text",
            "config": {
                "model_server_url": "http://127.0.0.1:5000",
            },
        },
    }


@pytest.mark.asyncio
async def test_completions():
    resp = {
        "predictions": ["This is a test!"],
        "headers": {"Content-Type": "application/json"},
    }
    config = completions_config()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("time.time", return_value=1677858242), mock.patch(
        "aiohttp.ClientSession", return_value=mock_client
    ) as mock_build_client:
        provider = MlflowModelServingProvider(RouteConfig(**config))
        payload = {
            "prompt": "Is this a test?",
            "temperature": 0.0,
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": None,
            "object": "text_completion",
            "created": 1677858242,
            "model": "text2text",
            "choices": [
                {
                    "text": "This is a test!",
                    "index": 0,
                    "finish_reason": None,
                }
            ],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }
        mock_build_client.assert_called_once()
        mock_client.post.assert_called_once_with(
            "http://127.0.0.1:5000/invocations",
            json={
                "inputs": ["Is this a test?"],
                "params": {
                    "temperature": 0.0,
                    "n": 1,
                },
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.parametrize(
    ("input_data", "expected_output"),
    [
        (
            {"predictions": ["string1", "string2"]},
            [
                completions.Choice(index=0, text="string1", finish_reason=None),
                completions.Choice(index=1, text="string2", finish_reason=None),
            ],
        ),
        (
            {"predictions": {"choices": ["string1", "string2"]}},
            [
                completions.Choice(index=0, text="string1", finish_reason=None),
                completions.Choice(index=1, text="string2", finish_reason=None),
            ],
        ),
        (
            {"predictions": {"choices": ["string1", "string2"], "ignored": ["a", "b"]}},
            [
                completions.Choice(index=0, text="string1", finish_reason=None),
                completions.Choice(index=1, text="string2", finish_reason=None),
            ],
        ),
        (
            {"predictions": {"arbitrary_key": ["string1", "string2", "string3"]}},
            [
                completions.Choice(index=0, text="string1", finish_reason=None),
                completions.Choice(index=1, text="string2", finish_reason=None),
                completions.Choice(index=2, text="string3", finish_reason=None),
            ],
        ),
    ],
)
def test_valid_completions_input_parsing(input_data, expected_output):
    config = completions_config()
    provider = MlflowModelServingProvider(RouteConfig(**config))
    parsed = provider._process_completions_response_for_mlflow_serving(input_data)

    assert parsed == expected_output


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"predictions": [1, 2, 3]},  # List of integers
        {"predictions": {"choices": [1, 2, 3]}},  # Dict with list of integers
        {"predictions": {"arbitrary_key": [1, 2, 3]}},  # Dict with list of integers
        {"predictions": {"key1": ["string1"], "key2": ["string2"]}},  # Multiple keys in dict
        {"predictions": []},  # Empty list
        {"predictions": {"choices": []}},  # Dict with empty list
    ],
)
def test_validation_errors(invalid_data):
    config = completions_config()
    provider = MlflowModelServingProvider(RouteConfig(**config))
    with pytest.raises(HTTPException, match=r".*") as e:
        provider._process_completions_response_for_mlflow_serving(invalid_data)
    assert e.value.status_code == 502
    assert "ServingTextResponse\npredictions" in e.value.detail


def test_invalid_return_key_from_mlflow_serving():
    config = completions_config()
    provider = MlflowModelServingProvider(RouteConfig(**config))
    with pytest.raises(HTTPException, match=r".*") as e:
        provider._process_completions_response_for_mlflow_serving(
            {"invalid_return_key": ["invalid", "response"]}
        )

    assert "1 validation error for ServingTextResponse\npredictions" in e.value.detail
    assert e.value.status_code == 502


def embedding_config():
    return {
        "name": "embeddings",
        "route_type": "llm/v1/embeddings",
        "model": {
            "provider": "mlflow-model-serving",
            "name": "sentence-piece",
            "config": {
                "model_server_url": "http://127.0.0.1:2000",
            },
        },
    }


@pytest.mark.asyncio
async def test_embeddings():
    resp = {
        "predictions": [[0.01, -0.1], [0.03, -0.03]],
        "headers": {"Content-Type": "application/json"},
    }
    config = embedding_config()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        provider = MlflowModelServingProvider(RouteConfig(**config))
        payload = {"input": ["test1", "test2"]}
        response = await provider.embeddings(embeddings.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [
                        0.01,
                        -0.1,
                    ],
                    "index": 0,
                },
                {
                    "object": "embedding",
                    "embedding": [
                        0.03,
                        -0.03,
                    ],
                    "index": 1,
                },
            ],
            "model": "sentence-piece",
            "usage": {"prompt_tokens": None, "total_tokens": None},
        }
        mock_build_client.assert_called_once()
        mock_client.post.assert_called_once_with(
            "http://127.0.0.1:2000/invocations",
            json={"inputs": ["test1", "test2"]},
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.parametrize(
    "response",
    [
        {MLFLOW_SERVING_RESPONSE_KEY: "string_value"},
        {MLFLOW_SERVING_RESPONSE_KEY: ["string", "values"]},
        {MLFLOW_SERVING_RESPONSE_KEY: [[1.0, 2.3], ["string", "values"]]},
        {MLFLOW_SERVING_RESPONSE_KEY: [[1.0, 2.3], [1.2, "string"]]},
        {MLFLOW_SERVING_RESPONSE_KEY: [[], []]},
        {MLFLOW_SERVING_RESPONSE_KEY: []},
    ],
)
def test_invalid_embeddings_response(response):
    config = embedding_config()
    provider = MlflowModelServingProvider(RouteConfig(**config))
    with pytest.raises(HTTPException, match=r".*") as e:
        provider._process_embeddings_response_for_mlflow_serving(response)

    assert "EmbeddingsResponse\npredictions" in e.value.detail
    assert e.value.status_code == 502


def chat_config():
    return {
        "name": "chat",
        "route_type": "llm/v1/chat",
        "model": {
            "provider": "mlflow-model-serving",
            "name": "chat-bot-9000",
            "config": {
                "model_server_url": "http://127.0.0.1:4000",
            },
        },
    }


@pytest.mark.asyncio
async def test_chat():
    resp = {
        "predictions": ["It is a test"],
        "headers": {"Content-Type": "application/json"},
    }
    config = chat_config()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("time.time", return_value=1700242674), mock.patch(
        "aiohttp.ClientSession", return_value=mock_client
    ) as mock_build_client:
        provider = MlflowModelServingProvider(RouteConfig(**config))
        payload = {"messages": [{"role": "user", "content": "Is this a test?"}]}
        response = await provider.chat(chat.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": None,
            "created": 1700242674,
            "object": "chat.completion",
            "model": "chat-bot-9000",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "It is a test",
                        "tool_calls": None,
                    },
                    "finish_reason": None,
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        }
        mock_build_client.assert_called_once()
        mock_client.post.assert_called_once_with(
            "http://127.0.0.1:4000/invocations",
            json={
                "inputs": ["Is this a test?"],
                "params": {"temperature": 0.0, "n": 1},
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.asyncio
async def test_chat_exception_raised_for_multiple_elements_in_query():
    resp = {"predictions": "It is a test"}
    config = chat_config()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        provider = MlflowModelServingProvider(RouteConfig(**config))
        payload = {
            "messages": [
                {"role": "user", "content": "Is this a test?"},
                {"role": "user", "content": "This is a second message."},
            ]
        }

        with pytest.raises(HTTPException, match=r".*") as e:
            await provider.chat(chat.RequestPayload(**payload))
        assert "MLflow chat models are only capable of processing" in e.value.detail


def test_route_construction_fails_with_invalid_config():
    with pytest.raises(pydantic.ValidationError, match="model_server_url"):
        MlflowModelServingConfig(model_server_url=None)
