from unittest import mock

from aiohttp import ClientTimeout
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
import pydantic
import pytest

from mlflow.gateway.config import MlflowModelServingConfig, RouteConfig
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
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
                "mlflow_server_url": "http://127.0.0.1:5000",
            },
        },
    }


@pytest.mark.asyncio
async def test_completions():
    resp = {"predictions": "This is a test!"}
    config = completions_config()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        provider = MlflowModelServingProvider(RouteConfig(**config))
        payload = {
            "prompt": "Is this a test?",
            "temperature": 0,
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "candidates": [{"text": "This is a test!", "metadata": {}}],
            "metadata": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "model": "text2text",
                "route_type": "llm/v1/completions",
            },
        }
        mock_build_client.assert_called_once()
        mock_client.post.assert_called_once_with(
            "http://127.0.0.1:5000/invocations",
            json={
                "inputs": ["Is this a test?"],
                "params": {
                    "temperature": 0,
                },
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


def embedding_config():
    return {
        "name": "embeddings",
        "route_type": "llm/v1/embeddings",
        "model": {
            "provider": "mlflow-model-serving",
            "name": "sentence-piece",
            "config": {
                "mlflow_server_url": "http://127.0.0.1:2000",
            },
        },
    }


@pytest.mark.asyncio
async def test_embeddings():
    resp = {"predictions": [[0.01, -0.1], [0.03, -0.03]]}
    config = embedding_config()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        provider = MlflowModelServingProvider(RouteConfig(**config))
        payload = {"text": ["test1", "test2"]}
        response = await provider.embeddings(embeddings.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "embeddings": [
                [0.01, -0.1],
                [0.03, -0.03],
            ],
            "metadata": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "model": "sentence-piece",
                "route_type": "llm/v1/embeddings",
            },
        }
        mock_build_client.assert_called_once()
        mock_client.post.assert_called_once_with(
            "http://127.0.0.1:2000/invocations",
            json={"inputs": ["test1", "test2"]},
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


def chat_config():
    return {
        "name": "chat",
        "route_type": "llm/v1/chat",
        "model": {
            "provider": "mlflow-model-serving",
            "name": "chat-bot-9000",
            "config": {
                "mlflow_server_url": "http://127.0.0.1:4000",
            },
        },
    }


@pytest.mark.asyncio
async def test_chat():
    resp = {"predictions": "It is a test"}
    config = chat_config()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client) as mock_build_client:
        provider = MlflowModelServingProvider(RouteConfig(**config))
        payload = {"messages": [{"role": "user", "content": "Is this a test?"}]}
        response = await provider.chat(chat.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "candidates": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "It is a test",
                    },
                    "metadata": {},
                }
            ],
            "metadata": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "model": "chat-bot-9000",
                "route_type": "llm/v1/chat",
            },
        }
        mock_build_client.assert_called_once()
        mock_client.post.assert_called_once_with(
            "http://127.0.0.1:4000/invocations",
            json={"inputs": ["Is this a test?"]},
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
    with pytest.raises(pydantic.ValidationError, match="none is not an allowed"):
        MlflowModelServingConfig(mlflow_server_url=None)
