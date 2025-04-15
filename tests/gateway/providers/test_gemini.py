from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.providers.gemini import GeminiProvider
from mlflow.gateway.schemas import embeddings

from tests.gateway.tools import MockAsyncResponse


def embedding_config():
    return {
        "name": "embeddings",
        "route_type": "llm/v1/embeddings",
        "model": {
            "provider": "gemini",
            "name": "text-embedding-004",
            "config": {
                "gemini_api_key": "key",
            },
        },
    }


def fake_single_embedding_response():
    return {"embeddings": [{"values": [0.1, 0.2, 0.3]}]}


def fake_batch_embedding_response():
    return {"embeddings": [{"values": [0.1, 0.2, 0.3]}, {"values": [0.4, 0.5, 0.6]}]}


@pytest.mark.asyncio
async def test_gemini_single_embedding():
    config = embedding_config()
    provider = GeminiProvider(RouteConfig(**config))
    payload = {"input": "This is a test embedding."}

    expected_payload = {"content": {"parts": [{"text": "This is a test embedding."}]}}
    expected_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
    )

    with mock.patch(
        "aiohttp.ClientSession.post",
        return_value=MockAsyncResponse(fake_single_embedding_response()),
    ) as mock_post:
        response = await provider.embeddings(embeddings.RequestPayload(**payload))

    expected_data = [embeddings.EmbeddingObject(embedding=[0.1, 0.2, 0.3], index=0)]
    expected_response = {
        "object": "list",
        "data": jsonable_encoder(expected_data),
        "model": "text-embedding-004",
        "usage": {"prompt_tokens": None, "total_tokens": None},
    }
    assert jsonable_encoder(response) == expected_response

    mock_post.assert_called_once_with(
        expected_url,
        json=expected_payload,
        timeout=mock.ANY,
    )


@pytest.mark.asyncio
async def test_gemini_batch_embedding():
    config = embedding_config()
    provider = GeminiProvider(RouteConfig(**config))
    payload = {"input": ["Test embedding 1.", "Test embedding 2."]}

    expected_payload = {
        "requests": [
            {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": "Test embedding 1."}]},
            },
            {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": "Test embedding 2."}]},
            },
        ]
    }
    expected_url = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents"

    with mock.patch(
        "aiohttp.ClientSession.post",
        return_value=MockAsyncResponse(fake_batch_embedding_response()),
    ) as mock_post:
        response = await provider.embeddings(embeddings.RequestPayload(**payload))

    expected_data = [
        embeddings.EmbeddingObject(embedding=[0.1, 0.2, 0.3], index=0),
        embeddings.EmbeddingObject(embedding=[0.4, 0.5, 0.6], index=1),
    ]
    expected_response = {
        "object": "list",
        "data": jsonable_encoder(expected_data),
        "model": "text-embedding-004",
        "usage": {"prompt_tokens": None, "total_tokens": None},
    }

    assert jsonable_encoder(response) == expected_response

    mock_post.assert_called_once_with(
        expected_url,
        json=expected_payload,
        timeout=mock.ANY,
    )
