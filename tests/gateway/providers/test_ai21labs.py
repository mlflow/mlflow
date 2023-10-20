from unittest import mock

import pytest
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import RouteConfig
from mlflow.gateway.providers.ai21labs import AI21LabsProvider
from mlflow.gateway.schemas import chat, completions, embeddings

from tests.gateway.tools import MockAsyncResponse


def completions_config():
    return {
        "name": "completions",
        "route_type": "llm/v1/completions",
        "model": {
            "provider": "ai21labs",
            "name": "j2-ultra",
            "config": {
                "ai21labs_api_key": "key",
            },
        },
    }


def completions_config_invalid_model():
    return {
        "name": "completions",
        "route_type": "llm/v1/completions",
        "model": {
            "provider": "ai21labs",
            "name": "j2",
            "config": {
                "ai21labs_api_key": "key",
            },
        },
    }


def embedding_config():
    return {
        "name": "embeddings",
        "route_type": "llm/v1/embeddings",
        "model": {
            "provider": "ai21labs",
            "name": "j2-ultra",
            "config": {
                "ai21labs_api_key": "key",
            },
        },
    }


def chat_config():
    return {
        "name": "chat",
        "route_type": "llm/v1/chat",
        "model": {
            "provider": "ai21labs",
            "name": "j2-ultra",
            "config": {
                "ai21labs_api_key": "key",
            },
        },
    }


def completions_response():
    return {
        "id": "7921a78e-d905-c9df-27e3-88e4831e3c3b",
        "prompt": {"text": "This is a test"},
        "completions": [
            {
                "data": {"text": "this is a test response"},
                "finishReason": {"reason": "length", "length": 2},
            }
        ],
    }


@pytest.mark.asyncio
async def test_completions():
    resp = completions_response()
    config = completions_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = AI21LabsProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "candidates": [
                {
                    "text": "this is a test response",
                    "metadata": {"finish_reason": "length"},
                }
            ],
            "metadata": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "model": "j2-ultra",
                "route_type": "llm/v1/completions",
            },
        }
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_param_invalid_model_name_is_not_permitted():
    config = completions_config_invalid_model()
    with pytest.raises(MlflowException, match=r"An Unsupported AI21Labs model.*"):
        RouteConfig(**config)


@pytest.mark.asyncio
async def test_param_maxTokens_is_not_permitted():
    config = completions_config()
    provider = AI21LabsProvider(RouteConfig(**config))
    payload = {
        "prompt": "This should fail",
        "maxTokens": 5000,
    }
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "Invalid parameter maxTokens. Use max_tokens instead." in e.value.detail
    assert e.value.status_code == 422


@pytest.mark.asyncio
async def test_param_model_is_not_permitted():
    config = completions_config()
    provider = AI21LabsProvider(RouteConfig(**config))
    payload = {
        "prompt": "This should fail",
        "model": "j2-light",
    }
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "The parameter 'model' is not permitted" in e.value.detail
    assert e.value.status_code == 422


@pytest.mark.asyncio
async def test_chat_is_not_supported_for_ai21labs():
    config = chat_config()
    provider = AI21LabsProvider(RouteConfig(**config))
    payload = {
        "messages": [{"role": "user", "content": "J2-ultra, can you chat with me? I'm lonely."}]
    }

    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.chat(chat.RequestPayload(**payload))
    assert "The chat route is not available for AI21Labs models" in e.value.detail
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_embeddings_are_not_supported_for_ai21labs():
    config = embedding_config()
    provider = AI21LabsProvider(RouteConfig(**config))
    payload = {"text": "give me that sweet, sweet vector, please."}

    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.embeddings(embeddings.RequestPayload(**payload))
    assert "The embeddings route is not available for AI21Labs models" in e.value.detail
    assert e.value.status_code == 404
