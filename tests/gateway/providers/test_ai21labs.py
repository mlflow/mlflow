from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi.encoders import jsonable_encoder

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import RouteConfig
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.ai21labs import AI21LabsProvider
from mlflow.gateway.schemas import chat, completions, embeddings

from tests.gateway.tools import MockAsyncResponse


def completions_config():
    return {
        "name": "completions",
        "endpoint_type": "llm/v1/completions",
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
        "endpoint_type": "llm/v1/completions",
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
        "endpoint_type": "llm/v1/embeddings",
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
        "endpoint_type": "llm/v1/chat",
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
    with (
        mock.patch("time.time", return_value=1677858242),
        mock.patch("aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)) as mock_post,
    ):
        provider = AI21LabsProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
            "temperature": 0.2,
            "max_tokens": 1000,
            "n": 1,
            "stop": ["foobazbardiddly"],
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": None,
            "object": "text_completion",
            "created": 1677858242,
            "model": "j2-ultra",
            "choices": [{"text": "this is a test response", "index": 0, "finish_reason": "length"}],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }
        mock_post.assert_called_once_with(
            "https://api.ai21.com/studio/v1/j2-ultra/complete",
            json={
                "temperature": 0.2,
                "numResults": 1,
                "stopSequences": ["foobazbardiddly"],
                "maxTokens": 1000,
                "prompt": "This is a test",
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


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
    with pytest.raises(AIGatewayException, match=r".*") as e:
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
    with pytest.raises(AIGatewayException, match=r".*") as e:
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

    with pytest.raises(AIGatewayException, match=r".*") as e:
        await provider.chat(chat.RequestPayload(**payload))
    assert "The chat route is not implemented for AI21Labs models" in e.value.detail
    assert e.value.status_code == 501


@pytest.mark.asyncio
async def test_embeddings_are_not_supported_for_ai21labs():
    config = embedding_config()
    provider = AI21LabsProvider(RouteConfig(**config))
    payload = {"input": "give me that sweet, sweet vector, please."}

    with pytest.raises(AIGatewayException, match=r".*") as e:
        await provider.embeddings(embeddings.RequestPayload(**payload))
    assert "The embeddings route is not implemented for AI21Labs models" in e.value.detail
    assert e.value.status_code == 501
