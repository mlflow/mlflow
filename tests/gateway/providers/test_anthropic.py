from unittest import mock

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
import pytest

from mlflow.gateway.providers.anthropic import AnthropicProvider
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.config import RouteConfig
from tests.gateway.tools import MockAsyncResponse


def completions_response():
    return {
        "completion": "Here is a basic overview of how a car works:\n\n1. The engine. "
        "The engine is the power source that makes the car move.",
        "stop_reason": "max_tokens",
        "model": "claude-instant-1.1",
        "truncated": False,
        "stop": None,
        "log_id": "dee173f87ddf1357da639dee3c38d833",
        "exception": None,
    }


def completions_config():
    return {
        "name": "completions",
        "route_type": "llm/v1/completions",
        "model": {
            "provider": "anthropic",
            "name": "claude-instant-1",
            "config": {
                "anthropic_api_base": "https://api.anthropic.com/v1",
                "anthropic_api_key": "key",
            },
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate_count", [1, None])
async def test_completions(candidate_count):
    resp = completions_response()
    config = completions_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = AnthropicProvider(RouteConfig(**config))
        payload = {"prompt": "How does a car work?", "max_tokens": 200}
        if candidate_count is not None:
            payload["candidate_count"] = candidate_count
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "candidates": [
                {
                    "text": "Here is a basic overview of how a car works:\n\n1. The engine. "
                    "The engine is the power source that makes the car move.",
                    "metadata": {"finish_reason": "length"},
                }
            ],
            "metadata": {
                "model": "claude-instant-1.1",
                "route_type": "llm/v1/completions",
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            },
        }
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_completions_throws_with_missing_max_tokens():
    config = completions_config()
    provider = AnthropicProvider(RouteConfig(**config))
    payload = {"prompt": "Would Fozzie or Kermet win in a fight?"}
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "You must set an integer value for 'max_tokens' for the Anthropic" in e.value.detail
    assert e.value.status_code == 400


@pytest.mark.asyncio
async def test_completions_throws_with_unsupported_candidate_count():
    config = completions_config()
    provider = AnthropicProvider(RouteConfig(**config))
    payload = {
        "prompt": "Would Fozzie or Kermet win in a fight?",
        "candidate_count": 5,
        "max_tokens": 10,
    }
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "'candidate_count' must be '1' for the Anthropic provider" in e.value.detail
    assert e.value.status_code == 422


@pytest.mark.asyncio
async def test_completions_throws_with_top_p_defined():
    config = completions_config()
    provider = AnthropicProvider(RouteConfig(**config))
    payload = {"prompt": "Would Fozzie or Kermet win in a fight?", "max_tokens": 500, "top_p": 0.6}
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "Cannot set both 'temperature' and 'top_p' parameters. Please" in e.value.detail
    assert e.value.status_code == 400


@pytest.mark.asyncio
async def test_completions_throws_with_stream_set_to_true():
    config = completions_config()
    provider = AnthropicProvider(RouteConfig(**config))
    payload = {
        "prompt": "Could the Millennium Falcon fight a Borg Cube and win?",
        "max_tokens": 5000,
        "stream": "true",
    }
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "Setting the 'stream' parameter to 'true' is not supported" in e.value.detail
    assert e.value.status_code == 400


def chat_config():
    return {
        "name": "chat",
        "route_type": "llm/v1/chat",
        "model": {
            "provider": "anthropic",
            "name": "claude-instant-1",
            "config": {
                "anthropic_api_base": "https://api.anthropic.com/v1",
                "anthropic_api_key": "key",
            },
        },
    }


@pytest.mark.asyncio
async def test_chat_is_not_supported_for_anthropic():
    config = chat_config()
    provider = AnthropicProvider(RouteConfig(**config))
    payload = {
        "messages": [{"role": "user", "content": "Claude, can you chat with me? I'm lonely."}]
    }

    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.chat(chat.RequestPayload(**payload))
    assert "The chat route is not available for Anthropic models" in e.value.detail
    assert e.value.status_code == 404


def embedding_config():
    return {
        "name": "embeddings",
        "route_type": "llm/v1/embeddings",
        "model": {
            "provider": "anthropic",
            "name": "claude-1.3-100k",
            "config": {
                "anthropic_api_base": "https://api.anthropic.com/v1",
                "anthropic_api_key": "key",
            },
        },
    }


@pytest.mark.asyncio
async def test_embeddings_are_not_supported_for_anthropic():
    config = embedding_config()
    provider = AnthropicProvider(RouteConfig(**config))
    payload = {"text": "give me that sweet, sweet vector, please."}

    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.embeddings(embeddings.RequestPayload(**payload))
    assert "The embeddings route is not available for Anthropic models" in e.value.detail
    assert e.value.status_code == 404
