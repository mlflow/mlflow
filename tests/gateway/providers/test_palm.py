from unittest import mock

import pytest
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.providers.palm import PaLMProvider
from mlflow.gateway.schemas import chat, completions, embeddings

from tests.gateway.tools import MockAsyncResponse


def completions_config():
    return {
        "name": "completions",
        "route_type": "llm/v1/completions",
        "model": {
            "provider": "palm",
            "name": "text-bison",
            "config": {
                "palm_api_key": "key",
            },
        },
    }


def completions_response():
    return {
        "candidates": [
            {
                "output": "This is a test",
                "safetyRatings": [
                    {"category": "HARM_CATEGORY_DEROGATORY", "probability": "NEGLIGIBLE"}
                ],
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }


@pytest.mark.asyncio
async def test_completions():
    resp = completions_response()
    config = completions_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = PaLMProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "candidates": [
                {
                    "text": "This is a test",
                    "metadata": {},
                }
            ],
            "metadata": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "model": "text-bison",
                "route_type": "llm/v1/completions",
            },
        }
        mock_post.assert_called_once()


def chat_config():
    return {
        "name": "chat",
        "route_type": "llm/v1/chat",
        "model": {
            "provider": "palm",
            "name": "chat-bison",
            "config": {
                "palm_api_key": "key",
            },
        },
    }


def chat_response():
    return {
        "candidates": [{"author": "1", "content": "Hi there! How can I help you today?"}],
        "messages": [{"author": "0", "content": "hi"}],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"messages": [{"role": "user", "content": "Tell me a joke"}]},
        {
            "messages": [
                {"role": "system", "content": "You're funny"},
                {"role": "user", "content": "Tell me a joke"},
            ]
        },
        {
            "messages": [{"role": "user", "content": "Tell me a joke"}],
            "temperature": 0.5,
        },
    ],
)
@pytest.mark.asyncio
async def test_chat(payload):
    resp = chat_response()
    config = chat_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = PaLMProvider(RouteConfig(**config))
        response = await provider.chat(chat.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "candidates": [
                {
                    "message": {
                        "role": "1",
                        "content": "Hi there! How can I help you today?",
                    },
                    "metadata": {"finish_reason": None},
                }
            ],
            "metadata": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "model": "chat-bison",
                "route_type": "llm/v1/chat",
            },
        }
        mock_post.assert_called_once()


def embeddings_config():
    return {
        "name": "embeddings",
        "route_type": "llm/v1/embeddings",
        "model": {
            "provider": "palm",
            "name": "embedding-gecko",
            "config": {
                "palm_api_key": "key",
            },
        },
    }


def embeddings_response():
    return {
        "embeddings": [
            {
                "value": [
                    3.25,
                    0.7685547,
                    2.65625,
                    -0.30126953,
                    -2.3554688,
                    1.2597656,
                ]
            }
        ],
        "headers": {"Content-Type": "application/json"},
    }


@pytest.mark.parametrize("prompt", ["This is a test", ["This is a test"]])
@pytest.mark.asyncio
async def test_embeddings(prompt):
    config = embeddings_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(embeddings_response())
    ) as mock_post:
        provider = PaLMProvider(RouteConfig(**config))
        payload = {"text": prompt}
        response = await provider.embeddings(embeddings.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
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
            "metadata": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "model": "embedding-gecko",
                "route_type": "llm/v1/embeddings",
            },
        }
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_param_model_is_not_permitted():
    config = completions_config()
    provider = PaLMProvider(RouteConfig(**config))
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
    provider = PaLMProvider(RouteConfig(**config))
    payload = {"prompt": prompt}
    with pytest.raises(ValidationError, match=r"prompt"):
        await provider.completions(completions.RequestPayload(**payload))


@pytest.mark.parametrize(
    "payload",
    [
        {
            "messages": [{"role": "user", "content": "This should fail."}],
            "max_tokens": 5000,
        },
        {
            "messages": [{"role": "user", "content": "This should fail."}],
            "maxOutputTokens": 5000,
        },
    ],
)
@pytest.mark.asyncio
async def test_param_max_tokens_for_chat_is_not_permitted(payload):
    config = chat_config()
    provider = PaLMProvider(RouteConfig(**config))
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.chat(chat.RequestPayload(**payload))
    assert "Max tokens is not supported for PaLM chat." in e.value.detail
    assert e.value.status_code == 422
