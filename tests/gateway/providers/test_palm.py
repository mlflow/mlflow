from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
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
    with mock.patch("time.time", return_value=1677858242), mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = PaLMProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
            "n": 1,
            "max_tokens": 1000,
            "stop": ["foobar"],
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": None,
            "object": "text_completion",
            "created": 1677858242,
            "model": "text-bison",
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
            "https://generativelanguage.googleapis.com/v1beta3/models/text-bison:generateText",
            json={
                "prompt": {
                    "text": "This is a test",
                },
                "temperature": 0.0,
                "candidateCount": 1,
                "maxOutputTokens": 1000,
                "stopSequences": ["foobar"],
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
        provider = PaLMProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
            "temperature": 0.5,
        }
        await provider.completions(completions.RequestPayload(**payload))
        assert mock_post.call_args[1]["json"]["temperature"] == 0.5 * 0.5


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
    ("payload", "expected_llm_input"),
    [
        (
            {"messages": [{"role": "user", "content": "Tell me a joke"}]},
            {
                "temperature": 0.0,
                "candidateCount": 1,
                "prompt": {"messages": [{"content": "Tell me a joke", "author": "user"}]},
            },
        ),
        (
            {
                "messages": [
                    {"role": "system", "content": "You're funny"},
                    {"role": "user", "content": "Tell me a joke"},
                ]
            },
            {
                "temperature": 0.0,
                "candidateCount": 1,
                "prompt": {
                    "messages": [
                        {"content": "You're funny", "author": "system"},
                        {"content": "Tell me a joke", "author": "user"},
                    ]
                },
            },
        ),
        (
            {
                "messages": [{"role": "user", "content": "Tell me a joke"}],
                "temperature": 0.5,
            },
            {
                "temperature": 0.25,
                "candidateCount": 1,
                "prompt": {"messages": [{"content": "Tell me a joke", "author": "user"}]},
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_chat(payload, expected_llm_input):
    resp = chat_response()
    config = chat_config()
    with mock.patch("time.time", return_value=1700242674), mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = PaLMProvider(RouteConfig(**config))
        response = await provider.chat(chat.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": None,
            "created": 1700242674,
            "object": "chat.completion",
            "model": "chat-bison",
            "choices": [
                {
                    "message": {
                        "role": "1",
                        "content": "Hi there! How can I help you today?",
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
        mock_post.assert_called_once_with(
            "https://generativelanguage.googleapis.com/v1beta3/models/chat-bison:generateMessage",
            json=expected_llm_input,
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


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


def embeddings_batch_response():
    return {
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
        payload = {"input": prompt}
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
            "model": "embedding-gecko",
            "usage": {"prompt_tokens": None, "total_tokens": None},
        }
        mock_post.assert_called_once()


async def test_embeddings_batch():
    config = embeddings_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(embeddings_batch_response())
    ) as mock_post:
        provider = PaLMProvider(RouteConfig(**config))
        payload = {"input": ["this is a", "batch test"]}
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
            "model": "embedding-gecko",
            "usage": {"prompt_tokens": None, "total_tokens": None},
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
