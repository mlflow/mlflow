from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow import MlflowException
from mlflow.gateway.config import RouteConfig
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.mosaicml import MosaicMLProvider
from mlflow.gateway.schemas import chat, completions, embeddings
from mlflow.gateway.schemas.chat import RequestMessage

from tests.gateway.tools import MockAsyncResponse


def completions_config():
    return {
        "name": "completions",
        "endpoint_type": "llm/v1/completions",
        "model": {
            "provider": "mosaicml",
            "name": "mpt-7b-instruct",
            "config": {
                "mosaicml_api_key": "key",
            },
        },
    }


def completions_response():
    return {
        "outputs": [
            "This is a test",
        ],
        "headers": {"Content-Type": "application/json"},
    }


@pytest.mark.asyncio
async def test_completions():
    resp = completions_response()
    config = completions_config()
    with (
        mock.patch("time.time", return_value=1677858242),
        mock.patch("aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)) as mock_post,
    ):
        provider = MosaicMLProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
            "max_tokens": 1000,
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": None,
            "object": "text_completion",
            "created": 1677858242,
            "model": "mpt-7b-instruct",
            "choices": [{"text": "This is a test", "index": 0, "finish_reason": None}],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }
        mock_post.assert_called_once_with(
            "https://models.hosted-on.mosaicml.hosting/mpt-7b-instruct/v1/predict",
            json={
                "inputs": ["This is a test"],
                "parameters": {"temperature": 0.0, "n": 1, "max_new_tokens": 1000},
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


def chat_config():
    return {
        "name": "chat",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "provider": "mosaicml",
            "name": "llama2-70b-chat",
            "config": {
                "mosaicml_api_key": "key",
            },
        },
    }


def chat_response():
    return {
        "outputs": [
            "This is a test",
        ],
        "headers": {"Content-Type": "application/json"},
    }


@pytest.mark.parametrize(
    ("payload", "expected_llm_input"),
    [
        (
            {"messages": [{"role": "user", "content": "Tell me a joke"}]},
            {
                "inputs": ["<s>[INST] Tell me a joke [/INST]"],
                "parameters": {
                    "temperature": 0.0,
                    "n": 1,
                },
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
                "inputs": ["<s>[INST] <<SYS>> You're funny <</SYS>> Tell me a joke [/INST]"],
                "parameters": {
                    "temperature": 0.0,
                    "n": 1,
                },
            },
        ),
        (
            {
                "messages": [{"role": "user", "content": "Tell me a joke"}],
                "temperature": 0.5,
                "max_tokens": 1000,
            },
            {
                "inputs": ["<s>[INST] Tell me a joke [/INST]"],
                "parameters": {
                    "temperature": 0.5,
                    "n": 1,
                    "max_new_tokens": 1000,
                },
            },
        ),
        (
            {
                "messages": [
                    {"role": "system", "content": "You're funny"},
                    {"role": "user", "content": "Tell me a joke"},
                    {"role": "assistant", "content": "Haha"},
                    {"role": "user", "content": "That was a bad joke"},
                ]
            },
            {
                "inputs": [
                    (
                        "<s>[INST] <<SYS>> You're funny <</SYS>>"
                        " Tell me a joke [/INST] Haha </s><s>[INST]"
                        " That was a bad joke [/INST]"
                    )
                ],
                "parameters": {
                    "temperature": 0.0,
                    "n": 1,
                },
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_chat(payload, expected_llm_input):
    resp = chat_response()
    config = chat_config()
    with (
        mock.patch("time.time", return_value=1700242674),
        mock.patch("aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)) as mock_post,
    ):
        provider = MosaicMLProvider(RouteConfig(**config))
        response = await provider.chat(chat.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": None,
            "created": 1700242674,
            "object": "chat.completion",
            "model": "llama2-70b-chat",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is a test",
                        "tool_calls": None,
                        "refusal": None,
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
            "https://models.hosted-on.mosaicml.hosting/llama2-70b-chat/v1/predict",
            json=expected_llm_input,
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


def embeddings_config():
    return {
        "name": "embeddings",
        "endpoint_type": "llm/v1/embeddings",
        "model": {
            "provider": "mosaicml",
            "name": "instructor-large",
            "config": {
                "mosaicml_api_key": "key",
            },
        },
    }


def embeddings_response():
    return {
        "outputs": [
            [
                3.25,
                0.7685547,
                2.65625,
                -0.30126953,
                -2.3554688,
                1.2597656,
            ]
        ],
        "headers": {"Content-Type": "application/json"},
    }


def embeddings_batch_response():
    return {
        "outputs": [
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


@pytest.mark.asyncio
async def test_embeddings():
    resp = embeddings_response()
    config = embeddings_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = MosaicMLProvider(RouteConfig(**config))
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
                }
            ],
            "model": "instructor-large",
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
        provider = MosaicMLProvider(RouteConfig(**config))
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
            "model": "instructor-large",
            "usage": {"prompt_tokens": None, "total_tokens": None},
        }
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_param_model_is_not_permitted():
    config = embeddings_config()
    provider = MosaicMLProvider(RouteConfig(**config))
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
    provider = MosaicMLProvider(RouteConfig(**config))
    payload = {"prompt": prompt}
    with pytest.raises(ValidationError, match=r"prompt"):
        await provider.completions(completions.RequestPayload(**payload))


@pytest.mark.parametrize(
    ("messages", "expected_output"),
    [
        (
            [
                RequestMessage(role="system", content="Hello"),
                RequestMessage(role="user", content="Hi there"),
                RequestMessage(role="assistant", content="How can I help?"),
                RequestMessage(role="user", content="Thanks!"),
            ],
            "<s>[INST] <<SYS>> Hello <</SYS>> Hi there [/INST] How can I help? </s>"
            "<s>[INST] Thanks! [/INST]",
        ),
        (
            [
                RequestMessage(role="system", content="Hello"),
                RequestMessage(role="user", content="Hi there"),
            ],
            "<s>[INST] <<SYS>> Hello <</SYS>> Hi there [/INST]",
        ),
        (
            [
                RequestMessage(role="user", content="Hi there"),
            ],
            "<s>[INST] Hi there [/INST]",
        ),
        (
            [
                RequestMessage(role="user", content="Hi there"),
                RequestMessage(role="assistant", content="How can I help?"),
                RequestMessage(role="user", content="Thanks!"),
            ],
            "<s>[INST] Hi there [/INST] How can I help? </s><s>[INST] Thanks! [/INST]",
        ),
        (
            [
                RequestMessage(role="user", content="Test"),
                RequestMessage(role="user", content="Test"),
            ],
            "<s>[INST] Test Test [/INST]",
        ),
        (
            [
                RequestMessage(role="system", content="Test"),
                RequestMessage(role="system", content="Test"),
            ],
            "<s>[INST] <<SYS>> Test <</SYS>> <<SYS>> Test <</SYS>> [/INST]",
        ),
        (
            [
                RequestMessage(role="system", content="Test"),
                RequestMessage(role="user", content="Test"),
                RequestMessage(role="user", content="Test"),
            ],
            "<s>[INST] <<SYS>> Test <</SYS>> Test Test [/INST]",
        ),
        (
            [
                RequestMessage(role="system", content="Test"),
                RequestMessage(role="user", content="Test"),
                RequestMessage(role="assistant", content="Test"),
                RequestMessage(role="assistant", content="Test"),
            ],
            "<s>[INST] <<SYS>> Test <</SYS>> Test [/INST] Test </s><s> Test ",
        ),
        (
            [RequestMessage(role="assistant", content="Test")],
            "<s> Test ",
        ),
        (
            [
                RequestMessage(role="system", content="Test"),
                RequestMessage(role="assistant", content="Test"),
            ],
            "<s>[INST] <<SYS>> Test <</SYS>> [/INST] Test ",
        ),
        (
            [
                RequestMessage(role="assistant", content="Test"),
                RequestMessage(role="system", content="Test"),
            ],
            "<s> Test </s><s>[INST] <<SYS>> Test <</SYS>> [/INST]",
        ),
        (
            [RequestMessage(role="system", content="Test")],
            "<s>[INST] <<SYS>> Test <</SYS>> [/INST]",
        ),
        (
            [
                RequestMessage(role="system", content="Test"),
                RequestMessage(role="user", content="Test"),
                RequestMessage(role="assistant", content="Test"),
                RequestMessage(role="system", content="Test"),
            ],
            "<s>[INST] <<SYS>> Test <</SYS>> Test [/INST] Test </s><s>[INST] <<SYS>> "
            "Test <</SYS>> [/INST]",
        ),
        (
            [
                RequestMessage(role="system", content="Test"),
                RequestMessage(role="user", content="Test"),
                RequestMessage(role="assistant", content="Test"),
                RequestMessage(role="user", content="Test"),
                RequestMessage(role="assistant", content="Test"),
                RequestMessage(role="user", content="Test"),
                RequestMessage(role="assistant", content="Test"),
            ],
            "<s>[INST] <<SYS>> Test <</SYS>> Test [/INST] Test </s><s>[INST] Test [/INST] "
            "Test </s><s>[INST] Test [/INST] Test ",
        ),
    ],
)
def test_valid_parsing(messages, expected_output):
    route_config = RouteConfig(**chat_config())

    assert (
        MosaicMLProvider(route_config)._parse_chat_messages_to_prompt(messages=messages)
        == expected_output
    )


@pytest.mark.parametrize(
    "messages",
    [
        [RequestMessage(role="invalid_role", content="Test")],
        [RequestMessage(role="another_invalid_role", content="Test")],
        [
            RequestMessage(role="system", content="Test"),
            RequestMessage(role="user", content="Test"),
            RequestMessage(role="invalid_role", content="Test"),
        ],
    ],
)
def test_invalid_role_submitted_raises(messages):
    route_config = RouteConfig(**chat_config())
    with pytest.raises(
        MlflowException, match=".*Must be one of 'system', 'user', or 'assistant'.*"
    ):
        MosaicMLProvider(route_config)._parse_chat_messages_to_prompt(messages)


def unsupported_mosaic_chat_model_config():
    return {
        "name": "chat",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "provider": "mosaicml",
            "name": "unsupported",
            "config": {
                "mosaicml_api_key": "key",
            },
        },
    }


def test_unsupported_model_name_raises_in_chat_parsing_route_configuration():
    with pytest.raises(MlflowException, match="An invalid model has been specified"):
        RouteConfig(**unsupported_mosaic_chat_model_config())


@pytest.mark.asyncio
async def test_completions_raises_with_invalid_max_tokens_too_large():
    config = completions_config()
    error_msg = {
        "message": "Error: prompt token count (29) + max output tokens (4085) cannot "
        "exceed 4096. Please reduce the length of your prompt and/or max "
        "output tokens generated.\n"
    }
    resp = {
        "message": error_msg["message"],
    }

    with mock.patch("aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp, status=500)):
        provider = MosaicMLProvider(RouteConfig(**config))
        payload = {
            "prompt": "How many puffins can fit on the flight deck of a Nimitz class "
            "aircraft carrier?",
            "max_tokens": 4080,
        }
        with pytest.raises(HTTPException, match=r".*") as e:
            await provider.completions(completions.RequestPayload(**payload))
        assert error_msg == e.value.detail
        assert e.value.status_code == 422


@pytest.mark.asyncio
async def test_chat_raises_with_invalid_max_tokens_too_large():
    config = chat_config()
    error_msg = {
        "message": "Error: max output tokens is limited to 4096 but 5000 was requested. "
        "Please use a lower token count.\n"
    }
    resp = {
        "message": error_msg["message"],
    }
    with mock.patch("aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp, status=500)):
        provider = MosaicMLProvider(RouteConfig(**config))
        payload = {
            "messages": [
                {"role": "system", "content": "You're an astronaut."},
                {"role": "user", "content": "When do you go to space next?"},
            ],
            "max_tokens": 5000,
        }
        with pytest.raises(HTTPException, match=r".*") as e:
            await provider.chat(chat.RequestPayload(**payload))
        assert error_msg == e.value.detail
        assert e.value.status_code == 422
