from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.constants import (
    MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
    MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS,
    MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS,
)
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.anthropic import AnthropicProvider
from mlflow.gateway.schemas import chat, completions, embeddings

from tests.gateway.tools import MockAsyncResponse, MockAsyncStreamingResponse


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
        "headers": {"Content-Type": "application/json"},
    }


def completions_config():
    return {
        "name": "completions",
        "endpoint_type": "llm/v1/completions",
        "model": {
            "provider": "anthropic",
            "name": "claude-instant-1",
            "config": {
                "anthropic_api_key": "key",
            },
        },
    }


def parsed_completions_response():
    return {
        "id": None,
        "object": "text_completion",
        "created": 1677858242,
        "model": "claude-instant-1.1",
        "choices": [
            {
                "text": "Here is a basic overview of how a car works:\n\n1. The engine. "
                "The engine is the power source that makes the car move.",
                "index": 0,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }


@pytest.mark.asyncio
async def test_completions():
    resp = completions_response()
    config = completions_config()
    with (
        mock.patch("time.time", return_value=1677858242),
        mock.patch("aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)) as mock_post,
    ):
        provider = AnthropicProvider(RouteConfig(**config))
        payload = {
            "prompt": "How does a car work?",
            "max_tokens": 200,
            "stop": ["foobazbardiddly"],
            "temperature": 1.0,
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == parsed_completions_response()
        mock_post.assert_called_once_with(
            "https://api.anthropic.com/v1/complete",
            json={
                "model": "claude-instant-1",
                "temperature": 0.5,
                "max_tokens_to_sample": 200,
                "prompt": "\n\nHuman: How does a car work?\n\nAssistant:",
                "stop_sequences": ["foobazbardiddly"],
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.asyncio
async def test_completions_with_default_max_tokens():
    resp = completions_response()
    config = completions_config()
    with (
        mock.patch("time.time", return_value=1677858242),
        mock.patch("aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)) as mock_post,
    ):
        provider = AnthropicProvider(RouteConfig(**config))
        payload = {"prompt": "How does a car work?"}
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == parsed_completions_response()
        mock_post.assert_called_once_with(
            "https://api.anthropic.com/v1/complete",
            json={
                "model": "claude-instant-1",
                "temperature": 0.0,
                "max_tokens_to_sample": 8192,
                "prompt": "\n\nHuman: How does a car work?\n\nAssistant:",
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.asyncio
async def test_completions_throws_with_invalid_max_tokens_too_large():
    config = completions_config()
    provider = AnthropicProvider(RouteConfig(**config))
    payload = {"prompt": "Would Fozzie or Kermet win in a fight?", "max_tokens": 1000001}
    with pytest.raises(AIGatewayException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert (
        "Invalid value for max_tokens: cannot exceed "
        f"{MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS}" in e.value.detail
    )
    assert e.value.status_code == 422


@pytest.mark.asyncio
async def test_completions_throws_with_unsupported_n():
    config = completions_config()
    provider = AnthropicProvider(RouteConfig(**config))
    payload = {
        "prompt": "Would Fozzie or Kermet win in a fight?",
        "n": 5,
        "max_tokens": 10,
    }
    with pytest.raises(AIGatewayException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "'n' must be '1' for the Anthropic provider" in e.value.detail
    assert e.value.status_code == 422


@pytest.mark.asyncio
async def test_completions_throws_with_top_p_defined():
    config = completions_config()
    provider = AnthropicProvider(RouteConfig(**config))
    payload = {"prompt": "Would Fozzie or Kermet win in a fight?", "max_tokens": 500, "top_p": 0.6}
    with pytest.raises(AIGatewayException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "Cannot set both 'temperature' and 'top_p' parameters. Please" in e.value.detail
    assert e.value.status_code == 422


@pytest.mark.asyncio
async def test_completions_throws_with_stream_set_to_true():
    config = completions_config()
    provider = AnthropicProvider(RouteConfig(**config))
    payload = {
        "prompt": "Could the Millennium Falcon fight a Borg Cube and win?",
        "max_tokens": 5000,
        "stream": "true",
    }
    with pytest.raises(AIGatewayException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "Setting the 'stream' parameter to 'true' is not supported" in e.value.detail
    assert e.value.status_code == 422


def chat_config():
    return {
        "name": "chat",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "provider": "anthropic",
            "name": "claude-2.1",
            "config": {
                "anthropic_api_key": "key",
            },
        },
    }


def chat_response():
    # see https://docs.anthropic.com/claude/reference/messages_post
    return {
        "content": [{"text": "Response message", "type": "text"}],
        "id": "test-id",
        "model": "claude-2.1",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 10, "output_tokens": 25},
    }


def chat_payload(stream: bool = False):
    payload = {
        "messages": [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Message 2"},
            {"role": "user", "content": "Message 3"},
        ],
        "temperature": 0.5,
    }
    if stream:
        payload["stream"] = True
    return payload


def chat_stream_response():
    return [
        b"event: message_start\n",
        b'data: {"type": "message_start", "message": {"id": "test-id", "type": "message", '
        b'"role": "assistant", "content": [], "model": "claude-2.1", "stop_reason": null, '
        b'"stop_sequence": null, "usage": {"input_tokens": 25, "output_tokens": 1}}}\n',
        b"\n",
        b"event: content_block_start\n",
        b'data: {"type": "content_block_start", "index":0, "content_block": {"type": "text", '
        b'"text": ""}}\n',
        b"\n",
        b"event: ping\n",
        b'data: {"type": "ping"}\n',
        b"\n",
        b"event: content_block_delta\n",
        b'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", '
        b'"text": "Hello"}}\n',
        b"\n",
        b"event: content_block_delta\n",
        b'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", '
        b'"text": "!"}}\n',
        b"\n",
        b"event: content_block_stop\n",
        b'data: {"type": "content_block_stop", "index": 0}\n',
        b"\n",
        b"event: message_delta\n",
        b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", '
        b'"stop_sequence":null, "usage":{"output_tokens": 15}}}\n',
        b"\n",
        b"event: message_stop\n",
        b'data: {"type": "message_stop"}\n',
    ]


@pytest.mark.asyncio
async def test_chat():
    resp = chat_response()
    config = chat_config()
    with (
        mock.patch("time.time", return_value=1677858242),
        mock.patch("aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)) as mock_post,
    ):
        provider = AnthropicProvider(RouteConfig(**config))
        payload = chat_payload()
        response = await provider.chat(chat.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "claude-2.1",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Response message", "type": "text"}],
                        "tool_calls": None,
                        "refusal": None,
                    },
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 25,
                "total_tokens": 35,
            },
        }
        mock_post.assert_called_once_with(
            "https://api.anthropic.com/v1/messages",
            json={
                "model": "claude-2.1",
                "messages": [
                    {"role": "user", "content": "Message 1"},
                    {"role": "assistant", "content": "Message 2"},
                    {"role": "user", "content": "Message 3"},
                ],
                "system": "System message",
                "max_tokens": MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
                "temperature": 0.25,
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.asyncio
async def test_chat_stream():
    resp = chat_stream_response()
    config = chat_config()

    with (
        mock.patch("time.time", return_value=1677858242),
        mock.patch(
            "aiohttp.ClientSession.post", return_value=MockAsyncStreamingResponse(resp)
        ) as mock_post,
    ):
        provider = AnthropicProvider(RouteConfig(**config))
        payload = chat_payload(stream=True)
        response = provider.chat_stream(chat.RequestPayload(**payload))
        chunks = [jsonable_encoder(chunk) async for chunk in response]
        assert chunks == [
            {
                "id": "test-id",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "claude-2.1",
                "choices": [
                    {"index": 0, "finish_reason": None, "delta": {"role": None, "content": ""}}
                ],
            },
            {
                "id": "test-id",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "claude-2.1",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": None,
                        "delta": {"role": None, "content": "Hello"},
                    }
                ],
            },
            {
                "id": "test-id",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "claude-2.1",
                "choices": [
                    {"index": 0, "finish_reason": None, "delta": {"role": None, "content": "!"}}
                ],
            },
            {
                "id": "test-id",
                "object": "chat.completion.chunk",
                "created": 1677858242,
                "model": "claude-2.1",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "delta": {"role": None, "content": None}}
                ],
            },
        ]
        mock_post.assert_called_once_with(
            "https://api.anthropic.com/v1/messages",
            json={
                "model": "claude-2.1",
                "messages": [
                    {"role": "user", "content": "Message 1"},
                    {"role": "assistant", "content": "Message 2"},
                    {"role": "user", "content": "Message 3"},
                ],
                "system": "System message",
                "max_tokens": MLFLOW_AI_GATEWAY_ANTHROPIC_DEFAULT_MAX_TOKENS,
                "temperature": 0.25,
                "stream": True,
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


def embedding_config():
    return {
        "name": "embeddings",
        "endpoint_type": "llm/v1/embeddings",
        "model": {
            "provider": "anthropic",
            "name": "claude-1.3-100k",
            "config": {
                "anthropic_api_key": "key",
            },
        },
    }


@pytest.mark.asyncio
async def test_embeddings_are_not_supported_for_anthropic():
    config = embedding_config()
    provider = AnthropicProvider(RouteConfig(**config))
    payload = {"input": "give me that sweet, sweet vector, please."}

    with pytest.raises(AIGatewayException, match=r".*") as e:
        await provider.embeddings(embeddings.RequestPayload(**payload))
    assert "The embeddings route is not implemented for Anthropic models" in e.value.detail
    assert e.value.status_code == 501


@pytest.mark.asyncio
async def test_param_model_is_not_permitted():
    config = completions_config()
    provider = AnthropicProvider(RouteConfig(**config))
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
    provider = AnthropicProvider(RouteConfig(**config))
    payload = {"prompt": prompt}
    with pytest.raises(ValidationError, match=r"prompt"):
        await provider.completions(completions.RequestPayload(**payload))
