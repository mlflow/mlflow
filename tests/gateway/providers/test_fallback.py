from typing import Any
from unittest import mock

import pytest

from mlflow.entities.gateway_endpoint import FallbackStrategy
from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.base import FallbackProvider
from mlflow.gateway.schemas import chat, embeddings

from tests.gateway.providers.test_openai import (
    _run_test_chat,
    _run_test_chat_stream,
    _run_test_completions,
    _run_test_completions_stream,
    _run_test_embeddings,
    chat_config,
    chat_response,
    chat_stream_response,
    chat_stream_response_incomplete,
    completions_config,
    completions_response,
    completions_stream_response,
    completions_stream_response_incomplete,
    embedding_config,
)
from tests.gateway.tools import MockAsyncResponse, MockAsyncStreamingResponse, mock_http_client


def _get_fallback_provider(
    endpoint_configs: list[dict[str, Any]], max_attempts: int | None = None
) -> FallbackProvider:
    configs = [EndpointConfig(**config) for config in endpoint_configs]
    return FallbackProvider(
        configs=configs,
        strategy=FallbackStrategy.SEQUENTIAL,
        max_attempts=max_attempts,
    )


@pytest.mark.asyncio
async def test_fallback_chat_first_provider_succeeds():
    config = chat_config()
    provider = _get_fallback_provider([config])
    await _run_test_chat(provider)


@pytest.mark.asyncio
async def test_fallback_chat_second_provider_succeeds():
    config1 = chat_config()
    config2 = chat_config()
    config2["name"] = "chat-fallback"

    provider = _get_fallback_provider([config1, config2])

    resp = chat_response()
    mock_client = mock_http_client(MockAsyncResponse(resp))

    with mock.patch("aiohttp.ClientSession") as mock_session:
        mock_session.return_value = mock_client
        mock_client.post.side_effect = [
            Exception("First provider failed"),
            MockAsyncResponse(resp),
        ]

        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Tell me a joke"}],
            temperature=0.5,
        )

        response = await provider.chat(payload)
        assert response.choices[0].message.content == "\n\nThis is a test!"
        # Verify we tried twice (first failed, second succeeded)
        assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_fallback_chat_all_providers_fail():
    config1 = chat_config()
    config2 = chat_config()
    config2["name"] = "chat-fallback"

    provider = _get_fallback_provider([config1, config2])

    # Mock both providers to fail
    with mock.patch("aiohttp.ClientSession") as mock_session:
        mock_client = mock_http_client(MockAsyncResponse({}))
        mock_session.return_value = mock_client
        mock_client.post.side_effect = [
            Exception("First provider failed"),
            Exception("Second provider failed"),
        ]

        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Tell me a joke"}],
            temperature=0.5,
        )

        with pytest.raises(Exception, match="All 2 fallback attempts failed"):
            await provider.chat(payload)

        # Verify we tried both providers
        assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_fallback_chat_max_attempts():
    config1 = chat_config()
    config2 = chat_config()
    config2["name"] = "chat-fallback2"
    config3 = chat_config()
    config3["name"] = "chat-fallback3"

    # Set max_attempts to 2, even though we have 3 providers
    provider = _get_fallback_provider([config1, config2, config3], max_attempts=2)

    # Mock all providers to fail
    with mock.patch("aiohttp.ClientSession") as mock_session:
        mock_client = mock_http_client(MockAsyncResponse({}))
        mock_session.return_value = mock_client
        mock_client.post.side_effect = [
            Exception("First provider failed"),
            Exception("Second provider failed"),
            Exception("Third provider failed"),
        ]

        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Tell me a joke"}],
            temperature=0.5,
        )

        with pytest.raises(Exception, match="All 2 fallback attempts failed"):
            await provider.chat(payload)

        # Verify we only tried 2 providers (max_attempts=2)
        assert mock_client.post.call_count == 2


@pytest.mark.parametrize("resp", [chat_stream_response(), chat_stream_response_incomplete()])
@pytest.mark.asyncio
async def test_fallback_chat_stream(resp):
    config = chat_config()
    provider = _get_fallback_provider([config])
    await _run_test_chat_stream(resp, provider)


@pytest.mark.asyncio
async def test_fallback_chat_stream_with_fallback():
    config1 = chat_config()
    config2 = chat_config()
    config2["name"] = "chat-fallback"

    provider = _get_fallback_provider([config1, config2])

    resp = chat_stream_response()
    mock_client = mock_http_client(MockAsyncStreamingResponse(resp))

    # Mock the first provider to fail and second to succeed
    with mock.patch("aiohttp.ClientSession") as mock_session:
        mock_session.return_value = mock_client
        mock_client.post.side_effect = [
            Exception("First provider failed"),
            MockAsyncStreamingResponse(resp),
        ]

        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Tell me a joke"}],
            temperature=0.5,
            stream=True,
        )

        chunks = [chunk async for chunk in provider.chat_stream(payload)]

        assert len(chunks) > 0
        assert mock_client.post.call_count == 2


@pytest.mark.parametrize("resp", [completions_response(), chat_response()])
@pytest.mark.asyncio
async def test_fallback_completions(resp):
    config = completions_config()
    provider = _get_fallback_provider([config])
    await _run_test_completions(resp, provider)


@pytest.mark.parametrize(
    "resp", [completions_stream_response(), completions_stream_response_incomplete()]
)
@pytest.mark.asyncio
async def test_fallback_completions_stream(resp):
    config = completions_config()
    provider = _get_fallback_provider([config])
    await _run_test_completions_stream(resp, provider)


@pytest.mark.asyncio
async def test_fallback_embeddings():
    config = embedding_config()
    provider = _get_fallback_provider([config])
    await _run_test_embeddings(provider)


@pytest.mark.asyncio
async def test_fallback_embeddings_with_fallback():
    config1 = embedding_config()
    config2 = embedding_config()
    config2["name"] = "embeddings-fallback"

    provider = _get_fallback_provider([config1, config2])

    embedding_response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8,
        },
    }

    mock_client = mock_http_client(MockAsyncResponse(embedding_response))

    # Mock the first provider to fail and second to succeed
    with mock.patch("aiohttp.ClientSession") as mock_session:
        mock_session.return_value = mock_client
        mock_client.post.side_effect = [
            Exception("First provider failed"),
            MockAsyncResponse(embedding_response),
        ]

        payload = embeddings.RequestPayload(input="Test input")
        response = await provider.embeddings(payload)

        assert response.data[0].embedding == [0.1, 0.2, 0.3]
        # Verify we tried twice (first failed, second succeeded)
        assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_fallback_provider_empty_configs():
    with pytest.raises(Exception, match="must contain at least one endpoint configuration"):
        FallbackProvider(configs=[], max_attempts=None, strategy=FallbackStrategy.SEQUENTIAL)


@pytest.mark.asyncio
async def test_fallback_provider_passthrough():
    config = chat_config()
    provider = _get_fallback_provider([config])

    passthrough_response = {"id": "test-id", "result": "success"}
    mock_client = mock_http_client(MockAsyncResponse(passthrough_response))

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        from mlflow.gateway.providers.base import PassthroughAction

        action = PassthroughAction.OPENAI_CHAT
        payload = {"messages": [{"role": "user", "content": "Hello"}]}

        response = await provider.passthrough(action, payload)
        assert response["result"] == "success"
