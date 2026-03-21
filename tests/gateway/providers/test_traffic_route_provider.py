from typing import Any

import pytest

from mlflow.gateway.config import EndpointConfig
from mlflow.gateway.providers.base import TrafficRouteProvider

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


def _get_traffic_route_provider(endpoint_config: dict[str, Any]) -> TrafficRouteProvider:
    """
    Returns a traffic route provider that forwards 100% traffic to the endpoint
    configured by `endpoint_config`
    """
    return TrafficRouteProvider(
        configs=[EndpointConfig(**endpoint_config)],
        traffic_splits=[100],
        routing_strategy="TRAFFIC_SPLIT",
    )


@pytest.mark.asyncio
async def test_chat():
    config = chat_config()
    provider = _get_traffic_route_provider(config)
    await _run_test_chat(provider)


@pytest.mark.parametrize("resp", [chat_stream_response(), chat_stream_response_incomplete()])
@pytest.mark.asyncio
async def test_chat_stream(resp):
    config = chat_config()
    provider = _get_traffic_route_provider(config)
    await _run_test_chat_stream(resp, provider)


@pytest.mark.parametrize("resp", [completions_response(), chat_response()])
@pytest.mark.asyncio
async def test_completions(resp):
    config = completions_config()
    provider = _get_traffic_route_provider(config)
    await _run_test_completions(resp, provider)


@pytest.mark.parametrize(
    "resp", [completions_stream_response(), completions_stream_response_incomplete()]
)
@pytest.mark.asyncio
async def test_completions_stream(resp):
    config = completions_config()
    provider = _get_traffic_route_provider(config)
    await _run_test_completions_stream(resp, provider)


@pytest.mark.asyncio
async def test_embeddings():
    config = embedding_config()
    provider = _get_traffic_route_provider(config)
    await _run_test_embeddings(provider)
