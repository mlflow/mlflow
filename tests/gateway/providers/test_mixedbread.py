from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
from mlflow.gateway.providers.mixedbread import MixedBreadProvider
from mlflow.gateway.schemas import embeddings

from tests.gateway.tools import MockAsyncResponse


def embeddings_config():
    return {
        "name": "embeddings",
        "route_type": "llm/v1/embeddings",
        "model": {
            "provider": "mixedbread",
            "name": "UAE-Large-V1",
            "config": {
                "mixedbread_api_key": "key",
            },
        },
    }


def embeddings_response():
    return {
        "id": "bc57846a-3e56-4327-8acc-588ca1a37b8a",
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
                "truncated": True,  # Corrected from 'true' to 'True'
            }
        ],
        "model": "UAE-Large-V1",
        "usage": {"prompt_tokens": 420, "total_tokens": 420},
        "normalized": True,  # Corrected from 'true' to 'True'
    }


@pytest.mark.asyncio
async def test_embeddings():
    resp = embeddings_response()
    config = embeddings_config()
    with mock.patch("time.time", return_value=1677858242), mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = MixedBreadProvider(RouteConfig(**config))
        payload = {
            "input": ["Who is german and likes bread?", "Everybody in Germany."],
            "model": "UAE-Large-V1",
        }
        response = await provider.embeddings(embeddings.RequestPayload(**payload))

        expected_response = {
            "object": "list",  # Corrected to "list" from list
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
            "model": "UAE-Large-V1",
            "usage": {"prompt_tokens": 420, "total_tokens": 420},
        }

        assert jsonable_encoder(response) == expected_response

        mock_post.assert_called_once_with(
            "https://api.mixedbread.ai/v1/embeddings",
            json={
                "input": ["Who is german and likes bread?", "Everybody in Germany."],
                "model": "UAE-Large-V1",
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )
