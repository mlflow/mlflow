from unittest import mock

import pytest
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.providers.ai21labs import AI21LabsProvider
from mlflow.gateway.schemas import completions, embeddings

from tests.gateway.tools import MockAsyncResponse


def completions_config():
    return {
        "name": "completions",
        "route_type": "llm/v1/completions",
        "model": {
            "provider": "ai21labs",
            "name": "command",
            "config": {
                "ai21labs_api_key": "key",
            },
        },
    }


def completions_response():
    return
    {
        "id": "7921a78e-d905-c9df-27e3-88e4831e3c3b",
        "prompt": {
            "text": "This is a test"
        },
        "completions": [
            {
                "data": {
                    "text": "this is a test response"
                },
                "finishReason": {
                    "reason": "length",
                    "length": 2
                }
            }
        ]
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
                "model": "command",
                "route_type": "llm/v1/completions",
            },
        }
        mock_post.assert_called_once()

@pytest.mark.asyncio
async def test_param_model_is_not_permitted():
    config = embeddings_config()
    provider = CohereProvider(RouteConfig(**config))
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
    provider = CohereProvider(RouteConfig(**config))
    payload = {"prompt": prompt}
    with pytest.raises(ValidationError, match=r"prompt"):
        await provider.completions(completions.RequestPayload(**payload))
