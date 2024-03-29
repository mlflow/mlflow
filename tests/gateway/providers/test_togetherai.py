import math
from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
from mlflow.gateway.providers.togetherai import TogetherAIProvider
from mlflow.gateway.schemas import chat, completions, embeddings

from tests.gateway.tools import MockAsyncResponse

import logging
import os 

def completions_config():
    return {
        "name": "completions",
        "route_type": "llm/v1/completions",
        "model": {
            "provider": "togetherai",
            "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "config": {
                "togetherai_api_key": "key",
            },
        },
    }


def completions_response():
    return {
        "id": "8447f286bbdb67b3-SJC",
        "choices": [
          {
            "index": 0,
            "text": "The capital of France is Paris. It's located in the north-central part of the country and is one of the most famous cities in the world, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and more. Paris is also the cultural, political, and economic center of France.",
            "finish_reason": None,
          }
        ],
        "usage": {
          "prompt_tokens": 16,
          "completion_tokens": 78,
          "total_tokens": 94
        },
        "created": 1705089226,
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "object": "text_completion"
    }


@pytest.mark.asyncio
async def test_completions():

    config = completions_config()
    resp = completions_response()

    with mock.patch("time.time", return_value=1677858242), mock.patch(
    "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        
        provider = TogetherAIProvider(RouteConfig(**config))

        payload = {
            "prompt": "Whats the capital of France?",
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "max_tokens": 200, 
            "temperature": 1.0,
            "n": 1,
        }

        response = await provider.completions(completions.RequestPayload(**payload))

        assert jsonable_encoder(response) == {
            "id": "8447f286bbdb67b3-SJC",
            "choices": [
              {
                "index": 0,
                "text": "The capital of France is Paris. It's located in the north-central part of the country and is one of the most famous cities in the world, known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and more. Paris is also the cultural, political, and economic center of France.",
                "finish_reason": None,
              }
            ],
            "usage": {
              "prompt_tokens": 16,
              "completion_tokens": 78,
              "total_tokens": 94
            },
            "created": 1705089226,
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "object": "text_completion"
        }

        mock_post.assert_called_once_with(
            "https://api.together.xyz/v1/completions",
            json={
                "prompt": "Whats the capital of France?",
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "max_tokens": 200,
                "temperature": 1,
                "n": 1,
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )

def embeddings_config():
    return {
        "name": "embeddings",
        "route_type": "llm/v1/embeddings",
        "model": {
            "provider": "togetherai",
            "name": "togethercomputer/m2-bert-80M-8k-retrieval",
            "config": {
                "togetherai_key": "key",
            },
        },
    }
    

def chat_config():
    return {
        "name": "chat",
        "route_type": "llm/v1/completions/chat",
        "model": {
            "provider": "togetherai",
            "name": "togethercomputer/m2-bert-80M-8k-retrieval",
            "config": {
                "togetherai_key": "key",
            },
        },
    }

