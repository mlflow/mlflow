import os
from unittest import mock

import pytest
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.constants import MLFLOW_AI_GATEWAY_ANTHROPIC_MAXIMUM_MAX_TOKENS
from mlflow.gateway.providers.bedrock import AWSBedrockModelProvider, AWSBedrockProvider
from mlflow.gateway.schemas import chat, completions, embeddings
from tests.gateway.providers.test_anthropic import (
    completions_response as anthro_completions_response,
)
from tests.gateway.providers.test_anthropic import (
    parsed_completions_response as anthro_parsed_completions_response,
)
from tests.gateway.providers.test_cohere import (
    completions_response as cohere_completions_response,
)
from tests.gateway.tools import MockAsyncResponse

bedrock_model_provider_fixtures = [
    {
        "provider": AWSBedrockModelProvider.ANTHROPIC,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "anthropic.claude-v1",
                "config": {"aws_config": {"aws_region": "us-east-1"}},
            },
        },
        "response": anthro_completions_response(),
        "expected": anthro_parsed_completions_response(),
        "request": {"prompt": "How does a car work?", "max_tokens": 200},
    },
    {
        "provider": AWSBedrockModelProvider.ANTHROPIC,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "anthropic.claude-v2",
                "config": {"aws_config": {"aws_region": "us-east-1"}},
            },
        },
        "response": anthro_completions_response(),
        "expected": anthro_parsed_completions_response(),
        "request": {"prompt": "How does a car work?", "max_tokens": 200},
    },
    {
        "provider": AWSBedrockModelProvider.ANTHROPIC,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "anthropic.claude-instant-v1",
                "config": {"aws_config": {"aws_region": "us-east-1"}},
            },
        },
        "response": anthro_completions_response(),
        "expected": anthro_parsed_completions_response(),
        "request": {"prompt": "How does a car work?", "max_tokens": 200},
    },
    {
        "provider": AWSBedrockModelProvider.AMAZON,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "amazon.titan-tg1-large",
                "config": {"aws_config": {"aws_region": "us-east-1"}},
            },
        },
        "request": {
            "prompt": "This is a test",
        },
        "response": {},
        "expected": {},
    },
    {
        "provider": AWSBedrockModelProvider.AI21,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "ai21.j2-ultra",
                "config": {"aws_config": {"aws_region": "us-east-1"}},
            },
        },
        "request": {
            "prompt": "This is a test",
        },
        "response": {},
        "expected": {},
    },
    {
        "provider": AWSBedrockModelProvider.AI21,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "ai21.j2-mid",
                "config": {"aws_config": {"aws_region": "us-east-1"}},
            },
        },
        "request": {
            "prompt": "This is a test",
        },
        "response": {},
        "expected": {},
    },
    {
        "provider": AWSBedrockModelProvider.COHERE,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "cohere.command",
                "config": {"aws_config": {"aws_region": "us-east-1"}},
            },
        },
        "request": {
            "prompt": "This is a test",
        },
        "response": cohere_completions_response(),
        "expected": {
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
                "model": "command",
                "route_type": "llm/v1/completions",
            },
        },
    },
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,config,payload,response,expected",
    [
        (fix["provider"], fix["config"], fix["request"], fix["response"], fix["expected"])
        for fix in bedrock_model_provider_fixtures
    ],
)
async def test_bedrock_request_response(provider, config, payload, response, expected):
    with mock.patch(
        "mlflow.gateway.providers.bedrock.AWSBedrockProvider._make_request", return_value=response
    ) as mock_post:
        print(config)
        if not expected:
            pytest.skip("no expected value")
        expected["metadata"]["model"] = config["model"]["name"]
        provider = AWSBedrockProvider(RouteConfig(**config))
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == expected
        mock_post.assert_called_once()


@pytest.mark.skipif(
    "MLFLOW_AI_GATEWAY_LIVE_TEST_BEDROCK" not in os.environ, reason="don't run live tests"
)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,config,payload",
    [
        pytest.param(
            fix["provider"],
            fix["config"],
            fix["request"],
            marks=[]
            if fix["provider"] is not AWSBedrockModelProvider.COHERE
            else pytest.mark.skip("Cohere isn't availabe on AWS Bedrock yet"),
        )
        for fix in bedrock_model_provider_fixtures
    ],
)
async def test_live_call_to_aws(provider, config, payload):
    provider = AWSBedrockProvider(RouteConfig(**config))
    response = await provider.completions(completions.RequestPayload(**payload))
    assert len(response.candidates) > 0
