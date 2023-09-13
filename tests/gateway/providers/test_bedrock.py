import os
from unittest import mock

import pytest
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from mlflow.gateway.config import (
    AWSBaseConfig,
    AWSBedrockConfig,
    AWSIdAndKey,
    AWSRole,
    RouteConfig,
)
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


def ai21_completion_response():
    return {
        "id": 1234,
        "prompt": {
            "text": "This is a test",
            "tokens": [
                {
                    "generatedToken": {
                        "token": "▁This▁is▁a",
                        "logprob": -7.127955436706543,
                        "raw_logprob": -7.127955436706543,
                    },
                    "topTokens": None,
                    "textRange": {"start": 0, "end": 9},
                },
                {
                    "generatedToken": {
                        "token": "▁test",
                        "logprob": -4.926638126373291,
                        "raw_logprob": -4.926638126373291,
                    },
                    "topTokens": None,
                    "textRange": {"start": 9, "end": 14},
                },
            ],
        },
        "completions": [
            {
                "data": {
                    "text": "\nIt looks like you're running a test. How can I assist you with this test?",
                    "tokens": [
                        {
                            "generatedToken": {
                                "token": "<|newline|>",
                                "logprob": -0.021781044080853462,
                                "raw_logprob": -0.021781044080853462,
                            },
                            "topTokens": None,
                            "textRange": {"start": 0, "end": 1},
                        },
                        {
                            "generatedToken": {
                                "token": "▁It▁looks▁like",
                                "logprob": -3.2340049743652344,
                                "raw_logprob": -3.2340049743652344,
                            },
                            "topTokens": None,
                            "textRange": {"start": 1, "end": 14},
                        },
                        {
                            "generatedToken": {
                                "token": "▁you're",
                                "logprob": -1.0763493776321411,
                                "raw_logprob": -1.0763493776321411,
                            },
                            "topTokens": None,
                            "textRange": {"start": 14, "end": 21},
                        },
                        {
                            "generatedToken": {
                                "token": "▁running▁a",
                                "logprob": -3.3509609699249268,
                                "raw_logprob": -3.3509609699249268,
                            },
                            "topTokens": None,
                            "textRange": {"start": 21, "end": 31},
                        },
                        {
                            "generatedToken": {
                                "token": "▁",
                                "logprob": -0.9933311939239502,
                                "raw_logprob": -0.9933311939239502,
                            },
                            "topTokens": None,
                            "textRange": {"start": 31, "end": 32},
                        },
                        {
                            "generatedToken": {
                                "token": "test.",
                                "logprob": -0.0008049347088672221,
                                "raw_logprob": -0.0008049347088672221,
                            },
                            "topTokens": None,
                            "textRange": {"start": 32, "end": 37},
                        },
                        {
                            "generatedToken": {
                                "token": "▁How▁can▁I",
                                "logprob": -0.19963902235031128,
                                "raw_logprob": -0.19963902235031128,
                            },
                            "topTokens": None,
                            "textRange": {"start": 37, "end": 47},
                        },
                        {
                            "generatedToken": {
                                "token": "▁assist▁you▁with",
                                "logprob": -1.4962196350097656,
                                "raw_logprob": -1.4962196350097656,
                            },
                            "topTokens": None,
                            "textRange": {"start": 47, "end": 63},
                        },
                        {
                            "generatedToken": {
                                "token": "▁this▁test",
                                "logprob": -0.11712213605642319,
                                "raw_logprob": -0.11712213605642319,
                            },
                            "topTokens": None,
                            "textRange": {"start": 63, "end": 73},
                        },
                        {
                            "generatedToken": {
                                "token": "?",
                                "logprob": -0.0002649671514518559,
                                "raw_logprob": -0.0002649671514518559,
                            },
                            "topTokens": None,
                            "textRange": {"start": 73, "end": 74},
                        },
                        {
                            "generatedToken": {
                                "token": "<|endoftext|>",
                                "logprob": -0.01683046855032444,
                                "raw_logprob": -0.01683046855032444,
                            },
                            "topTokens": None,
                            "textRange": {"start": 74, "end": 74},
                        },
                    ],
                },
                "finishReason": {"reason": "endoftext"},
            }
        ],
    }


def ai21_parsed_completion_response():
    return {
        "candidates": [
            {
                "metadata": {},
                "text": "\n"
                "It looks like you're running a test. How can I "
                "assist you with this test?",
            }
        ],
        "metadata": {
            "model": "ai21.j2-ultra",
            "route_type": "llm/v1/completions",
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
    }


bedrock_model_provider_fixtures = [
    {
        "provider": AWSBedrockModelProvider.ANTHROPIC,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "anthropic.claude-v1",
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
            },
        },
        "request": {
            "prompt": "This is a test",
        },
        "response": {
            "results": [
                {
                    "tokenCount": 5,
                    "outputText": "\nThis is a test",
                    "completionReason": "FINISH",
                }
            ],
            "inputTextTokenCount": 4,
        },
        "expected": {
            "candidates": [{"metadata": {}, "text": "\nThis is a test"}],
            "metadata": {
                "model": "amazon.titan-tg1-large",
                "route_type": "llm/v1/completions",
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            },
        },
    },
    {
        "provider": AWSBedrockModelProvider.AI21,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "ai21.j2-ultra",
            },
        },
        "request": {
            "prompt": "This is a test",
        },
        "response": ai21_completion_response(),
        "expected": ai21_parsed_completion_response(),
    },
    {
        "provider": AWSBedrockModelProvider.AI21,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "ai21.j2-mid",
            },
        },
        "request": {
            "prompt": "This is a test",
        },
        "response": ai21_completion_response(),
        "expected": ai21_parsed_completion_response(),
    },
    {
        "provider": AWSBedrockModelProvider.COHERE,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "cohere.command",
            },
        },
        "request": {
            "prompt": "This is a test",
        },
        "response": cohere_completions_response(),
        "expected": {},
    },
]

bedrock_live_aws_configs = [
    {"aws_region": "us-east-1"},
    pytest.param(
        {
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "aws_session_token": os.environ.get("AWS_SESSION_TOKEN"),
        },
        marks=pytest.mark.skipif(
            "AWS_ACCESS_KEY_ID" not in os.environ or "AWS_SECRET_ACCESS_KEY" not in os.environ,
            reason="No AWS credentials in environment",
        ),
    ),
]
bedrock_aws_configs = [
    ({"aws_region": "us-east-1"}, AWSBaseConfig),
    (
        {
            "aws_region": "us-east-1",
            "aws_access_key_id": "test-access-key-id",
            "aws_secret_access_key": "test-secret-access-key",
            "aws_session_token": "test-session-token",
        },
        AWSIdAndKey,
    ),
    (
        {
            "aws_region": "us-east-1",
            "aws_access_key_id": "test-access-key-id",
            "aws_secret_access_key": "test-secret-access-key",
        },
        AWSIdAndKey,
    ),
    ({"aws_region": "us-east-1", "aws_role_arn": "test-aws-role-arn"}, AWSRole),
]


def _merge_model_and_aws_config(config, aws_config):
    return {
        **config,
        "model": {
            **config["model"],
            "config": {**config["model"].get("config", {}), "aws_config": aws_config},
        },
    }


def _assert_any_call_at_least(mobj, *args, **kwargs):
    if not mobj.call_args_list:
        raise AssertionError(f"no calls to {mobj=}")
    for call in mobj.call_args_list:
        if all(call.kwargs.get(k) == v for k, v in kwargs.items()) and all(
            call.args[i] == v for i, v in enumerate(args)
        ):
            return
    else:
        raise AssertionError(f"No valid call to {mobj=} with {args=} and {kwargs=}")


@pytest.mark.parametrize("aws_config,expected", bedrock_aws_configs)
def test_bedrock_aws_config(aws_config, expected):
    assert isinstance(AWSBedrockConfig.parse_obj(dict(aws_config=aws_config)).aws_config, expected)


@pytest.mark.parametrize(
    "provider,config",
    [(fix["provider"], fix["config"]) for fix in bedrock_model_provider_fixtures][:1],
)
@pytest.mark.parametrize("aws_config", [c[0] for c in bedrock_aws_configs])
def test_bedrock_aws_client(provider, config, aws_config):
    with mock.patch("boto3.Session", return_value=mock.Mock()) as mock_session:
        mock_client = mock.Mock()
        mock_assume_role = mock.Mock()
        mock_assume_role.return_value = mock.MagicMock()

        mock_session.return_value.client = mock_client
        mock_client.return_value.assume_role = mock_assume_role

        provider = AWSBedrockProvider(
            RouteConfig(**_merge_model_and_aws_config(config, aws_config))
        )
        provider.get_bedrock_client()

        if "aws_region" in aws_config:
            _assert_any_call_at_least(mock_session, region_name=aws_config["aws_region"])

        if "aws_role_arn" in aws_config:
            _assert_any_call_at_least(mock_client, service_name="sts")
            _assert_any_call_at_least(mock_assume_role, RoleArn=aws_config["aws_role_arn"])
            _assert_any_call_at_least(mock_client, service_name="bedrock")

        elif {"aws_secret_access_key", "aws_access_key_id"} <= set(aws_config):
            _assert_any_call_at_least(mock_client, service_name="bedrock")
            _assert_any_call_at_least(
                mock_client,
                **{
                    k: v
                    for k, v in aws_config.items()
                    if k in {"aws_secret_access_key", "aws_access_key_id"}
                },
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("aws_config", [c[0] for c in bedrock_aws_configs])
@pytest.mark.parametrize(
    "provider,config,payload,response,expected",
    [
        pytest.param(
            fix["provider"],
            fix["config"],
            fix["request"],
            fix["response"],
            fix["expected"],
            marks=[]
            if fix["provider"] is not AWSBedrockModelProvider.COHERE
            else pytest.mark.skip("Cohere isn't availabe on AWS Bedrock yet"),
        )
        for fix in bedrock_model_provider_fixtures
    ],
)
async def test_bedrock_request_response(provider, config, payload, response, expected, aws_config):
    with mock.patch(
        "mlflow.gateway.providers.bedrock.AWSBedrockProvider._make_request", return_value=response
    ) as mock_post:
        if not expected:
            pytest.skip("no expected value")
        expected["metadata"]["model"] = config["model"]["name"]
        provider = AWSBedrockProvider(
            RouteConfig(**_merge_model_and_aws_config(config, aws_config))
        )
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == expected
        mock_post.assert_called_once()


@pytest.mark.skipif(
    "MLFLOW_AI_GATEWAY_LIVE_TEST_BEDROCK" not in os.environ, reason="don't run live tests"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("aws_config", bedrock_live_aws_configs)
@pytest.mark.parametrize(
    "provider,config,payload,expected",
    [
        pytest.param(
            fix["provider"],
            fix["config"],
            fix["request"],
            fix["expected"],
            marks=[]
            if fix["provider"] is not AWSBedrockModelProvider.COHERE
            else pytest.mark.skip("Cohere isn't availabe on AWS Bedrock yet"),
        )
        for fix in bedrock_model_provider_fixtures
    ],
)
async def test_live_call_to_aws(provider, config, payload, expected, aws_config):
    config["model"]["config"] = {"aws_config": aws_config}
    provider = AWSBedrockProvider(RouteConfig(**config))
    response = await provider.completions(completions.RequestPayload(**payload))
    if not expected:
        raise ValueError("asdf")

    # print([provider, response.dict()])
    assert len(response.candidates) > 0
