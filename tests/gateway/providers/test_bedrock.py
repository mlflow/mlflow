from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import (
    AmazonBedrockConfig,
    AWSBaseConfig,
    AWSIdAndKey,
    AWSRole,
    RouteConfig,
)
from mlflow.gateway.providers.bedrock import AmazonBedrockModelProvider, AmazonBedrockProvider
from mlflow.gateway.schemas import completions

from tests.gateway.providers.test_anthropic import (
    completions_response as anthropic_completions_response,
)
from tests.gateway.providers.test_anthropic import (
    parsed_completions_response as anthropic_parsed_completions_response,
)
from tests.gateway.providers.test_cohere import completions_response as cohere_completions_response


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
                    "text": "\nIt looks like",
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
                                "token": "<|endoftext|>",
                                "logprob": -0.01683046855032444,
                                "raw_logprob": -0.01683046855032444,
                            },
                            "topTokens": None,
                            "textRange": {"start": 14, "end": 14},
                        },
                    ],
                },
                "finishReason": {"reason": "endoftext"},
            }
        ],
    }


def ai21_parsed_completion_response(mdl):
    return {
        "id": None,
        "object": "text_completion",
        "created": 1677858242,
        "model": mdl,
        "choices": [
            {
                "text": "\nIt looks like",
                "index": 0,
                "finish_reason": None,
            }
        ],
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    }


bedrock_model_provider_fixtures = [
    {
        "provider": AmazonBedrockModelProvider.ANTHROPIC,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "anthropic.claude-v1",
            },
        },
        "response": anthropic_completions_response(),
        "expected": anthropic_parsed_completions_response(),
        "request": {"prompt": "How does a car work?", "max_tokens": 200},
        "model_request": {
            "max_tokens_to_sample": 200,
            "prompt": "\n\nHuman: How does a car work?\n\nAssistant:",
            "stop_sequences": ["\n\nHuman:"],
        },
    },
    {
        "provider": AmazonBedrockModelProvider.ANTHROPIC,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "anthropic.claude-v2",
            },
        },
        "response": anthropic_completions_response(),
        "expected": anthropic_parsed_completions_response(),
        "request": {"prompt": "How does a car work?", "max_tokens": 200},
        "model_request": {
            "max_tokens_to_sample": 200,
            "prompt": "\n\nHuman: How does a car work?\n\nAssistant:",
            "stop_sequences": ["\n\nHuman:"],
        },
    },
    {
        "provider": AmazonBedrockModelProvider.ANTHROPIC,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "anthropic.claude-instant-v1",
            },
        },
        "response": anthropic_completions_response(),
        "expected": anthropic_parsed_completions_response(),
        "request": {"prompt": "How does a car work?", "max_tokens": 200},
        "model_request": {
            "max_tokens_to_sample": 200,
            "prompt": "\n\nHuman: How does a car work?\n\nAssistant:",
            "stop_sequences": ["\n\nHuman:"],
        },
    },
    {
        "provider": AmazonBedrockModelProvider.AMAZON,
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
            "n": 1,
            "temperature": 0.5,
            "stop": ["foobar"],
            "max_tokens": 1000,
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
            "id": None,
            "object": "text_completion",
            "created": 1677858242,
            "model": "amazon.titan-tg1-large",
            "choices": [
                {
                    "text": "\nThis is a test",
                    "index": 0,
                    "finish_reason": None,
                }
            ],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        },
        "model_request": {
            "inputText": "This is a test",
            "textGenerationConfig": {
                "temperature": 0.25,
                "stopSequences": ["foobar"],
                "maxTokenCount": 1000,
            },
        },
    },
    {
        "provider": AmazonBedrockModelProvider.AI21,
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
        "expected": ai21_parsed_completion_response("ai21.j2-ultra"),
        "model_request": {"prompt": "This is a test"},
    },
    {
        "provider": AmazonBedrockModelProvider.AI21,
        "config": {
            "name": "completions",
            "route_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "ai21.j2-mid",
            },
        },
        "request": {"prompt": "This is a test", "n": 2, "max_tokens": 1000, "stop": ["foobar"]},
        "response": ai21_completion_response(),
        "expected": ai21_parsed_completion_response("ai21.j2-mid"),
        "model_request": {
            "prompt": "This is a test",
            "stopSequences": ["foobar"],
            "maxTokens": 1000,
            "numResults": 2,
        },
    },
    {
        "provider": AmazonBedrockModelProvider.COHERE,
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
        "model_request": {},
    },
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


@pytest.mark.parametrize(("aws_config", "expected"), bedrock_aws_configs)
def test_bedrock_aws_config(aws_config, expected):
    assert isinstance(
        AmazonBedrockConfig.parse_obj({"aws_config": aws_config}).aws_config, expected
    )


@pytest.mark.parametrize(
    ("provider", "config"),
    [(fix["provider"], fix["config"]) for fix in bedrock_model_provider_fixtures][:1],
)
@pytest.mark.parametrize("aws_config", [c for c, _ in bedrock_aws_configs])
def test_bedrock_aws_client(provider, config, aws_config):
    with mock.patch("boto3.Session", return_value=mock.Mock()) as mock_session:
        mock_client = mock.Mock()
        mock_assume_role = mock.Mock()
        mock_assume_role.return_value = mock.MagicMock()

        mock_session.return_value.client = mock_client
        mock_client.return_value.assume_role = mock_assume_role

        provider = AmazonBedrockProvider(
            RouteConfig(**_merge_model_and_aws_config(config, aws_config))
        )
        provider.get_bedrock_client()

        if "aws_region" in aws_config:
            _assert_any_call_at_least(mock_session, region_name=aws_config["aws_region"])

        if "aws_role_arn" in aws_config:
            _assert_any_call_at_least(mock_client, service_name="sts")
            _assert_any_call_at_least(mock_assume_role, RoleArn=aws_config["aws_role_arn"])
            _assert_any_call_at_least(mock_client, service_name="bedrock-runtime")

        elif {"aws_secret_access_key", "aws_access_key_id"} <= set(aws_config):
            _assert_any_call_at_least(mock_client, service_name="bedrock-runtime")
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
    ("provider", "config", "payload", "response", "expected", "model_request"),
    [
        pytest.param(
            fix["provider"],
            fix["config"],
            fix["request"],
            fix["response"],
            fix["expected"],
            fix["model_request"],
            marks=[]
            if fix["provider"] is not AmazonBedrockModelProvider.COHERE
            else pytest.mark.skip("Cohere isn't available on Amazon Bedrock yet"),
        )
        for fix in bedrock_model_provider_fixtures
    ],
)
async def test_bedrock_request_response(
    provider, config, payload, response, expected, model_request, aws_config
):
    with mock.patch("time.time", return_value=1677858242), mock.patch(
        "mlflow.gateway.providers.bedrock.AmazonBedrockProvider._request", return_value=response
    ) as mock_request:
        if not expected:
            pytest.skip("no expected value")

        expected["model"] = config["model"]["name"]

        provider = AmazonBedrockProvider(
            RouteConfig(**_merge_model_and_aws_config(config, aws_config))
        )
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == expected

        mock_request.assert_called_once()
        mock_request.assert_called_once_with(model_request)
