import base64
import io
import json
import struct
import zlib
from typing import Any
from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.environment_variables import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
from mlflow.gateway.config import (
    AmazonBedrockConfig,
    AWSBaseConfig,
    AWSIdAndKey,
    AWSRole,
    EndpointConfig,
)
from mlflow.gateway.exceptions import AIGatewayException
from mlflow.gateway.providers.base import PassthroughAction
from mlflow.gateway.providers.bedrock import AmazonBedrockModelProvider, AmazonBedrockProvider
from mlflow.gateway.schemas import chat, completions, embeddings

from tests.gateway.providers.test_anthropic import (
    completions_response as anthropic_completions_response,
)
from tests.gateway.providers.test_anthropic import (
    parsed_completions_response as anthropic_parsed_completions_response,
)
from tests.gateway.providers.test_cohere import completions_response as cohere_completions_response
from tests.gateway.tools import MockAsyncResponse, MockAsyncStreamingResponse, mock_http_client


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
            "endpoint_type": "llm/v1/completions",
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
            "anthropic_version": "bedrock-2023-05-31",
        },
    },
    {
        "provider": AmazonBedrockModelProvider.ANTHROPIC,
        "config": {
            "name": "completions",
            "endpoint_type": "llm/v1/completions",
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
            "anthropic_version": "bedrock-2023-05-31",
        },
    },
    {
        "provider": AmazonBedrockModelProvider.ANTHROPIC,
        "config": {
            "name": "completions",
            "endpoint_type": "llm/v1/completions",
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
            "anthropic_version": "bedrock-2023-05-31",
        },
    },
    {
        "provider": AmazonBedrockModelProvider.AMAZON,
        "config": {
            "name": "completions",
            "endpoint_type": "llm/v1/completions",
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
        # Titan names the nucleus-sampling field `topP`; forwarding `top_p` verbatim
        # sends a key Bedrock does not recognise and silently ignores.
        "provider": AmazonBedrockModelProvider.AMAZON,
        "config": {
            "name": "completions",
            "endpoint_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "amazon.titan-tg1-large",
            },
        },
        "request": {
            "prompt": "This is a test",
            "max_tokens": 1000,
            "top_p": 0.9,
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
                "maxTokenCount": 1000,
                "topP": 0.9,
            },
        },
    },
    {
        "provider": AmazonBedrockModelProvider.AI21,
        "config": {
            "name": "completions",
            "endpoint_type": "llm/v1/completions",
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
        # Same as above for Jurassic, which also names the field `topP`.
        "provider": AmazonBedrockModelProvider.AI21,
        "config": {
            "name": "completions",
            "endpoint_type": "llm/v1/completions",
            "model": {
                "provider": "bedrock",
                "name": "ai21.j2-ultra",
            },
        },
        "request": {
            "prompt": "This is a test",
            "max_tokens": 1000,
            "top_p": 0.9,
        },
        "response": ai21_completion_response(),
        "expected": ai21_parsed_completion_response("ai21.j2-ultra"),
        "model_request": {
            "prompt": "This is a test",
            "maxTokens": 1000,
            "topP": 0.9,
        },
    },
    {
        "provider": AmazonBedrockModelProvider.AI21,
        "config": {
            "name": "completions",
            "endpoint_type": "llm/v1/completions",
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
            "endpoint_type": "llm/v1/completions",
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


def test_get_provider_name():
    provider = AmazonBedrockProvider.__new__(AmazonBedrockProvider)
    assert provider.DISPLAY_NAME == "Amazon Bedrock"
    assert provider.get_provider_name() == "bedrock"


@pytest.mark.parametrize(("aws_config", "expected"), bedrock_aws_configs)
def test_bedrock_aws_config(aws_config, expected):
    assert isinstance(
        AmazonBedrockConfig.model_validate({"aws_config": aws_config}).aws_config, expected
    )


@pytest.mark.parametrize(
    ("provider", "config"),
    [(fix["provider"], fix["config"]) for fix in bedrock_model_provider_fixtures][:1],
)
@pytest.mark.parametrize("aws_config", [c for c, _ in bedrock_aws_configs])
def test_bedrock_aws_client(provider, config, aws_config):
    with mock.patch("boto3.Session") as mock_session:
        mock_client = mock.Mock()
        mock_assume_role = mock.Mock()
        mock_assume_role.return_value = mock.MagicMock()

        mock_session.return_value.client = mock_client
        mock_client.return_value.assume_role = mock_assume_role

        provider = AmazonBedrockProvider(
            EndpointConfig(**_merge_model_and_aws_config(config, aws_config))
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
    with (
        mock.patch("time.time", return_value=1677858242),
        mock.patch(
            "mlflow.gateway.providers.bedrock.AmazonBedrockProvider._request", return_value=response
        ) as mock_request,
    ):
        if not expected:
            pytest.skip("no expected value")

        expected["model"] = config["model"]["name"]

        provider = AmazonBedrockProvider(
            EndpointConfig(**_merge_model_and_aws_config(config, aws_config))
        )
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == expected

        mock_request.assert_called_once()
        mock_request.assert_called_once_with(model_request)


@pytest.mark.asyncio
@pytest.mark.parametrize("aws_config", [c[0] for c in bedrock_aws_configs])
async def test_bedrock_titan_rejects_top_p_zero(aws_config):
    # MLflow accepts top_p=0, but Titan requires topP to be strictly greater than 0.
    config = {
        "name": "completions",
        "endpoint_type": "llm/v1/completions",
        "model": {
            "provider": "bedrock",
            "name": "amazon.titan-tg1-large",
        },
    }
    provider = AmazonBedrockProvider(
        EndpointConfig(**_merge_model_and_aws_config(config, aws_config))
    )
    payload = completions.RequestPayload(prompt="This is a test", max_tokens=1000, top_p=0)

    with (
        mock.patch(
            "mlflow.gateway.providers.bedrock.AmazonBedrockProvider._request"
        ) as mock_request,
        pytest.raises(
            AIGatewayException, match="'top_p' must be greater than 0 for AWS Titan models"
        ) as exc_info,
    ):
        await provider.completions(payload)

    assert exc_info.value.status_code == 422
    mock_request.assert_not_called()


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("us.anthropic.claude-3-sonnet", AmazonBedrockModelProvider.ANTHROPIC),
        ("apac.anthropic.claude-3-haiku", AmazonBedrockModelProvider.ANTHROPIC),
        ("anthropic.claude-3-5-sonnet", AmazonBedrockModelProvider.ANTHROPIC),
        ("ai21.jamba-1-5-large-v1:0", AmazonBedrockModelProvider.AI21),
        ("cohere.embed-multilingual-v3", AmazonBedrockModelProvider.COHERE),
        ("us.amazon.nova-premier-v1:0", AmazonBedrockModelProvider.AMAZON),
    ],
)
def test_amazon_bedrock_model_provider(model_name, expected):
    provider = AmazonBedrockModelProvider.of_str(model_name)
    assert provider == expected


# ---- Converse API tests ----


def _make_converse_provider():
    """Create a provider with a mock boto3 client for Converse API tests."""

    config = {
        "name": "chat",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "provider": "bedrock",
            "name": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "config": {"aws_config": {"aws_region": "us-east-1"}},
        },
    }
    return AmazonBedrockProvider(EndpointConfig(**config))


def _converse_response():
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello from Bedrock!"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 20,
            "totalTokens": 30,
        },
    }


def _converse_response_with_tool_use():
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool_abc123",
                            "name": "add",
                            "input": {"a": 17, "b": 25},
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {
            "inputTokens": 30,
            "outputTokens": 10,
            "totalTokens": 40,
        },
    }


def _converse_stream_response():
    return {
        "stream": iter([
            {"contentBlockDelta": {"delta": {"text": "Hello"}}},
            {"contentBlockDelta": {"delta": {"text": " from Bedrock!"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30}}},
        ])
    }


def _embeddings_invoke_response():
    body = io.BytesIO(b'{"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 5}')
    return {"body": body}


@pytest.mark.asyncio
async def test_bedrock_converse_chat():

    provider = _make_converse_provider()
    mock_client = mock.Mock()
    mock_client.converse.return_value = _converse_response()

    with mock.patch.object(provider, "get_bedrock_client", return_value=mock_client):
        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Hello"}],
        )
        response = await provider.chat(payload)

    result = jsonable_encoder(response)
    assert result["choices"][0]["message"]["content"] == "Hello from Bedrock!"
    assert result["choices"][0]["message"]["role"] == "assistant"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 20
    mock_client.converse.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_converse_chat_with_reasoning():
    provider = _make_converse_provider()
    mock_client = mock.Mock()
    mock_client.converse.return_value = response = _converse_response()
    reasoning = {"reasoningContent": {"reasoningText": {"text": "Think.", "signature": "sig"}}}
    response["output"]["message"]["content"].insert(0, reasoning)

    with mock.patch.object(provider, "get_bedrock_client", return_value=mock_client):
        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Hello"}],
            custom_inputs={"thinking": {"type": "enabled", "budget_tokens": 1024}},
        )
        response = await provider.chat(payload)

    call_kwargs = mock_client.converse.call_args.kwargs
    assert call_kwargs["additionalModelRequestFields"] == payload.custom_inputs
    content = jsonable_encoder(response)["choices"][0]["message"]["content"]
    assert content[0]["summary"][0]["text"] == "Think."
    assert content[0]["signature"] == "sig"
    assert content[1] == {"type": "text", "text": "Hello from Bedrock!"}


@pytest.mark.asyncio
async def test_bedrock_converse_chat_stream():

    provider = _make_converse_provider()
    mock_client = mock.Mock()
    mock_client.converse_stream.return_value = _converse_stream_response()

    with mock.patch.object(provider, "get_bedrock_client", return_value=mock_client):
        payload = chat.RequestPayload(
            messages=[{"role": "user", "content": "Hello"}],
        )
        chunks = [jsonable_encoder(chunk) async for chunk in provider.chat_stream(payload)]

    # Should have: 2 text deltas + 1 stop + 1 usage
    assert len(chunks) == 4
    assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
    assert chunks[1]["choices"][0]["delta"]["content"] == " from Bedrock!"
    assert chunks[2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[3]["usage"]["prompt_tokens"] == 10
    mock_client.converse_stream.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_embeddings():

    config = {
        "name": "embeddings",
        "endpoint_type": "llm/v1/embeddings",
        "model": {
            "provider": "bedrock",
            "name": "amazon.titan-embed-text-v1",
            "config": {"aws_config": {"aws_region": "us-east-1"}},
        },
    }
    provider = AmazonBedrockProvider(EndpointConfig(**config))
    mock_client = mock.Mock()
    mock_client.invoke_model.return_value = _embeddings_invoke_response()

    with mock.patch.object(provider, "get_bedrock_client", return_value=mock_client):
        payload = embeddings.RequestPayload(input="Test text")
        response = await provider.embeddings(payload)

    result = jsonable_encoder(response)
    assert result["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert result["usage"]["prompt_tokens"] == 5
    mock_client.invoke_model.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_converse_with_system_message():

    provider = _make_converse_provider()
    mock_client = mock.Mock()
    mock_client.converse.return_value = _converse_response()

    with mock.patch.object(provider, "get_bedrock_client", return_value=mock_client):
        payload = chat.RequestPayload(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
        )
        await provider.chat(payload)

    call_kwargs = mock_client.converse.call_args.kwargs
    assert call_kwargs["system"] == [{"text": "You are helpful"}]
    assert len(call_kwargs["messages"]) == 1  # only user message


@pytest.mark.asyncio
async def test_bedrock_converse_chat_with_tool_call():

    provider = _make_converse_provider()
    mock_client = mock.Mock()
    mock_client.converse.return_value = _converse_response_with_tool_use()

    with mock.patch.object(provider, "get_bedrock_client", return_value=mock_client):
        payload = chat.RequestPayload(messages=[{"role": "user", "content": "add 17 and 25"}])
        response = await provider.chat(payload)

    result = jsonable_encoder(response)
    tool_calls = result["choices"][0]["message"]["tool_calls"]
    assert tool_calls[0]["function"]["name"] == "add"


@pytest.mark.asyncio
async def test_bedrock_converse_serializes_assistant_tool_call_history():
    provider = _make_converse_provider()
    mock_client = mock.Mock()
    mock_client.converse.return_value = _converse_response()

    with mock.patch.object(provider, "get_bedrock_client", return_value=mock_client):
        payload = chat.RequestPayload(
            messages=[
                {"role": "user", "content": "Compute 17+25"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tool_abc123",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a": 17, "b": 25}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "tool_abc123", "content": "42"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two integers.",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
        )
        await provider.chat(payload)

    call_kwargs = mock_client.converse.call_args.kwargs
    assistant_blocks = call_kwargs["messages"][1]["content"]
    tool_uses = [b["toolUse"] for b in assistant_blocks if "toolUse" in b]
    assert tool_uses == [{"toolUseId": "tool_abc123", "name": "add", "input": {"a": 17, "b": 25}}]
    tool_results = [
        b["toolResult"] for b in call_kwargs["messages"][2]["content"] if "toolResult" in b
    ]
    assert tool_results == [{"toolUseId": "tool_abc123", "content": [{"text": "42"}]}]
    mock_client.converse.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_converse_groups_consecutive_tool_results_into_one_user_message():
    provider = _make_converse_provider()
    mock_client = mock.Mock()
    mock_client.converse.return_value = _converse_response()

    with mock.patch.object(provider, "get_bedrock_client", return_value=mock_client):
        payload = chat.RequestPayload(
            messages=[
                {"role": "user", "content": "Compute both sums"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tool_abc123",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a": 17, "b": 25}'},
                        },
                        {
                            "id": "tool_def456",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a": 10, "b": 5}'},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "tool_abc123", "content": "42"},
                {"role": "tool", "tool_call_id": "tool_def456", "content": "15"},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two integers.",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
        )
        await provider.chat(payload)

    call_kwargs = mock_client.converse.call_args.kwargs
    assert len(call_kwargs["messages"]) == 3
    tool_results = [
        b["toolResult"] for b in call_kwargs["messages"][2]["content"] if "toolResult" in b
    ]
    assert tool_results == [
        {"toolUseId": "tool_abc123", "content": [{"text": "42"}]},
        {"toolUseId": "tool_def456", "content": [{"text": "15"}]},
    ]
    mock_client.converse.assert_called_once()


@pytest.mark.parametrize("arguments", ["not-json", "", "   "])
@pytest.mark.asyncio
async def test_bedrock_converse_rejects_invalid_assistant_tool_call_arguments(arguments):
    provider = _make_converse_provider()
    mock_client = mock.Mock()
    mock_client.converse.return_value = _converse_response()

    with mock.patch.object(provider, "get_bedrock_client", return_value=mock_client):
        payload = chat.RequestPayload(
            messages=[
                {"role": "user", "content": "Compute 17+25"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tool_bad_args",
                            "type": "function",
                            "function": {"name": "add", "arguments": arguments},
                        }
                    ],
                },
            ]
        )
        with pytest.raises(
            AIGatewayException, match="Invalid assistant tool call arguments: not valid JSON"
        ) as exc_info:
            await provider.chat(payload)

    assert exc_info.value.status_code == 422
    assert "tool_call_id=tool_bad_args" in exc_info.value.detail
    assert "tool_name=add" in exc_info.value.detail
    mock_client.converse.assert_not_called()


@pytest.mark.asyncio
async def test_bedrock_converse_rejects_assistant_tool_call_with_missing_name():
    provider = _make_converse_provider()
    mock_client = mock.Mock()
    mock_client.converse.return_value = _converse_response()

    with mock.patch.object(provider, "get_bedrock_client", return_value=mock_client):
        payload = chat.RequestPayload(
            messages=[
                {"role": "user", "content": "Compute 17+25"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tool_missing_name",
                            "type": "function",
                            "function": {"name": None, "arguments": '{"a": 17, "b": 25}'},
                        }
                    ],
                },
            ]
        )
        with pytest.raises(
            AIGatewayException, match="Invalid assistant tool call: missing function name"
        ) as exc_info:
            await provider.chat(payload)

    assert exc_info.value.status_code == 422
    assert "tool_call_id=tool_missing_name" in exc_info.value.detail
    mock_client.converse.assert_not_called()


# ---- Anthropic Messages passthrough tests ----

_CLAUDE_MODEL_ID = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"


def _make_api_key_provider(model_name: str = _CLAUDE_MODEL_ID) -> AmazonBedrockProvider:
    config = {
        "name": "claude",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "provider": "bedrock",
            "name": model_name,
            "config": {
                "aws_config": {"aws_bearer_token": "bedrock-api-key", "aws_region": "eu-west-1"}
            },
        },
    }
    return AmazonBedrockProvider(EndpointConfig(**config))


def _anthropic_messages_response():
    return {
        "id": "msg_bdrk_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [
            {"type": "thinking", "thinking": "The user said hello.", "signature": "sig"},
            {"type": "text", "text": "Hello!"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def _anthropic_stream_events():
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_bdrk_01",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-sonnet-4-5-20250929",
                "usage": {"input_tokens": 12, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "The user said hello."},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 30},
        },
        {"type": "message_stop"},
    ]


def _event_stream_message(headers: dict[str, str], payload: bytes) -> bytes:
    # AWS event stream framing: a prelude (total length, headers length, prelude CRC),
    # string-valued headers, the payload, then a CRC over the whole message.
    encoded_headers = b"".join(
        bytes([len(name)])
        + name.encode()
        + b"\x07"
        + struct.pack(">H", len(value))
        + value.encode()
        for name, value in headers.items()
    )
    prelude = struct.pack(">II", 16 + len(encoded_headers) + len(payload), len(encoded_headers))
    message = prelude + struct.pack(">I", zlib.crc32(prelude)) + encoded_headers + payload
    return message + struct.pack(">I", zlib.crc32(message))


def _bedrock_chunk(event: dict[str, Any]) -> bytes:
    return _event_stream_message(
        {":event-type": "chunk", ":content-type": "application/json", ":message-type": "event"},
        json.dumps({"bytes": base64.b64encode(json.dumps(event).encode()).decode()}).encode(),
    )


@pytest.mark.parametrize(
    ("model_name", "model_path"),
    [
        (_CLAUDE_MODEL_ID, _CLAUDE_MODEL_ID),
        (
            f"arn:aws:bedrock:eu-west-1:123456789012:inference-profile/{_CLAUDE_MODEL_ID}",
            f"arn:aws:bedrock:eu-west-1:123456789012:inference-profile%2F{_CLAUDE_MODEL_ID}",
        ),
    ],
)
@pytest.mark.asyncio
async def test_bedrock_anthropic_passthrough_posts_to_invoke(model_name, model_path):
    provider = _make_api_key_provider(model_name)
    payload = {
        "model": "claude",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 2048,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }
    captured_session_headers = {}
    mock_client = mock_http_client(MockAsyncResponse(_anthropic_messages_response()))

    def mock_client_session(headers=None, **kwargs):
        captured_session_headers.update(headers or {})
        return mock_client

    with mock.patch("aiohttp.ClientSession", mock_client_session):
        response = await provider.passthrough(
            PassthroughAction.ANTHROPIC_MESSAGES,
            payload,
            headers={
                "authorization": "Bearer client-token",
                "x-api-key": "client-key",
                "user-agent": "claude-cli/2.0.37 (external, cli)",
                "x-request-id": "req-1",
                "host": "gateway.example.com",
            },
        )

    assert response == _anthropic_messages_response()
    mock_client.post.assert_called_once_with(
        f"https://bedrock-runtime.eu-west-1.amazonaws.com/model/{model_path}/invoke",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 2048,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "anthropic_version": "bedrock-2023-05-31",
        },
        timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS.get()),
        allow_redirects=False,
    )
    # FallbackProvider passes the same payload to the next provider if this one fails.
    assert payload["model"] == "claude"
    # A credential agent's own key is dropped too; Bedrock only accepts the endpoint's key.
    assert captured_session_headers["Authorization"] == "Bearer bedrock-api-key"
    assert "authorization" not in captured_session_headers
    assert "x-api-key" not in captured_session_headers
    assert "host" not in captured_session_headers
    assert captured_session_headers["x-request-id"] == "req-1"


@pytest.mark.asyncio
async def test_bedrock_anthropic_passthrough_stream_converts_event_stream_to_sse():
    provider = _make_api_key_provider()
    events = _anthropic_stream_events()
    data = b"".join(map(_bedrock_chunk, events))
    # Frames arrive split at arbitrary byte offsets, not one frame per read.
    chunks = [data[i : i + 37] for i in range(0, len(data), 37)]
    mock_client = mock_http_client(MockAsyncStreamingResponse(chunks))
    payload = {
        "model": "claude",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
        "stream": True,
    }

    with (
        mock.patch("aiohttp.ClientSession", return_value=mock_client),
        mock.patch.object(provider, "_set_span_token_usage") as mock_set_span_token_usage,
    ):
        stream = await provider.passthrough(PassthroughAction.ANTHROPIC_MESSAGES, payload)
        output = b"".join([chunk async for chunk in stream])

    assert output == b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events
    )
    mock_client.post.assert_called_once()
    assert mock_client.post.call_args[0][0] == (
        f"https://bedrock-runtime.eu-west-1.amazonaws.com/model/{_CLAUDE_MODEL_ID}"
        "/invoke-with-response-stream"
    )
    assert mock_client.post.call_args[1]["json"] == {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
        "anthropic_version": "bedrock-2023-05-31",
    }
    # FallbackProvider passes the same payload to the next provider if this one fails.
    assert payload == {
        "model": "claude",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
        "stream": True,
    }
    mock_set_span_token_usage.assert_called_once_with({
        "input_tokens": 12,
        "output_tokens": 30,
        "total_tokens": 42,
    })


@pytest.mark.parametrize(
    ("exception_type", "status_code"),
    [
        ("throttlingException", 429),
        ("serviceUnavailableException", 503),
        ("modelTimeoutException", 408),
        ("internalServerException", 502),
    ],
)
@pytest.mark.asyncio
async def test_bedrock_anthropic_passthrough_stream_raises_bedrock_exception(
    exception_type, status_code
):
    provider = _make_api_key_provider()
    data = _bedrock_chunk(_anthropic_stream_events()[0]) + _event_stream_message(
        {
            ":exception-type": exception_type,
            ":content-type": "application/json",
            ":message-type": "exception",
        },
        b'{"message": "The request could not be completed."}',
    )
    mock_client = mock_http_client(MockAsyncStreamingResponse([data]))
    payload = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64, "stream": True}

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        stream = await provider.passthrough(PassthroughAction.ANTHROPIC_MESSAGES, payload)
        with pytest.raises(HTTPException, match=f"{exception_type} while streaming") as exc_info:
            [chunk async for chunk in stream]

    assert exc_info.value.status_code == status_code
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_anthropic_passthrough_stream_raises_bedrock_error_frame():
    # An error frame (":message-type: error") carries ":error-code"/":error-message" headers
    # with an empty payload, rather than an ":exception-type" and a JSON body, so the status
    # and detail come from those header fallbacks.
    provider = _make_api_key_provider()
    data = _bedrock_chunk(_anthropic_stream_events()[0]) + _event_stream_message(
        {
            ":error-code": "throttlingException",
            ":error-message": "Rate exceeded.",
            ":message-type": "error",
        },
        b"",
    )
    mock_client = mock_http_client(MockAsyncStreamingResponse([data]))
    payload = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64, "stream": True}

    with mock.patch("aiohttp.ClientSession", return_value=mock_client):
        stream = await provider.passthrough(PassthroughAction.ANTHROPIC_MESSAGES, payload)
        with pytest.raises(
            HTTPException, match="throttlingException while streaming: Rate exceeded."
        ) as exc_info:
            [chunk async for chunk in stream]

    assert exc_info.value.status_code == 429
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_bedrock_anthropic_passthrough_stream_requires_botocore():
    provider = _make_api_key_provider()
    mock_client = mock_http_client(MockAsyncStreamingResponse([]))
    payload = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64, "stream": True}

    with (
        mock.patch("aiohttp.ClientSession", return_value=mock_client),
        mock.patch.dict("sys.modules", {"botocore.eventstream": None}),
    ):
        stream = await provider.passthrough(PassthroughAction.ANTHROPIC_MESSAGES, payload)
        with pytest.raises(ImportError, match="requires boto3"):
            [chunk async for chunk in stream]

    # The import fails before the request is sent, so nothing reaches Bedrock.
    mock_client.post.assert_not_called()


def test_bedrock_anthropic_passthrough_token_usage():
    provider = _make_api_key_provider()
    usage = provider._extract_passthrough_token_usage(
        PassthroughAction.ANTHROPIC_MESSAGES, _anthropic_messages_response()
    )
    assert usage == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}


@pytest.mark.asyncio
async def test_bedrock_anthropic_passthrough_rejects_non_anthropic_model():
    provider = _make_api_key_provider("amazon.nova-pro-v1:0")
    with pytest.raises(AIGatewayException, match="only supports Anthropic models") as exc_info:
        await provider.passthrough(
            PassthroughAction.ANTHROPIC_MESSAGES, {"messages": [], "max_tokens": 64}
        )
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("aws_config", [c for c, _ in bedrock_aws_configs])
@pytest.mark.asyncio
async def test_bedrock_anthropic_passthrough_requires_api_key_auth(aws_config):
    config = {
        "name": "claude",
        "endpoint_type": "llm/v1/chat",
        "model": {"provider": "bedrock", "name": _CLAUDE_MODEL_ID, "config": {}},
    }
    provider = AmazonBedrockProvider(
        EndpointConfig(**_merge_model_and_aws_config(config, aws_config))
    )
    with pytest.raises(AIGatewayException, match="API key") as exc_info:
        await provider.passthrough(
            PassthroughAction.ANTHROPIC_MESSAGES, {"messages": [], "max_tokens": 64}
        )
    assert exc_info.value.status_code == 501


@pytest.mark.asyncio
async def test_bedrock_passthrough_rejects_unsupported_action():
    provider = _make_api_key_provider()
    with pytest.raises(AIGatewayException, match="Unsupported passthrough endpoint"):
        await provider.passthrough(PassthroughAction.OPENAI_CHAT, {"messages": []})
