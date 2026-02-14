from unittest import mock

import pytest
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import (
    AmazonBedrockConfig,
    AWSBaseConfig,
    AWSIdAndKey,
    AWSRole,
    EndpointConfig,
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
    with mock.patch("boto3.Session", return_value=mock.Mock()) as mock_session:
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


# Converse API Test Fixtures and Tests


_DEFAULT_BEDROCK_CHAT_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"


def bedrock_chat_config_dict(model_name=_DEFAULT_BEDROCK_CHAT_MODEL):
    return {
        "name": "chat",
        "endpoint_type": "llm/v1/chat",
        "model": {
            "provider": "bedrock",
            "name": model_name,
            "config": {"aws_config": {"aws_region": "us-east-1"}},
        },
    }


def make_bedrock_chat_config(model_name=_DEFAULT_BEDROCK_CHAT_MODEL):
    return EndpointConfig(**bedrock_chat_config_dict(model_name=model_name))


@pytest.fixture
def bedrock_chat_config():
    return make_bedrock_chat_config()


def make_converse_stream(text_chunks, stop_reason="end_turn", usage=None, metrics=None):
    events = [
        {"messageStart": {"role": "assistant"}},
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {"text": ""},
            }
        },
    ]
    events.extend(
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": text},
            }
        }
        for text in text_chunks
    )
    events.append({"contentBlockStop": {"contentBlockIndex": 0}})
    events.append({"messageStop": {"stopReason": stop_reason}})
    metadata = {}
    if usage is not None:
        metadata["usage"] = usage
    if metrics is not None:
        metadata["metrics"] = metrics
    if metadata:
        events.append({"metadata": metadata})
    return events


def converse_chat_response():
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello! How can I assist you today?"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 12,
            "totalTokens": 22,
        },
        "metrics": {"latencyMs": 551},
    }


def parsed_converse_chat_response():
    return {
        "id": None,
        "object": "chat.completion",
        "created": 1677858242,
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                    "refusal": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
    }


converse_api_test_fixtures = [
    {
        "name": "basic_chat",
        "config": bedrock_chat_config_dict(),
        "request": {
            "messages": [{"role": "user", "content": "Hello"}],
        },
        "expected_converse_request": {
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
        },
        "response": converse_chat_response(),
        "expected": parsed_converse_chat_response(),
    },
    {
        "name": "chat_with_params",
        "config": bedrock_chat_config_dict(),
        "request": {
            "messages": [{"role": "user", "content": "Tell me a joke"}],
            "temperature": 0.8,
            "max_tokens": 500,
            "top_p": 0.9,
            "stop": ["END"],
        },
        "expected_converse_request": {
            "messages": [{"role": "user", "content": [{"text": "Tell me a joke"}]}],
            "inferenceConfig": {
                "temperature": 0.8,
                "maxTokens": 500,
                "topP": 0.9,
                "stopSequences": ["END"],
            },
        },
        "response": converse_chat_response(),
        "expected": parsed_converse_chat_response(),
    },
    {
        "name": "chat_with_system_message",
        "config": bedrock_chat_config_dict(),
        "request": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
        },
        "expected_converse_request": {
            "system": [{"text": "You are a helpful assistant."}],
            "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
        },
        "response": converse_chat_response(),
        "expected": parsed_converse_chat_response(),
    },
    {
        "name": "multi_turn_conversation",
        "config": bedrock_chat_config_dict(),
        "request": {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "user", "content": "And what is 3+3?"},
            ],
        },
        "expected_converse_request": {
            "messages": [
                {"role": "user", "content": [{"text": "What is 2+2?"}]},
                {"role": "assistant", "content": [{"text": "2+2 equals 4."}]},
                {"role": "user", "content": [{"text": "And what is 3+3?"}]},
            ],
        },
        "response": converse_chat_response(),
        "expected": parsed_converse_chat_response(),
    },
]


@pytest.mark.parametrize(
    "fixture",
    converse_api_test_fixtures,
    ids=[f["name"] for f in converse_api_test_fixtures],
)
def test_chat_to_model_transformation(fixture):
    from mlflow.gateway.providers.bedrock import ConverseAdapter

    config = EndpointConfig(**fixture["config"])
    payload = fixture["request"]

    result = ConverseAdapter.chat_to_model(payload, config)

    assert result["messages"] == fixture["expected_converse_request"]["messages"]

    if "inferenceConfig" in fixture["expected_converse_request"]:
        assert result["inferenceConfig"] == fixture["expected_converse_request"]["inferenceConfig"]


def test_chat_to_model_uses_max_completion_tokens(bedrock_chat_config):
    from mlflow.gateway.providers.bedrock import ConverseAdapter

    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_completion_tokens": 123,
    }

    result = ConverseAdapter.chat_to_model(payload, bedrock_chat_config)

    assert "inferenceConfig" in result
    assert result["inferenceConfig"]["maxTokens"] == 123


@pytest.mark.parametrize(
    "fixture",
    converse_api_test_fixtures,
    ids=[f["name"] for f in converse_api_test_fixtures],
)
def test_model_to_chat_transformation(fixture):
    from mlflow.gateway.providers.bedrock import ConverseAdapter

    config = EndpointConfig(**fixture["config"])

    with mock.patch("time.time", return_value=1677858242):
        result = ConverseAdapter.model_to_chat(fixture["response"], config)

    result_dict = jsonable_encoder(result)

    assert result_dict["object"] == "chat.completion"
    assert result_dict["model"] == config.model.name
    assert len(result_dict["choices"]) == 1

    choice = result_dict["choices"][0]
    expected_choice = fixture["expected"]["choices"][0]
    assert choice["message"]["role"] == expected_choice["message"]["role"]
    assert choice["message"]["content"] == expected_choice["message"]["content"]

    assert choice["finish_reason"] == expected_choice["finish_reason"]

    assert result_dict["usage"] == fixture["expected"]["usage"]


def test_model_to_chat_multiple_text_blocks(bedrock_chat_config):
    from mlflow.gateway.providers.bedrock import ConverseAdapter

    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "First part"},
                    {"text": "Second part"},
                ],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
    }

    with mock.patch("time.time", return_value=1677858242):
        result = ConverseAdapter.model_to_chat(response, bedrock_chat_config)

    result_dict = jsonable_encoder(result)
    assert result_dict["choices"][0]["message"]["content"] == "First partSecond part"


@pytest.mark.asyncio
async def test_chat_stream_basic(bedrock_chat_config):
    from mlflow.gateway.providers.bedrock import AmazonBedrockProvider
    from mlflow.gateway.schemas import chat

    with mock.patch.object(AmazonBedrockProvider, "get_bedrock_client") as mock_client:
        mock_stream = make_converse_stream(
            ["Hello! ", "How can I help you?"],
            usage={"inputTokens": 10, "outputTokens": 8, "totalTokens": 18},
        )

        mock_client.return_value.converse_stream.return_value = {"stream": iter(mock_stream)}

        provider = AmazonBedrockProvider(bedrock_chat_config)

        payload = chat.RequestPayload(messages=[chat.RequestMessage(role="user", content="Hello")])

        chunks = [jsonable_encoder(chunk) async for chunk in provider.chat_stream(payload) if chunk]

        assert len(chunks) > 0, "Should receive streaming chunks"

        text_chunks = [c for c in chunks if c["choices"][0].get("delta", {}).get("content")]

        assert len(text_chunks) == 2, "Should receive two text delta chunks"
        assert text_chunks[0]["choices"][0]["delta"]["content"] == "Hello! "
        assert text_chunks[1]["choices"][0]["delta"]["content"] == "How can I help you?"

        final_chunk = chunks[-1]
        assert final_chunk["choices"][0]["finish_reason"] == "stop"


def test_system_message_with_content_parts_raises_error(bedrock_chat_config):
    from mlflow.gateway.exceptions import AIGatewayException
    from mlflow.gateway.providers.bedrock import ConverseAdapter

    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                ],
            },
            {"role": "user", "content": "Hello"},
        ],
    }

    with pytest.raises(
        AIGatewayException, match="System message content must be a string"
    ) as exc_info:
        ConverseAdapter.chat_to_model(payload, bedrock_chat_config)

    assert exc_info.value.status_code == 422
    assert "Bedrock Converse API does not support content parts" in str(exc_info.value.detail)


@pytest.mark.parametrize(
    "part",
    [
        {
            "type": "input_audio",
            "input_audio": {"data": "base64data", "format": "wav"},
        }
    ],
    ids=["input_audio"],
)
def test_unsupported_content_part_types_raise_error(bedrock_chat_config, part):
    from mlflow.gateway.exceptions import AIGatewayException
    from mlflow.gateway.providers.bedrock import ConverseAdapter

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    part,
                ],
            }
        ],
    }

    part_type = part["type"]
    with pytest.raises(
        AIGatewayException, match=f"Unsupported content part type: '{part_type}'"
    ) as exc_info:
        ConverseAdapter.chat_to_model(payload, bedrock_chat_config)

    assert exc_info.value.status_code == 422
    assert "Converse API currently only supports 'text' content parts" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_streaming_include_usage_not_supported(bedrock_chat_config):
    from mlflow.gateway.exceptions import AIGatewayException
    from mlflow.gateway.providers.bedrock import AmazonBedrockProvider
    from mlflow.gateway.schemas import chat

    with mock.patch.object(AmazonBedrockProvider, "get_bedrock_client"):
        provider = AmazonBedrockProvider(bedrock_chat_config)

        payload = chat.RequestPayload(
            messages=[chat.RequestMessage(role="user", content="Hello")],
            stream_options={"include_usage": True},
        )

        with pytest.raises(
            AIGatewayException, match="stream_options.include_usage is not supported"
        ) as exc_info:
            async for _ in provider.chat_stream(payload):
                pass

        assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_streaming_omits_usage_when_stream_options_not_enabled(bedrock_chat_config):
    from mlflow.gateway.providers.bedrock import AmazonBedrockProvider
    from mlflow.gateway.schemas import chat

    with mock.patch.object(AmazonBedrockProvider, "get_bedrock_client") as mock_client:
        mock_stream = make_converse_stream(
            ["Hello! How can I help you today?"],
            usage={"inputTokens": 10, "outputTokens": 12, "totalTokens": 22},
            metrics={"latencyMs": 551},
        )

        mock_client.return_value.converse_stream.return_value = {"stream": iter(mock_stream)}

        provider = AmazonBedrockProvider(bedrock_chat_config)

        payload = chat.RequestPayload(
            messages=[chat.RequestMessage(role="user", content="Hello")],
        )

        chunks = [jsonable_encoder(chunk) async for chunk in provider.chat_stream(payload) if chunk]

        chunks_with_usage = [c for c in chunks if c.get("usage")]
        assert len(chunks_with_usage) == 0, "Usage should not be emitted without include_usage"


def test_n_parameter_accepts_one():
    from mlflow.gateway.providers.bedrock import ConverseAdapter

    config = make_bedrock_chat_config(model_name="anthropic.claude-3-sonnet-20240229-v1:0")
    payload_n1 = {
        "messages": [{"role": "user", "content": "Hello"}],
        "n": 1,
    }
    result = ConverseAdapter.chat_to_model(payload_n1, config)
    assert "messages" in result


@pytest.mark.parametrize("n", [2, 3])
def test_n_parameter_rejects_non_one(n):
    from mlflow.gateway.exceptions import AIGatewayException
    from mlflow.gateway.providers.bedrock import ConverseAdapter

    config = make_bedrock_chat_config(model_name="anthropic.claude-3-sonnet-20240229-v1:0")
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "n": n,
    }
    with pytest.raises(AIGatewayException, match="'n' must be '1'") as exc_info:
        ConverseAdapter.chat_to_model(payload, config)

    assert exc_info.value.status_code == 422
    assert "Bedrock Converse API" in str(exc_info.value.detail)


def test_n_parameter_defaults_to_one():
    from mlflow.gateway.providers.bedrock import ConverseAdapter

    config = make_bedrock_chat_config(model_name="anthropic.claude-3-sonnet-20240229-v1:0")
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
    }
    result = ConverseAdapter.chat_to_model(payload, config)
    assert "messages" in result


def test_tool_calling_not_yet_supported(bedrock_chat_config):
    from mlflow.gateway.exceptions import AIGatewayException
    from mlflow.gateway.providers.bedrock import ConverseAdapter

    payload_with_tools = {
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }
    with pytest.raises(AIGatewayException, match="Tool calling is not yet supported") as exc_info:
        ConverseAdapter.chat_to_model(payload_with_tools, bedrock_chat_config)
    assert exc_info.value.status_code == 422

    payload_with_tool_choice = {
        "messages": [{"role": "user", "content": "Hello"}],
        "tool_choice": "auto",
    }
    with pytest.raises(AIGatewayException, match="tool_choice is not yet supported") as exc_info:
        ConverseAdapter.chat_to_model(payload_with_tool_choice, bedrock_chat_config)
    assert exc_info.value.status_code == 422


def test_tool_messages_not_yet_supported(bedrock_chat_config):
    from mlflow.gateway.exceptions import AIGatewayException
    from mlflow.gateway.providers.bedrock import ConverseAdapter

    payload_with_tool_role = {
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": '{"temperature": 72}',
            },
        ],
    }
    with pytest.raises(
        AIGatewayException, match=r"Tool result messages \(role='tool'\).*not yet supported"
    ) as exc_info:
        ConverseAdapter.chat_to_model(payload_with_tool_role, bedrock_chat_config)
    assert exc_info.value.status_code == 422

    payload_with_tool_calls = {
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Tokyo"}',
                        },
                    }
                ],
            },
        ],
    }
    with pytest.raises(AIGatewayException, match="tool_calls are not yet supported") as exc_info:
        ConverseAdapter.chat_to_model(payload_with_tool_calls, bedrock_chat_config)
    assert exc_info.value.status_code == 422
