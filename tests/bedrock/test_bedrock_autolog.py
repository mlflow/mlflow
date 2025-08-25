import base64
import io
import json
from pathlib import Path
from unittest import mock

import boto3
import pytest
from botocore.exceptions import NoCredentialsError
from botocore.response import StreamingBody
from packaging.version import Version

import mlflow
from mlflow.tracing.constant import SpanAttributeKey

from tests.tracing.helper import get_traces

_IS_CONVERSE_API_AVAILABLE = Version(boto3.__version__) >= Version("1.35")


def _assert_token_usage_matches(span, expected_usage):
    """Assert span token usage matches expected values with descriptive error messages."""
    if expected_usage is None:
        assert SpanAttributeKey.CHAT_USAGE not in span.attributes, (
            f"Expected no usage but found: {span.attributes.get(SpanAttributeKey.CHAT_USAGE)}"
        )
    else:
        usage = span.attributes.get(SpanAttributeKey.CHAT_USAGE)
        assert usage == expected_usage, f"Expected usage {expected_usage} but got {usage}"


# https://docs.aws.amazon.com/code-library/latest/ug/python_3_bedrock-runtime_code_examples.html#anthropic_claude
_ANTHROPIC_REQUEST = {
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 300,
    "anthropic_version": "bedrock-2023-05-31",
    "temperature": 0.1,
    "top_p": 0.9,
}

_ANTHROPIC_RESPONSE = {
    "id": "id-123",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20241022",
    "content": [{"type": "text", "text": "Hello! How can I help you today?"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 8,
        "output_tokens": 12,
    },
}

# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-jamba.html
_AI21_JAMBA_REQUEST = {
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 300,
    "temperature": 0.1,
}

_AI21_JAMBA_RESPONSE = {
    "id": "id-123",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
                "tool_calls": None,
            },
            "finish_reason": "stop",
        },
    ],
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 12,
        "total_tokens": 20,
    },
    "model": "jamba-instruct",
    "meta": {"requestDurationMillis": 288},
}

# https://docs.aws.amazon.com/nova/latest/userguide/using-invoke-api.html
_AMAZON_NOVA_REQUEST = {
    "messages": [{"role": "user", "content": [{"text": "Hi!"}]}],
    "system": [{"text": "This is a system prompt"}],
    "schemaVersion": "messages-v1",
    "inferenceConfig": {
        "max_new_tokens": 512,
        "temperature": 0.5,
    },
}

_AMAZON_NOVA_RESPONSE = {
    "message": {
        "role": "assistant",
        "content": [{"text": "Sure! How can I help you today?"}],
    },
    "stopReason": "end_turn",
    "usage": {
        "inputTokens": 8,
        "outputTokens": 12,
        "totalTokens": 20,
    },
}

# https://docs.aws.amazon.com/code-library/latest/ug/python_3_bedrock-runtime_code_examples.html#cohere_command
_COHERE_REQUEST = {
    "message": "Hi!",
    "max_tokens": 512,
    "temperature": 0.5,
}

_COHERE_RESPONSE = {
    "response_id": "id-123",
    "text": "Sure! How can I help you today?",
    "generation_id": "generation-id-123",
    "chat_history": [
        {"role": "USER", "message": "Hi"},
        {"role": "CHATBOT", "message": "Hello! How can I help you today?"},
    ],
    "finish_reason": "COMPLETE",
}

# https://docs.aws.amazon.com/code-library/latest/ug/python_3_bedrock-runtime_code_examples.html#meta_llama
# (same as Mistral)
_META_LLAMA_REQUEST = {
    "prompt": """
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Hi!
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
""",
    "max_gen_len": 300,
    "temperature": 0.1,
}

_META_LLAMA_RESPONSE = {
    "generation": "Hello! How can I help you today?",
    "prompt_token_count": 2,
    "generation_token_count": 12,
    "stop_reason": "stop",
}


# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
_AMAZON_EMBEDDING_REQUEST = {
    "inputText": "Hi",
    "dimensions": 8,
}

_AMAZON_EMBEDDING_RESPONSE = {
    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "inputTextTokenCount": 15,
}


def _create_dummy_invoke_model_response(llm_response):
    llm_response_encoded = json.dumps(llm_response).encode("utf-8")
    return {
        "body": StreamingBody(io.BytesIO(llm_response_encoded), len(llm_response_encoded)),
        "ResponseMetadata": {
            "RequestId": "request-id-123",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {"content-type": "application/json"},
            "RetryAttempts": 0,
        },
        "contentType": "application/json",
    }


@pytest.mark.parametrize(
    ("model_id", "llm_request", "llm_response", "expected_usage"),
    [
        (
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            _ANTHROPIC_REQUEST,
            _ANTHROPIC_RESPONSE,
            {"input_tokens": 8, "output_tokens": 12, "total_tokens": 20},
        ),
        (
            "ai21.jamba-instruct-v1:0",
            _AI21_JAMBA_REQUEST,
            _AI21_JAMBA_RESPONSE,
            {"input_tokens": 8, "output_tokens": 12, "total_tokens": 20},
        ),
        (
            "us.amazon.nova-lite-v1:0",
            _AMAZON_NOVA_REQUEST,
            _AMAZON_NOVA_RESPONSE,
            {"input_tokens": 8, "output_tokens": 12, "total_tokens": 20},
        ),
        (
            "cohere.command-r-plus-v1:0",
            _COHERE_REQUEST,
            _COHERE_RESPONSE,
            None,  # No usage in response
        ),
        (
            "meta.llama3-8b-instruct-v1:0",
            _META_LLAMA_REQUEST,
            _META_LLAMA_RESPONSE,
            {"input_tokens": 2, "output_tokens": 12, "total_tokens": 14},
        ),
    ],
)
def test_bedrock_autolog_invoke_model_llm(model_id, llm_request, llm_response, expected_usage):
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    request_body = json.dumps(llm_request)

    # Ref: https://docs.getmoto.org/en/latest/docs/services/patching_other_services.html
    with mock.patch(
        "botocore.client.BaseClient._make_api_call",
        return_value=_create_dummy_invoke_model_response(llm_response),
    ):
        response = client.invoke_model(body=request_body, modelId=model_id)

    response_body = json.loads(response["body"].read())
    assert response_body == llm_response

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"

    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "BedrockRuntime.invoke_model"
    assert span.span_type == "LLM"
    assert span.inputs == {"body": request_body, "modelId": model_id}
    assert span.outputs == {
        "body": llm_response,
        "ResponseMetadata": response["ResponseMetadata"],
        "contentType": "application/json",
    }

    # Check token usage validation using parameterized expected values
    _assert_token_usage_matches(span, expected_usage)


def test_bedrock_autolog_invoke_model_embeddings():
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    request_body = json.dumps(_AMAZON_EMBEDDING_REQUEST)
    model_id = "amazon.titan-embed-text-v1"

    # Ref: https://docs.getmoto.org/en/latest/docs/services/patching_other_services.html
    with mock.patch(
        "botocore.client.BaseClient._make_api_call",
        return_value=_create_dummy_invoke_model_response(_AMAZON_EMBEDDING_RESPONSE),
    ):
        response = client.invoke_model(body=request_body, modelId=model_id)

    response_body = json.loads(response["body"].read())
    assert response_body == _AMAZON_EMBEDDING_RESPONSE

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"

    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "BedrockRuntime.invoke_model"
    assert span.span_type == "EMBEDDING"
    assert span.inputs == {"body": request_body, "modelId": model_id}
    assert span.outputs == {
        "body": _AMAZON_EMBEDDING_RESPONSE,
        "ResponseMetadata": response["ResponseMetadata"],
        "contentType": "application/json",
    }


def test_bedrock_autolog_invoke_model_capture_exception():
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    request_body = json.dumps(
        {
            # Invalid user role to trigger an exception
            "messages": [{"role": "invalid-user", "content": "Hi"}],
            "max_tokens": 300,
            "anthropic_version": "bedrock-2023-05-31",
            "temperature": 0.1,
            "top_p": 0.9,
        }
    )

    with pytest.raises(NoCredentialsError, match="Unable to locate credentials"):
        client.invoke_model(
            body=request_body,
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        )

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"

    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "BedrockRuntime.invoke_model"
    assert span.status.status_code == "ERROR"
    assert span.inputs == {
        "body": request_body,
        "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    }
    assert span.outputs is None
    assert len(span.events) == 1
    assert span.events[0].name == "exception"
    assert span.events[0].attributes["exception.message"].startswith("Unable to locate credentials")


def test_bedrock_autolog_invoke_model_stream():
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    request_body = json.dumps(_ANTHROPIC_REQUEST)

    dummy_chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "123",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 8, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "content_block": {"type": "text", "text": "Hello"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "! How can I "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "help you today?"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 12},
        },
        {"type": "message_stop", "amazon-bedrock-invocationMetrics": {"invocationLatency": 909}},
    ]

    # Mimic event stream
    def dummy_stream():
        for chunk in dummy_chunks:
            yield {"chunk": {"bytes": json.dumps(chunk).encode("utf-8")}}

    # Ref: https://docs.getmoto.org/en/latest/docs/services/patching_other_services.html
    with mock.patch(
        "botocore.client.BaseClient._make_api_call",
        return_value={"body": dummy_stream()},
    ):
        response = client.invoke_model_with_response_stream(
            body=request_body, modelId=_ANTHROPIC_MODEL_ID
        )

    # Trace should not be created until the stream is consumed
    assert get_traces() == []

    events = list(response["body"])
    assert len(events) == len(dummy_chunks)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"

    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "BedrockRuntime.invoke_model_with_response_stream"
    assert span.span_type == "LLM"
    assert span.inputs == {"body": request_body, "modelId": _ANTHROPIC_MODEL_ID}
    assert span.outputs == {"body": "EventStream"}
    # Raw chunks must be recorded as span events
    assert len(span.events) == len(dummy_chunks)
    for i in range(len(dummy_chunks)):
        assert span.events[i].name == dummy_chunks[i]["type"]
        assert json.loads(span.events[i].attributes["json"]) == dummy_chunks[i]

    # Check that token usage is captured from streaming chunks
    assert SpanAttributeKey.CHAT_USAGE in span.attributes
    usage = span.attributes[SpanAttributeKey.CHAT_USAGE]
    assert usage["input_tokens"] == 8
    assert usage["output_tokens"] == 12  # Updated from message_delta event
    assert usage["total_tokens"] == 20  # Calculated as input + output


@pytest.mark.parametrize("config", [{"disable": True}, {"log_traces": False}])
def test_bedrock_autolog_trace_disabled(config):
    mlflow.bedrock.autolog(**config)

    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    request_body = json.dumps(_ANTHROPIC_REQUEST)

    with mock.patch(
        "botocore.client.BaseClient._make_api_call",
        return_value=_create_dummy_invoke_model_response(_ANTHROPIC_RESPONSE),
    ):
        response = client.invoke_model(
            body=request_body, modelId="anthropic.claude-3-5-sonnet-20241022-v2:0"
        )

    response_body = json.loads(response["body"].read())
    assert response_body == _ANTHROPIC_RESPONSE

    traces = get_traces()
    assert len(traces) == 0


# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
_ANTHROPIC_MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

_CONVERSE_REQUEST = {
    "modelId": _ANTHROPIC_MODEL_ID,
    "messages": [{"role": "user", "content": [{"text": "Hi"}]}],
    "inferenceConfig": {
        "maxTokens": 300,
        "temperature": 0.1,
        "topP": 0.9,
    },
}

_CONVERSE_RESPONSE = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "Hello! How can I help you today?"}],
        },
    },
    "stopReason": "end_turn",
    "usage": {"inputTokens": 8, "outputTokens": 12},
    "metrics": {"latencyMs": 551},
}

_CONVERSE_EXPECTED_CHAT_ATTRIBUTE = [
    {
        "role": "user",
        "content": [{"text": "Hi", "type": "text"}],
    },
    {
        "role": "assistant",
        "content": [{"text": "Hello! How can I help you today?", "type": "text"}],
    },
]


# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
_CONVERSE_TOOL_CALLING_REQUEST = {
    "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "messages": [
        {"role": "user", "content": [{"text": "What's the weather like in San Francisco?"}]},
        {
            "role": "assistant",
            "content": [
                {"text": "I'll use the get_unit function to determine the temperature unit."},
                {
                    "toolUse": {
                        "toolUseId": "tool_1",
                        "name": "get_unit",
                        "input": {"location": "San Francisco, CA"},
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tool_1",
                        "content": [{"json": {"unit": "fahrenheit"}}],
                    }
                }
            ],
        },
    ],
    "inferenceConfig": {
        "maxTokens": 300,
        "temperature": 0.1,
        "topP": 0.9,
    },
    "toolConfig": {
        "tools": [
            {
                "toolSpec": {
                    "name": "get_unit",
                    "description": "Get the temperature unit in a given location",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g., San Francisco, CA",
                                },
                            },
                            "required": ["location"],
                        },
                    },
                },
            },
            {
                "toolSpec": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g., San Francisco, CA",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": '"celsius" or "fahrenheit"',
                                },
                            },
                            "required": ["location"],
                        },
                    },
                },
            },
        ]
    },
}

_CONVERSE_TOOL_CALLING_RESPONSE = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [
                {"text": "Now I'll check the current weather in San Francisco."},
                {
                    "toolUse": {
                        "toolUseId": "tool_2",
                        "name": "get_weather",
                        "input": '{"location": "San Francisco, CA", "unit": "fahrenheit"}',
                    }
                },
            ],
        },
    },
    "stopReason": "end_turn",
    "usage": {"inputTokens": 8, "outputTokens": 12},
    "metrics": {"latencyMs": 551},
}

_CONVERSE_TOOL_CALLING_EXPECTED_CHAT_ATTRIBUTE = [
    {
        "role": "user",
        "content": [{"text": "What's the weather like in San Francisco?", "type": "text"}],
    },
    {
        "role": "assistant",
        "content": [
            {
                "text": "I'll use the get_unit function to determine the temperature unit.",
                "type": "text",
            },
        ],
        "tool_calls": [
            {
                "id": "tool_1",
                "function": {
                    "name": "get_unit",
                    "arguments": '{"location": "San Francisco, CA"}',
                },
                "type": "function",
            },
        ],
    },
    {
        "role": "tool",
        "content": [{"text": '{"unit": "fahrenheit"}', "type": "text"}],
        "tool_call_id": "tool_1",
    },
    {
        "role": "assistant",
        "content": [
            {"text": "Now I'll check the current weather in San Francisco.", "type": "text"},
        ],
        "tool_calls": [
            {
                "id": "tool_2",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "San Francisco, CA", "unit": "fahrenheit"}',
                },
                "type": "function",
            },
        ],
    },
]

_CONVERSE_TOOL_CALLING_EXPECTED_TOOL_ATTRIBUTE = [
    {
        "type": "function",
        "function": {
            "name": "get_unit",
            "description": "Get the temperature unit in a given location",
            "parameters": {
                "properties": {
                    "location": {
                        "description": "The city and state, e.g., San Francisco, CA",
                        "type": "string",
                    },
                },
                "required": ["location"],
                "type": "object",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "properties": {
                    "location": {
                        "description": "The city and state, e.g., San Francisco, CA",
                        "type": "string",
                    },
                    "unit": {
                        "description": '"celsius" or "fahrenheit"',
                        "enum": ["celsius", "fahrenheit"],
                        "type": "string",
                    },
                },
                "required": ["location"],
                "type": "object",
            },
        },
    },
]


def _get_test_image(is_base64: bool):
    image_dir = Path(__file__).parent.parent / "resources" / "images"
    with open(image_dir / "test.png", "rb") as f:
        image_bytes = f.read()
        return base64.b64encode(image_bytes).decode("utf-8") if is_base64 else image_bytes


def _get_converse_multi_modal_request(is_base64: bool):
    return {
        "modelId": _ANTHROPIC_MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"text": "What text is in this image?"},
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": _get_test_image(is_base64)},
                        },
                    },
                ],
            }
        ],
    }


_CONVERSE_MULTI_MODAL_RESPONSE = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "MLflow"}],
        },
    },
    "stopReason": "end_turn",
    "usage": {"inputTokens": 8, "outputTokens": 2},
}

_CONVERSE_MULTI_MODAL_EXPECTED_CHAT_ATTRIBUTE = [
    {
        "role": "user",
        "content": [
            {"text": "What text is in this image?", "type": "text"},
            {
                "image_url": {
                    "url": f"data:image/png;base64,{_get_test_image(True)}",
                    "detail": "auto",
                },
                "type": "image_url",
            },
        ],
    },
    {
        "role": "assistant",
        "content": [{"text": "MLflow", "type": "text"}],
    },
]


@pytest.mark.skipif(not _IS_CONVERSE_API_AVAILABLE, reason="Converse API is not available")
@pytest.mark.parametrize(
    ("_request", "response", "expected_chat_attr", "expected_tool_attr", "expected_usage"),
    [
        # 1. Normal conversation
        (
            _CONVERSE_REQUEST,
            _CONVERSE_RESPONSE,
            _CONVERSE_EXPECTED_CHAT_ATTRIBUTE,
            None,
            {"input_tokens": 8, "output_tokens": 12, "total_tokens": 20},
        ),
        # 2. Conversation with tool calling
        (
            _CONVERSE_TOOL_CALLING_REQUEST,
            _CONVERSE_TOOL_CALLING_RESPONSE,
            _CONVERSE_TOOL_CALLING_EXPECTED_CHAT_ATTRIBUTE,
            _CONVERSE_TOOL_CALLING_EXPECTED_TOOL_ATTRIBUTE,
            {"input_tokens": 8, "output_tokens": 12, "total_tokens": 20},
        ),
        # 3. Conversation with image input (raw bytes)
        (
            _get_converse_multi_modal_request(is_base64=False),
            _CONVERSE_MULTI_MODAL_RESPONSE,
            _CONVERSE_MULTI_MODAL_EXPECTED_CHAT_ATTRIBUTE,
            None,
            {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
        ),
        # 4. Conversation with image input (base64)
        (
            _get_converse_multi_modal_request(is_base64=True),
            _CONVERSE_MULTI_MODAL_RESPONSE,
            _CONVERSE_MULTI_MODAL_EXPECTED_CHAT_ATTRIBUTE,
            None,
            {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
        ),
    ],
)
def test_bedrock_autolog_converse(
    _request, response, expected_chat_attr, expected_tool_attr, expected_usage
):
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    with mock.patch("botocore.client.BaseClient._make_api_call", return_value=response):
        response = client.converse(**_request)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"

    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "BedrockRuntime.converse"
    assert span.inputs is not None  # request with bytes is stringified and not recoverable
    assert span.outputs == response
    assert span.get_attribute(SpanAttributeKey.CHAT_TOOLS) == expected_tool_attr
    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "bedrock"

    # Validate token usage against parameterized expected values
    _assert_token_usage_matches(span, expected_usage)


@pytest.mark.skipif(not _IS_CONVERSE_API_AVAILABLE, reason="Converse API is not available")
def test_bedrock_autolog_converse_error():
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    with pytest.raises(NoCredentialsError, match="Unable to locate credentials"):
        client.converse(**_CONVERSE_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "ERROR"

    span = traces[0].data.spans[0]
    assert span.name == "BedrockRuntime.converse"
    assert span.status.status_code == "ERROR"
    assert span.inputs == _CONVERSE_REQUEST
    assert span.outputs is None
    assert len(span.events) == 1


@pytest.mark.skipif(not _IS_CONVERSE_API_AVAILABLE, reason="Converse API is not available")
def test_bedrock_autolog_converse_skip_unsupported_content():
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    with mock.patch("botocore.client.BaseClient._make_api_call", return_value=_CONVERSE_RESPONSE):
        client.converse(
            modelId=_ANTHROPIC_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"video": b"\xe3\x81\xad\xe3\x81\x93"},
                        {"text": "What you can see in this video?"},
                    ],
                }
            ],
        )

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"

    span = traces[0].data.spans[0]
    assert span.name == "BedrockRuntime.converse"


@pytest.mark.skipif(not _IS_CONVERSE_API_AVAILABLE, reason="Converse API is not available")
@pytest.mark.parametrize(
    ("_request", "expected_response", "expected_chat_attr", "expected_tool_attr", "expected_usage"),
    [
        # 1. Normal conversation
        (
            _CONVERSE_REQUEST,
            _CONVERSE_RESPONSE,
            _CONVERSE_EXPECTED_CHAT_ATTRIBUTE,
            None,
            {"input_tokens": 8, "output_tokens": 12, "total_tokens": 20},
        ),
        # 2. Conversation with tool calling
        (
            _CONVERSE_TOOL_CALLING_REQUEST,
            _CONVERSE_TOOL_CALLING_RESPONSE,
            _CONVERSE_TOOL_CALLING_EXPECTED_CHAT_ATTRIBUTE,
            _CONVERSE_TOOL_CALLING_EXPECTED_TOOL_ATTRIBUTE,
            {"input_tokens": 8, "output_tokens": 12, "total_tokens": 20},
        ),
    ],
)
def test_bedrock_autolog_converse_stream(
    _request, expected_response, expected_chat_attr, expected_tool_attr, expected_usage
):
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    with mock.patch(
        "botocore.client.BaseClient._make_api_call",
        return_value={"stream": _event_stream(expected_response)},
    ):
        response = client.converse_stream(**_request)

    assert get_traces() == []

    chunks = list(response["stream"])
    assert chunks == list(_event_stream(expected_response))

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"

    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "BedrockRuntime.converse_stream"
    assert span.inputs == _request
    assert span.outputs == expected_response
    assert span.get_attribute(SpanAttributeKey.CHAT_TOOLS) == expected_tool_attr
    assert len(span.events) > 0
    assert span.events[0].name == "messageStart"
    assert json.loads(span.events[0].attributes["json"]) == {"role": "assistant"}

    # Validate token usage against parameterized expected values
    _assert_token_usage_matches(span, expected_usage)


def _event_stream(raw_response, chunk_size=10):
    """Split the raw response into chunks to simulate the event stream."""
    content = raw_response["output"]["message"]["content"]

    yield {"messageStart": {"role": "assistant"}}

    text_content = content[0]["text"]
    for i in range(0, len(text_content), chunk_size):
        yield {"contentBlockDelta": {"delta": {"text": text_content[i : i + chunk_size]}}}

    yield {"contentBlockStop": {}}

    yield from _generate_tool_use_chunks_if_present(content)

    yield {"messageStop": {"stopReason": "end_turn"}}

    yield {"metadata": {"usage": raw_response["usage"], "metrics": {"latencyMs": 551}}}


def _generate_tool_use_chunks_if_present(content, chunk_size=10):
    if len(content) > 1 and (tool_content := content[1].get("toolUse")):
        yield {
            "contentBlockStart": {
                "start": {
                    "toolUse": {
                        "toolUseId": tool_content["toolUseId"],
                        "name": tool_content["name"],
                    }
                }
            }
        }

        for i in range(0, len(tool_content["input"]), chunk_size):
            yield {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": tool_content["input"][i : i + chunk_size]}}
                }
            }
        yield {"contentBlockStop": {}}


# Shared - Token usage edge case data for different APIs
def _create_converse_response_with_usage(usage_data):
    """Create a converse response with the given usage data."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello! How can I help you today?"}],
            },
        },
        "stopReason": "end_turn",
        "usage": usage_data,
    }


def _create_invoke_model_response_with_usage(usage_data):
    """Create an invoke_model response with the given usage data."""
    return {
        "id": "id-123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "usage": usage_data,
    }


# Shared - Token usage edge cases - these test the same parsing logic across different APIs
TOKEN_USAGE_EDGE_CASE_DATA = [
    # 1. Missing usage field entirely
    {
        "name": "missing_usage_field",
        "usage_data": None,  # No usage field at all
        "expected_usage": None,
    },
    # 2. Empty usage dict
    {
        "name": "empty_usage_dict",
        "usage_data": {},  # Empty dict
        "expected_usage": None,
    },
    # 3. Usage with only input tokens
    {
        "name": "only_input_tokens",
        "usage_data": {"inputTokens": 10},  # Only input
        "expected_usage": None,  # Should return None since output is missing
    },
    # 4. Usage with only output tokens
    {
        "name": "only_output_tokens",
        "usage_data": {"outputTokens": 5},  # Only output
        "expected_usage": None,  # Should return None since input is missing
    },
    # 5. Zero token values
    {
        "name": "zero_token_values",
        "usage_data": {"inputTokens": 0, "outputTokens": 0},
        "expected_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    },
    # 6. Very large token values
    {
        "name": "large_token_values",
        "usage_data": {"inputTokens": 999999, "outputTokens": 888888},
        "expected_usage": {
            "input_tokens": 999999,
            "output_tokens": 888888,
            "total_tokens": 1888887,
        },
    },
    # 7. Usage with None values
    {
        "name": "none_token_values",
        "usage_data": {"inputTokens": None, "outputTokens": None},
        "expected_usage": None,
    },
    # 8. Usage with string values (should be ignored)
    {
        "name": "string_token_values",
        "usage_data": {"inputTokens": "10", "outputTokens": "5"},
        "expected_usage": None,  # Should return None since values are strings
    },
]


# Invoke model token usage edge cases using shared data
TOKEN_USAGE_EDGE_CASES = [
    {
        "name": f"invoke_model_{case['name']}",
        "response": _create_invoke_model_response_with_usage(case["usage_data"]),
        "expected_usage": case["expected_usage"],
    }
    for case in TOKEN_USAGE_EDGE_CASE_DATA
] + [
    # Additional edge case specific to invoke_model (duplicate keys)
    {
        "name": "invoke_model_duplicate_keys_in_usage",
        "response": {
            "id": "id-123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {
                "input_tokens": 8,
                "output_tokens": 1,
                "input_tokens": 10,  # noqa: F601 - Testing duplicate key behavior
                "output_tokens": 5,  # noqa: F601 - Testing duplicate key behavior
            },
        },
        "expected_usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    },
]


# Converse token usage edge cases using shared data
CONVERSE_TOKEN_USAGE_EDGE_CASES = [
    {
        "name": f"converse_{case['name']}",
        "response": _create_converse_response_with_usage(case["usage_data"]),
        "expected_usage": case["expected_usage"],
    }
    for case in TOKEN_USAGE_EDGE_CASE_DATA
]


@pytest.mark.parametrize(
    "test_case", TOKEN_USAGE_EDGE_CASES, ids=[c["name"] for c in TOKEN_USAGE_EDGE_CASES]
)
def test_bedrock_autolog_token_usage_edge_cases(test_case):
    mlflow.bedrock.autolog()
    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    with mock.patch(
        "botocore.client.BaseClient._make_api_call",
        return_value=_create_dummy_invoke_model_response(test_case["response"]),
    ):
        client.invoke_model(
            body=json.dumps(_ANTHROPIC_REQUEST),
            modelId=_ANTHROPIC_MODEL_ID,
        )

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]

    # Check token usage
    _assert_token_usage_matches(span, test_case["expected_usage"])


STREAM_TOKEN_USAGE_EDGE_CASES = [
    # 1. No usage in any chunk
    {
        "name": "no_usage_in_stream",
        "chunks": [
            {
                "type": "message_start",
                "message": {
                    "id": "123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    # No usage field
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "content_block": {"type": "text", "text": "Hello!"},
            },
            {"type": "message_stop"},
        ],
        "expected_usage": None,
    },
    # 2. Usage only in first chunk
    {
        "name": "usage_only_in_first_chunk",
        "chunks": [
            {
                "type": "message_start",
                "message": {
                    "id": "123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 8, "output_tokens": 1},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "content_block": {"type": "text", "text": "Hello!"},
            },
            {"type": "message_stop"},
        ],
        "expected_usage": {"input_tokens": 8, "output_tokens": 1, "total_tokens": 9},
    },
    # 3. Usage updated in later chunk
    {
        "name": "usage_updated_in_later_chunk",
        "chunks": [
            {
                "type": "message_start",
                "message": {
                    "id": "123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 8, "output_tokens": 1},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "content_block": {"type": "text", "text": "Hello!"},
            },
            {
                "type": "message_delta",
                "delta": {},
                "usage": {"output_tokens": 12},  # Updated output tokens
            },
            {"type": "message_stop"},
        ],
        "expected_usage": {"input_tokens": 8, "output_tokens": 12, "total_tokens": 20},
    },
    # 4. Malformed usage in chunk
    {
        "name": "malformed_usage_in_chunk",
        "chunks": [
            {
                "type": "message_start",
                "message": {
                    "id": "123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "usage": "not_a_dict",  # Malformed usage
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "content_block": {"type": "text", "text": "Hello!"},
            },
            {"type": "message_stop"},
        ],
        "expected_usage": None,
    },
    # 5. Usage with None values
    {
        "name": "usage_with_none_values",
        "chunks": [
            {
                "type": "message_start",
                "message": {
                    "id": "123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": None, "output_tokens": None},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "content_block": {"type": "text", "text": "Hello!"},
            },
            {"type": "message_stop"},
        ],
        "expected_usage": None,
    },
    # 6. Same key appears in multiple chunks (later should overwrite earlier)
    {
        "name": "same_key_in_multiple_chunks",
        "chunks": [
            {
                "type": "message_start",
                "message": {
                    "id": "123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 8, "output_tokens": 1},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "content_block": {"type": "text", "text": "Hello!"},
            },
            {
                "type": "message_delta",
                "delta": {},
                "usage": {"input_tokens": 10, "output_tokens": 5},  # Overwrites earlier values
            },
            {
                "type": "message_delta",
                "delta": {},
                "usage": {"output_tokens": 12},  # Only updates output_tokens
            },
            {"type": "message_stop"},
        ],
        "expected_usage": {"input_tokens": 10, "output_tokens": 12, "total_tokens": 22},
    },
]


@pytest.mark.parametrize(
    "test_case",
    STREAM_TOKEN_USAGE_EDGE_CASES,
    ids=[c["name"] for c in STREAM_TOKEN_USAGE_EDGE_CASES],
)
def test_bedrock_autolog_stream_token_usage_edge_cases(test_case):
    """Test streaming token usage edge cases."""
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")
    request_body = json.dumps(_ANTHROPIC_REQUEST)

    # Mimic event stream
    def dummy_stream():
        for chunk in test_case["chunks"]:
            yield {"chunk": {"bytes": json.dumps(chunk).encode("utf-8")}}

    with mock.patch(
        "botocore.client.BaseClient._make_api_call",
        return_value={"body": dummy_stream()},
    ):
        response = client.invoke_model_with_response_stream(
            body=request_body, modelId=_ANTHROPIC_MODEL_ID
        )

        # Consume the stream
        list(response["body"])

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]

    # Check token usage
    _assert_token_usage_matches(span, test_case["expected_usage"])


@pytest.mark.skipif(not _IS_CONVERSE_API_AVAILABLE, reason="Converse API is not available")
@pytest.mark.parametrize(
    "test_case",
    CONVERSE_TOKEN_USAGE_EDGE_CASES,
    ids=[c["name"] for c in CONVERSE_TOKEN_USAGE_EDGE_CASES],
)
def test_bedrock_autolog_converse_token_usage_edge_cases(test_case):
    """Test converse token usage edge cases."""
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    with mock.patch(
        "botocore.client.BaseClient._make_api_call",
        return_value=test_case["response"],
    ):
        client.converse(**_CONVERSE_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]

    # Check token usage
    _assert_token_usage_matches(span, test_case["expected_usage"])


# Helper function to create converse stream events with usage
def _create_converse_stream_events_with_usage(usage_data):
    """Create converse stream events with the given usage data in metadata."""
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Hello!"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]

    if usage_data is not None:
        events.append({"metadata": {"usage": usage_data}})

    return events


# Converse stream token usage edge cases using shared data
CONVERSE_STREAM_TOKEN_USAGE_EDGE_CASES = [
    {
        "name": f"converse_stream_{case['name']}",
        "events": _create_converse_stream_events_with_usage(case["usage_data"]),
        "expected_usage": case["expected_usage"],
    }
    for case in TOKEN_USAGE_EDGE_CASE_DATA
] + [
    # Additional edge case specific to converse stream (malformed usage)
    {
        "name": "converse_stream_malformed_usage_in_metadata",
        "events": [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "Hello!"}}},
            {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": "not_a_dict"}},  # Malformed usage
        ],
        "expected_usage": None,
    },
]


@pytest.mark.skipif(not _IS_CONVERSE_API_AVAILABLE, reason="Converse API is not available")
@pytest.mark.parametrize(
    "test_case",
    CONVERSE_STREAM_TOKEN_USAGE_EDGE_CASES,
    ids=[c["name"] for c in CONVERSE_STREAM_TOKEN_USAGE_EDGE_CASES],
)
def test_bedrock_autolog_converse_stream_token_usage_edge_cases(test_case):
    """Test converse stream token usage edge cases."""
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    def dummy_converse_stream():
        for event in test_case["events"]:
            yield event

    with mock.patch(
        "botocore.client.BaseClient._make_api_call",
        return_value={"stream": dummy_converse_stream()},
    ):
        response = client.converse_stream(**_CONVERSE_REQUEST)

        # Consume the stream
        list(response["stream"])

    traces = get_traces()
    assert len(traces) == 1
    span = traces[0].data.spans[0]

    # Check token usage
    _assert_token_usage_matches(span, test_case["expected_usage"])
