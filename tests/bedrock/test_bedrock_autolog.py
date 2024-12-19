import io
import json
from unittest import mock

import boto3
import pytest
from botocore.exceptions import NoCredentialsError
from botocore.response import StreamingBody
from packaging.version import Version

import mlflow

from tests.tracing.helper import get_traces

_IS_CONVERSE_API_AVAILABLE = Version(boto3.__version__) >= Version("1.35")


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
        {"role": "CHATBOT", "message": "Sure! How can I help you today?"},
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
    ("model_id", "llm_request", "llm_response"),
    [
        (
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            _ANTHROPIC_REQUEST,
            _ANTHROPIC_RESPONSE,
        ),
        (
            "ai21.jamba-instruct-v1:0",
            _AI21_JAMBA_REQUEST,
            _AI21_JAMBA_RESPONSE,
        ),
        (
            "us.amazon.nova-lite-v1:0",
            _AMAZON_NOVA_REQUEST,
            _AMAZON_NOVA_RESPONSE,
        ),
        (
            "cohere.command-r-plus-v1:0",
            _COHERE_REQUEST,
            _COHERE_RESPONSE,
        ),
        (
            "meta.llama3-8b-instruct-v1:0",
            _META_LLAMA_REQUEST,
            _META_LLAMA_RESPONSE,
        ),
    ],
)
def test_bedrock_autolog_invoke_model_llm(model_id, llm_request, llm_response):
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


@pytest.mark.skipif(not _IS_CONVERSE_API_AVAILABLE, reason="Converse API is not available")
def test_bedrock_autolog_converse():
    mlflow.bedrock.autolog()

    client = boto3.client("bedrock-runtime", region_name="us-west-2")

    dummy_response = {
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

    with mock.patch("botocore.client.BaseClient._make_api_call", return_value=dummy_response):
        response = client.converse(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            inferenceConfig={
                "maxTokens": 300,
                "temperature": 0.1,
                "topP": 0.9,
            },
        )

    assert response["output"]["message"]["content"][0]["text"] == "Hello! How can I help you today?"

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"

    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "BedrockRuntime.converse"
    assert span.inputs["messages"] == [{"role": "user", "content": [{"text": "Hi"}]}]
    assert span.inputs["modelId"] == "anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert span.outputs == dummy_response
