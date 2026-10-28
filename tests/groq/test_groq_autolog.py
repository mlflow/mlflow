import json
from unittest.mock import patch

import groq
import pytest
from groq.types.audio.transcription import Transcription
from groq.types.audio.translation import Translation
from groq.types.chat import ChatCompletionMessageToolCall
from groq.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)

import mlflow.groq
from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.distributed import (
    _get_tracing_headers_from_span,
    set_tracing_context_from_http_request_headers,
)
from mlflow.tracing.utils import aggregate_usage_from_spans
from mlflow.version import IS_TRACING_SDK_ONLY

from tests.tracing.helper import get_traces

DUMMY_CHAT_COMPLETION_REQUEST = {
    "model": "test_model",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "test message"}],
}

DUMMY_COMPLETION_USAGE = CompletionUsage(
    completion_tokens=648,
    prompt_tokens=20,
    total_tokens=668,
    completion_time=0.54,
    prompt_time=0.000181289,
    queue_time=0.012770949,
    total_time=0.540181289,
)

DUMMY_CHAT_COMPLETION_RESPONSE = ChatCompletion(
    id="chatcmpl-test-id",
    choices=[
        Choice(
            finish_reason="stop",
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content="test response",
                role="assistant",
                function_call=None,
                tool_calls=None,
                reasoning=None,
            ),
        )
    ],
    created=1733574047,
    model="llama3-8b-8192",
    object="chat.completion",
    system_fingerprint="fp_test",
    usage=DUMMY_COMPLETION_USAGE,
    x_groq={"id": "req_test"},
)


@pytest.fixture(autouse=True)
def init_state(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test_key")
    yield
    mlflow.groq.autolog(disable=True)


def test_chat_completion_autolog(mock_litellm_cost):
    mlflow.groq.autolog()
    client = groq.Groq()

    with patch("groq._client.Groq.post", return_value=DUMMY_CHAT_COMPLETION_RESPONSE):
        client.chat.completions.create(**DUMMY_CHAT_COMPLETION_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Completions"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == DUMMY_CHAT_COMPLETION_REQUEST
    assert span.outputs == DUMMY_CHAT_COMPLETION_RESPONSE.to_dict(exclude_unset=False)
    assert span.model_name == "test_model"

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 20,
        "output_tokens": 648,
        "total_tokens": 668,
    }
    if not IS_TRACING_SDK_ONLY:
        # Verify cost is calculated (20 input tokens * 1.0 + 648 output tokens * 2.0)
        assert span.llm_cost == {
            "input_cost": 20.0,
            "output_cost": 1296.0,
            "total_cost": 1316.0,
        }

    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "groq"

    assert traces[0].info.token_usage == {
        "input_tokens": 20,
        "output_tokens": 648,
        "total_tokens": 668,
    }

    mlflow.groq.autolog(disable=True)
    client = groq.Groq()

    with patch("groq._client.Groq.post", return_value=DUMMY_CHAT_COMPLETION_RESPONSE):
        client.chat.completions.create(**DUMMY_CHAT_COMPLETION_REQUEST)

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1


@pytest.mark.parametrize("user_headers", [None, {"X-Custom": "my-value"}])
def test_tracing_headers_parent_gateway_span_without_double_counting(user_headers):
    mlflow.groq.autolog()
    client = groq.Groq()
    request = DUMMY_CHAT_COMPLETION_REQUEST.copy()
    if user_headers is not None:
        request["extra_headers"] = user_headers

    with patch("groq._client.Groq.post", return_value=DUMMY_CHAT_COMPLETION_RESPONSE) as mock_post:
        client.chat.completions.create(**request)

    span = get_traces()[0].data.spans[0]
    sent_headers = mock_post.call_args.kwargs["options"]["headers"]
    # Stored spans do not retain the live sampling flag, so compare the trace and span IDs.
    assert (
        sent_headers["traceparent"].split("-")[:3]
        == _get_tracing_headers_from_span(span)["traceparent"].split("-")[:3]
    )
    if user_headers is not None:
        assert sent_headers["X-Custom"] == "my-value"
    assert span.inputs.get("extra_headers") == user_headers
    assert "traceparent" not in (span.inputs.get("extra_headers") or {})

    # A gateway span using the propagated context is a child, so its repeated usage is skipped.
    with set_tracing_context_from_http_request_headers(sent_headers):
        with mlflow.start_span("gateway") as gateway_span:
            gateway_span.set_attribute(
                SpanAttributeKey.CHAT_USAGE, span.get_attribute(SpanAttributeKey.CHAT_USAGE)
            )

    assert gateway_span.trace_id == span.trace_id
    assert gateway_span.parent_id == span.span_id
    assert aggregate_usage_from_spans([span, gateway_span]) == span.get_attribute(
        SpanAttributeKey.CHAT_USAGE
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]
DUMMY_TOOL_CALL_REQUEST = {
    "model": "test_model",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "What is 25 * 4 + 10?"}],
    "tools": TOOLS,
}
DUMMY_TOOL_CALL_RESPONSE = ChatCompletion(
    id="chatcmpl-test-id",
    choices=[
        Choice(
            finish_reason="stop",
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content=None,
                role="assistant",
                function_call=None,
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="tool call id",
                        function={
                            "name": "calculate",
                            "arguments": json.dumps({"expression": "25 * 4 + 10"}),
                        },
                        type="function",
                    )
                ],
                reasoning=None,
            ),
        )
    ],
    created=1733574047,
    model="llama3-8b-8192",
    object="chat.completion",
    system_fingerprint="fp_test",
    usage=DUMMY_COMPLETION_USAGE,
    x_groq={"id": "req_test"},
)


def test_tool_calling_autolog():
    mlflow.groq.autolog()
    client = groq.Groq()

    with patch("groq._client.Groq.post", return_value=DUMMY_TOOL_CALL_RESPONSE):
        client.chat.completions.create(**DUMMY_TOOL_CALL_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Completions"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == DUMMY_TOOL_CALL_REQUEST
    assert span.outputs == DUMMY_TOOL_CALL_RESPONSE.to_dict(exclude_unset=False)
    assert span.get_attribute("mlflow.chat.tools") == TOOLS
    assert span.model_name == "test_model"

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 20,
        "output_tokens": 648,
        "total_tokens": 668,
    }

    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "groq"

    assert traces[0].info.token_usage == {
        "input_tokens": 20,
        "output_tokens": 648,
        "total_tokens": 668,
    }


DUMMY_TOOL_RESPONSE_REQUEST = {
    "model": "test_model",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "What is 25 * 4 + 10?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tool call id",
                    "function": {
                        "name": "calculate",
                        "arguments": json.dumps({"expression": "25 * 4 + 10"}),
                    },
                    "type": "function",
                }
            ],
        },
        {"role": "tool", "name": "calculate", "content": json.dumps({"result": 110})},
    ],
    "tools": TOOLS,
}
DUMMY_TOOL_RESPONSE_RESPONSE = ChatCompletion(
    id="chatcmpl-test-id",
    choices=[
        Choice(
            finish_reason="stop",
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content="The result of the calculation is 110",
                role="assistant",
                function_call=None,
                reasoning=None,
                tool_calls=None,
            ),
        )
    ],
    created=1733574047,
    model="llama3-8b-8192",
    object="chat.completion",
    system_fingerprint="fp_test",
    usage=DUMMY_COMPLETION_USAGE,
    x_groq={"id": "req_test"},
)


def test_tool_response_autolog():
    mlflow.groq.autolog()
    client = groq.Groq()

    with patch("groq._client.Groq.post", return_value=DUMMY_TOOL_RESPONSE_RESPONSE):
        client.chat.completions.create(**DUMMY_TOOL_RESPONSE_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Completions"
    assert span.span_type == SpanType.CHAT_MODEL
    assert span.inputs == DUMMY_TOOL_RESPONSE_REQUEST
    assert span.outputs == DUMMY_TOOL_RESPONSE_RESPONSE.to_dict(exclude_unset=False)
    assert span.model_name == "test_model"

    assert span.get_attribute(SpanAttributeKey.CHAT_USAGE) == {
        "input_tokens": 20,
        "output_tokens": 648,
        "total_tokens": 668,
    }

    assert span.get_attribute(SpanAttributeKey.MESSAGE_FORMAT) == "groq"

    assert traces[0].info.token_usage == {
        "input_tokens": 20,
        "output_tokens": 648,
        "total_tokens": 668,
    }


BINARY_CONTENT = b"\x00\x00\x00\x14ftypM4A \x00\x00\x00\x00mdat\x00\x01\x02\x03"

DUMMY_AUDIO_TRANSCRIPTION_REQUEST = {
    "file": ("test_audio.m4a", BINARY_CONTENT),
    "model": "whisper-large-v3-turbo",
}

DUMMY_AUDIO_TRANSCRIPTION_RESPONSE = Transcription(text="Test audio", x_groq={"id": "req_test"})


def test_audio_transcription_autolog():
    mlflow.groq.autolog()
    client = groq.Groq()

    with patch("groq._client.Groq.post", return_value=DUMMY_AUDIO_TRANSCRIPTION_RESPONSE):
        client.audio.transcriptions.create(**DUMMY_AUDIO_TRANSCRIPTION_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Transcriptions"
    assert span.span_type == SpanType.LLM
    assert span.inputs["file"][0] == DUMMY_AUDIO_TRANSCRIPTION_REQUEST["file"][0]
    assert span.inputs["file"][1] == str(DUMMY_AUDIO_TRANSCRIPTION_REQUEST["file"][1])
    assert span.inputs["model"] == DUMMY_AUDIO_TRANSCRIPTION_REQUEST["model"]
    assert span.outputs == DUMMY_AUDIO_TRANSCRIPTION_RESPONSE.to_dict(exclude_unset=False)
    assert span.model_name == "whisper-large-v3-turbo"

    mlflow.groq.autolog(disable=True)
    client = groq.Groq()

    with patch("groq._client.Groq.post", return_value=DUMMY_AUDIO_TRANSCRIPTION_RESPONSE):
        client.audio.transcriptions.create(**DUMMY_AUDIO_TRANSCRIPTION_REQUEST)

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1


DUMMY_AUDIO_TRANSLATION_REQUEST = {
    "file": ("test_audio.m4a", BINARY_CONTENT),
    "model": "whisper-large-v3",
}

DUMMY_AUDIO_TRANSLATION_RESPONSE = Translation(text="Test audio", x_groq={"id": "req_test"})


def test_audio_translation_autolog():
    mlflow.groq.autolog()
    client = groq.Groq()

    with patch("groq._client.Groq.post", return_value=DUMMY_AUDIO_TRANSLATION_RESPONSE):
        client.audio.translations.create(**DUMMY_AUDIO_TRANSLATION_REQUEST)

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0].info.status == "OK"
    assert len(traces[0].data.spans) == 1
    span = traces[0].data.spans[0]
    assert span.name == "Translations"
    assert span.span_type == SpanType.LLM
    assert span.inputs["file"][0] == DUMMY_AUDIO_TRANSLATION_REQUEST["file"][0]
    assert span.inputs["file"][1] == str(DUMMY_AUDIO_TRANSLATION_REQUEST["file"][1])
    assert span.inputs["model"] == DUMMY_AUDIO_TRANSLATION_REQUEST["model"]
    assert span.outputs == DUMMY_AUDIO_TRANSLATION_RESPONSE.to_dict(exclude_unset=False)
    assert span.model_name == "whisper-large-v3"

    mlflow.groq.autolog(disable=True)
    client = groq.Groq()

    with patch("groq._client.Groq.post", return_value=DUMMY_AUDIO_TRANSLATION_RESPONSE):
        client.audio.translations.create(**DUMMY_AUDIO_TRANSLATION_REQUEST)

    # No new trace should be created
    traces = get_traces()
    assert len(traces) == 1
