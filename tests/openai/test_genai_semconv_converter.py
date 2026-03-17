import importlib.metadata
import json

import openai
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from packaging.version import Version

import mlflow
from mlflow.openai.genai_semconv_converter import _convert_content, _convert_message
from mlflow.tracing.processor.otel import OtelSpanProcessor
from mlflow.tracing.provider import provider as tracer_provider_wrapper

from tests.tracing.helper import reset_autolog_state  # noqa: F401

MODEL = "gpt-4o-mini"

_openai_version = Version(importlib.metadata.version("openai"))
requires_responses_api = pytest.mark.skipif(
    _openai_version < Version("1.66.0"),
    reason="OpenAI < 1.66.0 does not support the Responses API",
)

MOCK_CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]

MOCK_RESPONSES_TOOLS = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        },
    }
]


@pytest.fixture
def genai_semconv_capture(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_OTEL_GENAI_SEMCONV", "true")
    exporter = InMemorySpanExporter()
    # Force-init the tracer provider so we can add our processor
    tracer_provider_wrapper.get_or_init_tracer("test")
    tp = tracer_provider_wrapper.get()
    processor = OtelSpanProcessor(span_exporter=exporter, export_metrics=False)
    # Skip trace registration since the regular MLflow processors already handle that
    processor._should_register_traces = False
    tp.add_span_processor(processor)
    yield exporter, processor
    processor.force_flush(timeout_millis=5000)
    processor.shutdown()


@pytest.fixture
def client(mock_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    return openai.OpenAI(api_key="test", base_url=mock_openai)


def _get_chat_span(exporter, processor):
    processor.force_flush(timeout_millis=5000)
    spans = exporter.get_finished_spans()
    return next(s for s in spans if s.attributes.get("gen_ai.operation.name") == "chat")


@pytest.mark.parametrize(
    "api",
    ["chat_completions", pytest.param("responses", marks=requires_responses_api)],
)
@pytest.mark.usefixtures("reset_autolog_state")
def test_autolog_basic(client, genai_semconv_capture, api):
    exporter, processor = genai_semconv_capture

    mlflow.openai.autolog()
    if api == "chat_completions":
        client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model=MODEL,
            temperature=0.5,
            top_p=0.9,
            max_tokens=100,
            stop=["\n", "END"],
        )
    else:
        client.responses.create(input="Hi", model=MODEL, temperature=0.5)

    chat_span = _get_chat_span(exporter, processor)
    assert chat_span.attributes["gen_ai.operation.name"] == "chat"
    assert chat_span.attributes["gen_ai.request.model"] == MODEL
    assert chat_span.attributes["gen_ai.request.temperature"] == 0.5

    if api == "chat_completions":
        assert chat_span.attributes["gen_ai.request.top_p"] == 0.9
        assert chat_span.attributes["gen_ai.request.max_tokens"] == 100
        assert list(chat_span.attributes["gen_ai.request.stop_sequences"]) == ["\n", "END"]

    input_msgs = json.loads(chat_span.attributes["gen_ai.input.messages"])
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["type"] == "text"
    assert input_msgs[0]["parts"][0]["content"] == "Hi"

    output_msgs = json.loads(chat_span.attributes["gen_ai.output.messages"])
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"

    assert chat_span.attributes["gen_ai.response.model"] == MODEL
    assert not any(k.startswith("mlflow.") for k in chat_span.attributes)


@pytest.mark.parametrize(
    "api",
    ["chat_completions", pytest.param("responses", marks=requires_responses_api)],
)
@pytest.mark.usefixtures("reset_autolog_state")
def test_autolog_with_tool_calls(client, genai_semconv_capture, api):
    exporter, processor = genai_semconv_capture

    mlflow.openai.autolog()
    if api == "chat_completions":
        client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "What's the weather in SF?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_123", "content": "Sunny"},
            ],
            model=MODEL,
            tools=MOCK_CHAT_TOOLS,
        )
    else:
        client.responses.create(
            input=[
                {"role": "user", "content": "What's the weather in SF?"},
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"city": "SF"}',
                    "status": "completed",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "Sunny",
                },
            ],
            model=MODEL,
            tools=MOCK_CHAT_TOOLS,
            instructions="Be helpful",
        )

    chat_span = _get_chat_span(exporter, processor)
    assert chat_span.attributes["gen_ai.operation.name"] == "chat"
    assert chat_span.attributes["gen_ai.request.model"] == MODEL

    tool_defs = json.loads(chat_span.attributes["gen_ai.tool.definitions"])
    assert "function" not in tool_defs[0]
    assert tool_defs[0]["name"] == "get_weather"

    input_msgs = json.loads(chat_span.attributes["gen_ai.input.messages"])
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["type"] == "text"
    assert input_msgs[0]["parts"][0]["content"] == "What's the weather in SF?"
    assert input_msgs[1]["role"] == "assistant"
    assert input_msgs[1]["parts"][0]["type"] == "tool_call"
    assert input_msgs[1]["parts"][0]["id"] == "call_123"
    assert input_msgs[1]["parts"][0]["name"] == "get_weather"
    assert input_msgs[1]["parts"][0]["arguments"] == {"city": "SF"}
    assert input_msgs[2]["role"] == "tool"
    assert input_msgs[2]["parts"][0]["type"] == "tool_call_response"
    assert input_msgs[2]["parts"][0]["id"] == "call_123"
    assert input_msgs[2]["parts"][0]["result"] == "Sunny"
    system_instructions = json.loads(chat_span.attributes["gen_ai.system_instructions"])
    assert system_instructions == [{"type": "text", "content": "Be helpful"}]

    output_msgs = json.loads(chat_span.attributes["gen_ai.output.messages"])
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"

    assert chat_span.attributes["gen_ai.response.model"] == MODEL
    assert not any(k.startswith("mlflow.") for k in chat_span.attributes)


@pytest.mark.parametrize(
    "api",
    ["chat_completions", pytest.param("responses", marks=requires_responses_api)],
)
@pytest.mark.usefixtures("reset_autolog_state")
def test_autolog_streaming(client, genai_semconv_capture, api):
    exporter, processor = genai_semconv_capture

    mlflow.openai.autolog()
    if api == "chat_completions":
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model=MODEL,
            stream=True,
        )
        for _ in stream:
            pass
    else:
        stream = client.responses.create(input="Hi", model=MODEL, stream=True)
        for _ in stream:
            pass

    chat_span = _get_chat_span(exporter, processor)
    assert chat_span.attributes["gen_ai.operation.name"] == "chat"
    assert chat_span.attributes["gen_ai.request.model"] == MODEL

    input_msgs = json.loads(chat_span.attributes["gen_ai.input.messages"])
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["type"] == "text"
    assert input_msgs[0]["parts"][0]["content"] == "Hi"

    output_msgs = json.loads(chat_span.attributes["gen_ai.output.messages"])
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"

    assert chat_span.attributes["gen_ai.response.model"] == MODEL
    assert not any(k.startswith("mlflow.") for k in chat_span.attributes)


@pytest.mark.parametrize(
    ("content_item", "expected"),
    [
        # Chat API: image_url with HTTP URL → UriPart
        (
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            {"type": "uri", "modality": "image", "uri": "https://example.com/img.png"},
        ),
        # Chat API: image_url with data URI → BlobPart
        (
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}},
            {"type": "blob", "modality": "image", "mime_type": "image/jpeg", "content": "abc123"},
        ),
        # Responses API: input_image with HTTP URL → UriPart
        (
            {"type": "input_image", "image_url": "https://example.com/img.png"},
            {"type": "uri", "modality": "image", "uri": "https://example.com/img.png"},
        ),
        # Responses API: input_image with data URI → BlobPart
        (
            {"type": "input_image", "image_url": "data:image/png;base64,xyz789"},
            {"type": "blob", "modality": "image", "mime_type": "image/png", "content": "xyz789"},
        ),
        # Chat API: input_audio → BlobPart
        (
            {"type": "input_audio", "input_audio": {"data": "audiodata", "format": "wav"}},
            {"type": "blob", "modality": "audio", "mime_type": "audio/wav", "content": "audiodata"},
        ),
        # Responses API: input_text → TextPart
        (
            {"type": "input_text", "text": "hello"},
            {"type": "text", "content": "hello"},
        ),
    ],
)
def test_convert_content_multimodal(content_item, expected):
    result = _convert_content([content_item])
    assert result == [expected]


def test_convert_message_audio_transcript_fallback():
    msg = {
        "role": "assistant",
        "content": None,
        "audio": {
            "id": "audio_abc123",
            "data": "SGVsbG8=",
            "expires_at": 9999999999,
            "transcript": "Yes, I am.",
        },
    }
    result = _convert_message(msg)
    assert result == {
        "role": "assistant",
        "parts": [{"type": "text", "content": "Yes, I am."}],
    }


def test_convert_message_audio_no_override():
    msg = {
        "role": "assistant",
        "content": "I have text.",
        "audio": {
            "id": "audio_abc123",
            "data": "SGVsbG8=",
            "expires_at": 9999999999,
            "transcript": "Different transcript.",
        },
    }
    result = _convert_message(msg)
    assert result == {
        "role": "assistant",
        "parts": [{"type": "text", "content": "I have text."}],
    }


def test_convert_message_no_audio_no_content():
    msg = {"role": "assistant", "content": None}
    result = _convert_message(msg)
    assert result == {"role": "assistant", "parts": []}
