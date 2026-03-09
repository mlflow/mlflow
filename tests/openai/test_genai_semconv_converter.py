import json

import openai
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import mlflow
from mlflow.openai.genai_semconv_converter import _convert_message
from mlflow.tracing.processor.otel import OtelSpanProcessor
from mlflow.tracing.provider import provider as tracer_provider_wrapper

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
    ("api", "expected_response_id", "expected_response_model"),
    [
        ("chat_completions", "chatcmpl-123", "gpt-4o-mini"),
        ("responses", "responses-123", "gpt-4o"),
    ],
)
def test_autolog_basic(
    client, genai_semconv_capture, api, expected_response_id, expected_response_model
):
    exporter, processor = genai_semconv_capture

    mlflow.openai.autolog()
    if api == "chat_completions":
        client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4o-mini",
            temperature=0.5,
            top_p=0.9,
            max_tokens=100,
            stop=["\n", "END"],
        )
    else:
        client.responses.create(input="Hi", model="gpt-4o-mini", temperature=0.5)

    chat_span = _get_chat_span(exporter, processor)
    assert chat_span.attributes["gen_ai.operation.name"] == "chat"
    assert chat_span.attributes["gen_ai.request.model"] == "gpt-4o-mini"
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
    assert output_msgs[0]["finish_reason"] == "stop"

    assert chat_span.attributes["gen_ai.response.id"] == expected_response_id
    assert chat_span.attributes["gen_ai.response.model"] == expected_response_model
    assert list(chat_span.attributes["gen_ai.response.finish_reasons"]) == ["stop"]

    # No mlflow.* attrs leaked
    assert not any(k.startswith("mlflow.") for k in chat_span.attributes)


@pytest.mark.parametrize(
    ("api", "expected_response_id", "expected_response_model"),
    [
        ("chat_completions", "chatcmpl-123", "gpt-4o-mini"),
        ("responses", "responses-123", "gpt-4o"),
    ],
)
def test_autolog_with_tool_calls(
    client, genai_semconv_capture, api, expected_response_id, expected_response_model
):
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
            model="gpt-4o-mini",
            temperature=0.7,
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
            model="gpt-4o-mini",
            temperature=0.7,
            tools=MOCK_CHAT_TOOLS,
            instructions="Be helpful",
        )

    chat_span = _get_chat_span(exporter, processor)
    assert chat_span.attributes["gen_ai.operation.name"] == "chat"
    assert chat_span.attributes["gen_ai.request.model"] == "gpt-4o-mini"
    assert chat_span.attributes["gen_ai.request.temperature"] == 0.7

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
    assert output_msgs[0]["finish_reason"] == "stop"

    assert chat_span.attributes["gen_ai.response.id"] == expected_response_id
    assert chat_span.attributes["gen_ai.response.model"] == expected_response_model
    assert list(chat_span.attributes["gen_ai.response.finish_reasons"]) == ["stop"]
    assert not any(k.startswith("mlflow.") for k in chat_span.attributes)
