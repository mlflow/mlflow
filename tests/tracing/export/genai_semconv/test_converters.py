import json

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags
from opentelemetry.trace.status import Status, StatusCode

from mlflow.openai.genai_semconv_converter import OpenAiSemconvConverter, _convert_message
from mlflow.tracing.constant import GenAiSemconvKey, SpanAttributeKey
from mlflow.tracing.export.genai_semconv.translator import _get_converter, translate_span_to_genai


def _make_span(name="test_span", attributes=None, kind=SpanKind.INTERNAL):
    context = SpanContext(
        trace_id=0x000000000000000000000000DEADBEEF,
        span_id=0x00000000DEADBEF0,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return ReadableSpan(
        name=name,
        context=context,
        kind=kind,
        attributes=attributes or {},
        start_time=1000000000,
        end_time=2000000000,
        status=Status(StatusCode.OK),
    )


# --- _get_converter dispatch ---


@pytest.mark.parametrize("fmt", ["openai", "groq", "bedrock"])
def test_get_converter_returns_openai_converter(fmt):
    converter = _get_converter(fmt)
    assert isinstance(converter, OpenAiSemconvConverter)


@pytest.mark.parametrize("fmt", [None, "anthropic", "unknown", ""])
def test_get_converter_returns_none_for_unsupported(fmt):
    assert _get_converter(fmt) is None


# --- _convert_message ---


def test_convert_message_basic():
    msg = {"role": "user", "content": "Hello"}
    assert _convert_message(msg) == {
        "role": "user",
        "parts": [{"type": "text", "content": "Hello"}],
    }


def test_convert_message_with_tool_calls():
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
            }
        ],
    }
    result = _convert_message(msg)
    assert result["role"] == "assistant"
    assert "tool_calls" not in result
    assert len(result["parts"]) == 1
    assert result["parts"][0] == {
        "type": "tool_call",
        "id": "call_123",
        "name": "get_weather",
        "arguments": {"city": "SF"},
    }


def test_convert_message_tool_response():
    msg = {"role": "tool", "content": "72°F", "tool_call_id": "call_123"}
    result = _convert_message(msg)
    assert result == {
        "role": "tool",
        "parts": [{"type": "tool_call_response", "id": "call_123", "result": "72°F"}],
    }


def test_convert_message_empty_dict():
    assert _convert_message({}) == {"role": "user", "parts": []}


# --- OpenAiSemconvConverter.convert_inputs ---


def test_convert_inputs_basic():
    converter = OpenAiSemconvConverter()
    inputs = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
    }
    result = converter.convert_inputs(inputs)
    # System messages are excluded from input messages
    assert len(result) == 1
    assert result[0] == {"role": "user", "parts": [{"type": "text", "content": "Hi"}]}


def test_convert_inputs_no_messages():
    converter = OpenAiSemconvConverter()
    assert converter.convert_inputs({"model": "gpt-4o"}) is None


def test_convert_inputs_messages_not_list():
    converter = OpenAiSemconvConverter()
    assert converter.convert_inputs({"messages": "not a list"}) is None


# --- OpenAiSemconvConverter.convert_system_instructions ---


def test_convert_system_instructions():
    converter = OpenAiSemconvConverter()
    inputs = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
    }
    result = converter.convert_system_instructions(inputs)
    assert result == [{"type": "text", "content": "You are helpful."}]


def test_convert_system_instructions_multiple():
    converter = OpenAiSemconvConverter()
    inputs = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
    }
    result = converter.convert_system_instructions(inputs)
    assert result == [
        {"type": "text", "content": "You are helpful."},
        {"type": "text", "content": "Be concise."},
    ]


def test_convert_system_instructions_none_when_absent():
    converter = OpenAiSemconvConverter()
    inputs = {"messages": [{"role": "user", "content": "Hi"}]}
    assert converter.convert_system_instructions(inputs) is None


# --- OpenAiSemconvConverter.convert_outputs ---


def test_convert_outputs_basic():
    converter = OpenAiSemconvConverter()
    outputs = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ]
    }
    result = converter.convert_outputs(outputs)
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert result[0]["parts"] == [{"type": "text", "content": "Hello!"}]
    assert result[0]["finish_reason"] == "stop"


def test_convert_outputs_with_tool_calls():
    converter = OpenAiSemconvConverter()
    outputs = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "test"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    result = converter.convert_outputs(outputs)
    assert "tool_calls" not in result[0]
    assert len(result[0]["parts"]) == 1
    assert result[0]["parts"][0] == {
        "type": "tool_call",
        "id": "call_1",
        "name": "search",
        "arguments": {"q": "test"},
    }
    assert result[0]["finish_reason"] == "tool_calls"


def test_convert_outputs_no_choices():
    converter = OpenAiSemconvConverter()
    assert converter.convert_outputs({"id": "chatcmpl-123"}) is None


def test_convert_outputs_streaming_delta():
    converter = OpenAiSemconvConverter()
    outputs = {
        "choices": [
            {
                "delta": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            }
        ]
    }
    result = converter.convert_outputs(outputs)
    assert result[0]["role"] == "assistant"
    assert result[0]["parts"] == [{"type": "text", "content": "Hi"}]


# --- extract_request_params ---


def test_extract_request_params():
    converter = OpenAiSemconvConverter()
    inputs = {
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "max_tokens": 100,
        "top_p": 0.9,
    }
    params = converter.extract_request_params(inputs)
    assert params[GenAiSemconvKey.REQUEST_TEMPERATURE] == 0.7
    assert params[GenAiSemconvKey.REQUEST_MAX_TOKENS] == 100
    assert params[GenAiSemconvKey.REQUEST_TOP_P] == 0.9


def test_extract_request_params_with_stop_string():
    converter = OpenAiSemconvConverter()
    params = converter.extract_request_params({"stop": "\n"})
    assert params[GenAiSemconvKey.REQUEST_STOP_SEQUENCES] == ["\n"]


def test_extract_request_params_with_stop_list():
    converter = OpenAiSemconvConverter()
    params = converter.extract_request_params({"stop": ["\n", "END"]})
    assert params[GenAiSemconvKey.REQUEST_STOP_SEQUENCES] == ["\n", "END"]


def test_extract_request_params_with_tools():
    converter = OpenAiSemconvConverter()
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}}]
    params = converter.extract_request_params({"tools": tools})
    # OpenAI's nested format should be flattened to spec format
    assert json.loads(params[GenAiSemconvKey.TOOL_DEFINITIONS]) == [
        {"type": "function", "name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}
    ]


def test_extract_request_params_empty():
    converter = OpenAiSemconvConverter()
    assert converter.extract_request_params({"messages": []}) == {}


# --- extract_response_attrs ---


def test_extract_response_attrs():
    converter = OpenAiSemconvConverter()
    outputs = {
        "id": "chatcmpl-abc123",
        "model": "gpt-4o-2024-05-13",
        "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    }
    attrs = converter.extract_response_attrs(outputs)
    assert attrs[GenAiSemconvKey.RESPONSE_ID] == "chatcmpl-abc123"
    assert attrs[GenAiSemconvKey.RESPONSE_MODEL] == "gpt-4o-2024-05-13"
    assert attrs[GenAiSemconvKey.RESPONSE_FINISH_REASONS] == ["stop"]


def test_extract_response_attrs_no_finish_reason():
    converter = OpenAiSemconvConverter()
    attrs = converter.extract_response_attrs({"id": "x", "choices": [{"message": {}}]})
    assert GenAiSemconvKey.RESPONSE_FINISH_REASONS not in attrs


def test_extract_response_attrs_empty():
    converter = OpenAiSemconvConverter()
    assert converter.extract_response_attrs({}) == {}


# --- translate (orchestration) ---


def test_translate_full():
    converter = OpenAiSemconvConverter()
    inputs = {
        "messages": [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ],
        "temperature": 0.5,
    }
    outputs = {
        "id": "chatcmpl-abc",
        "model": "gpt-4o",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hi there!"},
                "finish_reason": "stop",
            }
        ],
    }
    result = converter.translate(inputs, outputs)

    input_messages = json.loads(result[GenAiSemconvKey.INPUT_MESSAGES])
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"

    system_instructions = json.loads(result[GenAiSemconvKey.SYSTEM_INSTRUCTIONS])
    assert system_instructions == [{"type": "text", "content": "Be helpful"}]

    output_messages = json.loads(result[GenAiSemconvKey.OUTPUT_MESSAGES])
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["finish_reason"] == "stop"

    assert result[GenAiSemconvKey.REQUEST_TEMPERATURE] == 0.5
    assert result[GenAiSemconvKey.RESPONSE_ID] == "chatcmpl-abc"
    assert result[GenAiSemconvKey.RESPONSE_MODEL] == "gpt-4o"
    assert result[GenAiSemconvKey.RESPONSE_FINISH_REASONS] == ["stop"]


def test_translate_no_system_instructions():
    converter = OpenAiSemconvConverter()
    result = converter.translate({"messages": [{"role": "user", "content": "Hi"}]}, None)
    assert GenAiSemconvKey.INPUT_MESSAGES in result
    assert GenAiSemconvKey.SYSTEM_INSTRUCTIONS not in result


def test_translate_inputs_only():
    converter = OpenAiSemconvConverter()
    result = converter.translate({"messages": [{"role": "user", "content": "Hi"}]}, None)
    assert GenAiSemconvKey.INPUT_MESSAGES in result
    assert GenAiSemconvKey.OUTPUT_MESSAGES not in result


def test_translate_outputs_only():
    converter = OpenAiSemconvConverter()
    outputs = {
        "choices": [
            {"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}
        ]
    }
    result = converter.translate(None, outputs)
    assert GenAiSemconvKey.OUTPUT_MESSAGES in result
    assert GenAiSemconvKey.INPUT_MESSAGES not in result


# --- End-to-end: translate_span_to_genai with converter ---


def test_e2e_openai_chat_span():
    inputs = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is MLflow?"},
        ],
        "temperature": 0.7,
    }
    outputs = {
        "id": "chatcmpl-xyz",
        "model": "gpt-4o",
        "choices": [
            {
                "message": {"role": "assistant", "content": "MLflow is a platform..."},
                "finish_reason": "stop",
            }
        ],
    }
    attrs = {
        SpanAttributeKey.SPAN_TYPE: json.dumps("CHAT_MODEL"),
        SpanAttributeKey.MODEL: json.dumps("gpt-4o"),
        SpanAttributeKey.MODEL_PROVIDER: json.dumps("openai"),
        SpanAttributeKey.MESSAGE_FORMAT: json.dumps("openai"),
        SpanAttributeKey.INPUTS: json.dumps(inputs),
        SpanAttributeKey.OUTPUTS: json.dumps(outputs),
    }
    span = _make_span(name="ChatCompletion.create", attributes=attrs)
    result = translate_span_to_genai(span)

    # Universal attrs
    assert result.attributes[GenAiSemconvKey.OPERATION_NAME] == "chat"
    assert result.attributes[GenAiSemconvKey.REQUEST_MODEL] == "gpt-4o"

    # System instructions extracted separately
    system_instructions = json.loads(result.attributes[GenAiSemconvKey.SYSTEM_INSTRUCTIONS])
    assert system_instructions == [{"type": "text", "content": "You are helpful."}]

    # Input messages exclude system messages
    input_msgs = json.loads(result.attributes[GenAiSemconvKey.INPUT_MESSAGES])
    assert len(input_msgs) == 1
    assert input_msgs[0]["role"] == "user"

    output_msgs = json.loads(result.attributes[GenAiSemconvKey.OUTPUT_MESSAGES])
    assert len(output_msgs) == 1
    assert output_msgs[0]["parts"] == [{"type": "text", "content": "MLflow is a platform..."}]

    # Request params and response attrs
    assert result.attributes[GenAiSemconvKey.REQUEST_TEMPERATURE] == 0.7
    assert result.attributes[GenAiSemconvKey.RESPONSE_ID] == "chatcmpl-xyz"
    assert result.attributes[GenAiSemconvKey.RESPONSE_FINISH_REASONS] == ["stop"]

    # No mlflow.* attrs leaked
    assert not any(k.startswith("mlflow.") for k in result.attributes)


def test_e2e_no_converter_for_unknown_format():
    inputs = {"messages": [{"role": "user", "content": "Hi"}]}
    attrs = {
        SpanAttributeKey.SPAN_TYPE: json.dumps("CHAT_MODEL"),
        SpanAttributeKey.MESSAGE_FORMAT: json.dumps("custom_format"),
        SpanAttributeKey.INPUTS: json.dumps(inputs),
    }
    span = _make_span(attributes=attrs)
    result = translate_span_to_genai(span)

    # Universal attrs present, but no message conversion
    assert result.attributes[GenAiSemconvKey.OPERATION_NAME] == "chat"
    assert GenAiSemconvKey.INPUT_MESSAGES not in result.attributes


def test_e2e_converter_error_is_swallowed():
    attrs = {
        SpanAttributeKey.SPAN_TYPE: json.dumps("CHAT_MODEL"),
        SpanAttributeKey.MESSAGE_FORMAT: json.dumps("openai"),
        # Inputs that will cause converter to fail (not a dict structure)
        SpanAttributeKey.INPUTS: json.dumps("not a dict"),
    }
    span = _make_span(attributes=attrs)
    # Should not raise
    result = translate_span_to_genai(span)
    assert result.attributes[GenAiSemconvKey.OPERATION_NAME] == "chat"
