import json
from unittest import mock

import pytest

from mlflow.entities.span import Span, SpanType
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.otel.translation import (
    translate_span_type_from_otel,
    translate_span_when_storing,
)
from mlflow.tracing.otel.translation.voltagent import VoltAgentTranslator


@pytest.mark.parametrize(
    ("attributes", "expected_type"),
    [
        ({"span.type": "agent"}, SpanType.AGENT),
        ({"span.type": "llm"}, SpanType.LLM),
        ({"span.type": "tool"}, SpanType.TOOL),
        ({"span.type": "memory"}, SpanType.MEMORY),
        ({"entity.type": "agent"}, SpanType.AGENT),
        ({"entity.type": "llm"}, SpanType.LLM),
        ({"entity.type": "tool"}, SpanType.TOOL),
        ({"entity.type": "memory"}, SpanType.MEMORY),
        ({"span.type": "llm", "entity.type": "agent"}, SpanType.LLM),
    ],
)
def test_voltagent_span_type_translation(attributes, expected_type):
    translator = VoltAgentTranslator()
    result = translator.translate_span_type(attributes)
    assert result == expected_type


@pytest.mark.parametrize(
    "attributes",
    [
        {"some.other.attribute": "value"},
        {"span.type": "unknown_type"},
        {"entity.type": "unknown_type"},
        {},
    ],
)
def test_voltagent_span_type_returns_none(attributes):
    translator = VoltAgentTranslator()
    result = translator.translate_span_type(attributes)
    assert result is None


@pytest.mark.parametrize(
    ("attributes", "expected_inputs", "expected_outputs", "output_is_json"),
    [
        (
            {
                "agent.messages": json.dumps(
                    [
                        {"role": "user", "content": "Hello, what can you do?"},
                        {"role": "assistant", "content": "I can help you with various tasks."},
                    ]
                ),
                "output": "I'm here to help!",
                "span.type": "agent",
                "voltagent.operation_id": "op-123",
            },
            [
                {"role": "user", "content": "Hello, what can you do?"},
                {"role": "assistant", "content": "I can help you with various tasks."},
            ],
            "I'm here to help!",
            False,
        ),
        (
            {
                "llm.messages": json.dumps(
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "What's the weather like?"},
                    ]
                ),
                "output": "I don't have access to real-time weather data.",
                "span.type": "llm",
            },
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather like?"},
            ],
            "I don't have access to real-time weather data.",
            False,
        ),
        (
            {
                "input": json.dumps({"location": "San Francisco"}),
                "output": json.dumps({"temperature": 72, "conditions": "sunny"}),
                "span.type": "tool",
            },
            {"location": "San Francisco"},
            {"temperature": 72, "conditions": "sunny"},
            True,
        ),
    ],
)
def test_voltagent_input_output_translation(
    attributes, expected_inputs, expected_outputs, output_is_json
):
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    span_dict = {"attributes": attributes}
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    inputs = json.loads(result["attributes"][SpanAttributeKey.INPUTS])
    assert inputs == expected_inputs

    outputs_raw = result["attributes"][SpanAttributeKey.OUTPUTS]
    if output_is_json:
        outputs = json.loads(outputs_raw)
        assert outputs == expected_outputs
    else:
        try:
            outputs = json.loads(outputs_raw)
        except json.JSONDecodeError:
            outputs = outputs_raw
        assert outputs == expected_outputs


@pytest.mark.parametrize(
    ("attributes", "expected_input_tokens", "expected_output_tokens", "expected_total_tokens"),
    [
        (
            {
                "usage.prompt_tokens": 100,
                "usage.completion_tokens": 50,
                "usage.total_tokens": 150,
            },
            100,
            50,
            150,
        ),
        (
            {
                "llm.usage.prompt_tokens": 200,
                "llm.usage.completion_tokens": 100,
                "llm.usage.total_tokens": 300,
            },
            200,
            100,
            300,
        ),
        (
            {
                "usage.prompt_tokens": 75,
                "usage.completion_tokens": 25,
                "llm.usage.prompt_tokens": 100,
                "llm.usage.completion_tokens": 50,
            },
            75,
            25,
            100,
        ),
    ],
)
def test_voltagent_token_usage_translation(
    attributes, expected_input_tokens, expected_output_tokens, expected_total_tokens
):
    translator = VoltAgentTranslator()

    input_tokens = translator.get_input_tokens(attributes)
    assert input_tokens == expected_input_tokens

    output_tokens = translator.get_output_tokens(attributes)
    assert output_tokens == expected_output_tokens

    total_tokens = translator.get_total_tokens(attributes)
    if "usage.total_tokens" in attributes or "llm.usage.total_tokens" in attributes:
        assert total_tokens == expected_total_tokens
    else:
        assert total_tokens is None


def test_voltagent_translator_detection_keys():
    translator = VoltAgentTranslator()

    assert "voltagent.operation_id" in translator.DETECTION_KEYS
    assert "voltagent.conversation_id" in translator.DETECTION_KEYS


def test_voltagent_translator_message_format():
    translator = VoltAgentTranslator()
    assert translator.MESSAGE_FORMAT == "voltagent"


def test_voltagent_translator_input_output_keys():
    translator = VoltAgentTranslator()

    assert "agent.messages" in translator.INPUT_VALUE_KEYS
    assert "llm.messages" in translator.INPUT_VALUE_KEYS
    assert "input" in translator.INPUT_VALUE_KEYS

    assert "output" in translator.OUTPUT_VALUE_KEYS


@pytest.mark.parametrize(
    ("attributes", "expected_type"),
    [
        # Test span.type attribute (used by child spans)
        ({"span.type": "agent"}, SpanType.AGENT),
        ({"span.type": "llm"}, SpanType.LLM),
        ({"span.type": "tool"}, SpanType.TOOL),
        ({"span.type": "memory"}, SpanType.MEMORY),
        # Test entity.type attribute (used by root agent spans)
        ({"entity.type": "agent"}, SpanType.AGENT),
        ({"entity.type": "llm"}, SpanType.LLM),
        ({"entity.type": "tool"}, SpanType.TOOL),
        ({"entity.type": "memory"}, SpanType.MEMORY),
        # Test span.type takes precedence over entity.type (child span scenario)
        ({"span.type": "llm", "entity.type": "agent"}, SpanType.LLM),
        ({"span.type": "tool", "entity.type": "agent"}, SpanType.TOOL),
        ({"span.type": "memory", "entity.type": "agent"}, SpanType.MEMORY),
        # Test with JSON-encoded values
        ({"span.type": '"llm"', "entity.type": '"agent"'}, SpanType.LLM),
        ({"entity.type": '"agent"'}, SpanType.AGENT),
    ],
)
def test_voltagent_span_type_from_otel(attributes, expected_type):
    result = translate_span_type_from_otel(attributes)
    assert result == expected_type
