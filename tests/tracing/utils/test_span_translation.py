import json
from unittest import mock

import pytest

from mlflow.entities.span import Span, SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.utils.span_translation import (
    translate_loaded_span,
    translate_span_type_from_otel,
    translate_span_when_storing,
)


@pytest.mark.parametrize(
    ("attr_key", "otel_kind", "expected_type"),
    [
        ("openinference.span.kind", "LLM", SpanType.LLM),
        ("openinference.span.kind", "CHAIN", SpanType.CHAIN),
        ("openinference.span.kind", "AGENT", SpanType.AGENT),
        ("openinference.span.kind", "TOOL", SpanType.TOOL),
        ("openinference.span.kind", "RETRIEVER", SpanType.RETRIEVER),
        ("openinference.span.kind", "EMBEDDING", SpanType.EMBEDDING),
        ("openinference.span.kind", "RERANKER", SpanType.RERANKER),
        ("openinference.span.kind", "GUARDRAIL", SpanType.GUARDRAIL),
        ("openinference.span.kind", "EVALUATOR", SpanType.EVALUATOR),
        ("traceloop.span.kind", "workflow", SpanType.WORKFLOW),
        ("traceloop.span.kind", "task", SpanType.TASK),
        ("traceloop.span.kind", "agent", SpanType.AGENT),
        ("traceloop.span.kind", "tool", SpanType.TOOL),
    ],
)
def test_translate_span_type_from_otel(attr_key, otel_kind, expected_type):
    attributes = {attr_key: otel_kind}
    result = translate_span_type_from_otel(attributes)
    assert result == expected_type


@pytest.mark.parametrize(
    "attributes",
    [
        {"some.other.attribute": "value"},
        {"openinference.span.kind": "UNKNOWN_TYPE"},
        {"traceloop.span.kind": "unknown_type"},
    ],
)
def test_translate_span_type_returns_none(attributes):
    result = translate_span_type_from_otel(attributes)
    assert result is None


@pytest.mark.parametrize(
    ("attr_key", "attr_value", "expected_type"),
    [
        ("openinference.span.kind", json.dumps("LLM"), SpanType.LLM),
        ("traceloop.span.kind", json.dumps("agent"), SpanType.AGENT),
    ],
)
def test_json_serialized_values(attr_key, attr_value, expected_type):
    attributes = {attr_key: attr_value}
    result = translate_span_type_from_otel(attributes)
    assert result == expected_type


def test_openinference_takes_precedence():
    attributes = {
        "openinference.span.kind": "LLM",
        "traceloop.span.kind": "workflow",
    }
    result = translate_span_type_from_otel(attributes)
    assert result == SpanType.LLM


@pytest.mark.parametrize(
    ("attr_key", "attr_value", "expected_type"),
    [
        ("openinference.span.kind", "LLM", SpanType.LLM),
        ("traceloop.span.kind", "agent", SpanType.AGENT),
    ],
)
def test_translate_loaded_span_sets_span_type(attr_key, attr_value, expected_type):
    span_dict = {"attributes": {attr_key: attr_value}}
    result = translate_loaded_span(span_dict)

    assert SpanAttributeKey.SPAN_TYPE in result["attributes"]
    span_type = json.loads(result["attributes"][SpanAttributeKey.SPAN_TYPE])
    assert span_type == expected_type


@pytest.mark.parametrize(
    ("span_dict", "should_have_span_type", "expected_type"),
    [
        (
            {
                "attributes": {
                    SpanAttributeKey.SPAN_TYPE: json.dumps(SpanType.TOOL),
                    "openinference.span.kind": "LLM",
                }
            },
            True,
            SpanType.TOOL,
        ),
        ({"attributes": {"some.other.attribute": "value"}}, False, None),
        ({}, False, None),
    ],
)
def test_translate_loaded_span_edge_cases(span_dict, should_have_span_type, expected_type):
    result = translate_loaded_span(span_dict)
    if should_have_span_type:
        assert SpanAttributeKey.SPAN_TYPE in result["attributes"]
        span_type = json.loads(result["attributes"][SpanAttributeKey.SPAN_TYPE])
        assert span_type == expected_type
    else:
        assert SpanAttributeKey.SPAN_TYPE not in result.get("attributes", {})


@pytest.mark.parametrize(
    ("input_tokens_key", "output_tokens_key", "total_tokens_key"),
    [
        ("gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "llm.usage.total_tokens"),
        ("gen_ai.usage.prompt_tokens", "gen_ai.usage.completion_tokens", None),
        ("llm.token_count.prompt", "llm.token_count.completion", "llm.token_count.total"),
    ],
)
def test_translate_token_usage_from_otel(input_tokens_key, output_tokens_key, total_tokens_key):
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    span_dict = {
        "attributes": {
            input_tokens_key: 100,
            output_tokens_key: 50,
        }
    }
    if total_tokens_key:
        span_dict["attributes"][total_tokens_key] = 150

    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    assert SpanAttributeKey.CHAT_USAGE in result["attributes"]
    usage = json.loads(result["attributes"][SpanAttributeKey.CHAT_USAGE])
    assert usage[TokenUsageKey.INPUT_TOKENS] == 100
    assert usage[TokenUsageKey.OUTPUT_TOKENS] == 50
    assert usage[TokenUsageKey.TOTAL_TOKENS] == 150


@pytest.mark.parametrize(
    ("attributes", "expected_input", "expected_output", "expected_total"),
    [
        (
            {"gen_ai.usage.input_tokens": 75, "gen_ai.usage.output_tokens": 25},
            75,
            25,
            100,
        ),
        (
            {
                SpanAttributeKey.CHAT_USAGE: json.dumps(
                    {
                        TokenUsageKey.INPUT_TOKENS: 200,
                        TokenUsageKey.OUTPUT_TOKENS: 100,
                        TokenUsageKey.TOTAL_TOKENS: 300,
                    }
                ),
                "gen_ai.usage.input_tokens": 50,
                "gen_ai.usage.output_tokens": 25,
            },
            200,
            100,
            300,
        ),
    ],
)
def test_translate_token_usage_edge_cases(
    attributes, expected_input, expected_output, expected_total
):
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    span_dict = {"attributes": attributes}
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    usage = json.loads(result["attributes"][SpanAttributeKey.CHAT_USAGE])
    assert usage[TokenUsageKey.INPUT_TOKENS] == expected_input
    assert usage[TokenUsageKey.OUTPUT_TOKENS] == expected_output
    assert usage[TokenUsageKey.TOTAL_TOKENS] == expected_total


@pytest.mark.parametrize(
    ("input_key", "input_value"),
    [
        ("input.value", "test input"),
        ("traceloop.entity.input", {"query": "test"}),
    ],
)
def test_translate_inputs_for_root_span(input_key, input_value):
    span = mock.Mock(spec=Span)
    span.parent_id = None  # Root span
    span_dict = {"attributes": {input_key: input_value}}
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    assert result["attributes"][SpanAttributeKey.INPUTS] == input_value


@pytest.mark.parametrize(
    ("output_key", "output_value"),
    [
        ("output.value", "test output"),
        ("traceloop.entity.output", {"result": "success"}),
    ],
)
def test_translate_outputs_for_root_span(output_key, output_value):
    span = mock.Mock(spec=Span)
    span.parent_id = None  # Root span
    span_dict = {"attributes": {output_key: output_value}}
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    assert result["attributes"][SpanAttributeKey.OUTPUTS] == output_value


@pytest.mark.parametrize(
    (
        "parent_id",
        "attributes",
        "should_have_inputs",
        "expected_inputs",
        "should_have_outputs",
        "expected_outputs",
    ),
    [
        (
            "parent_123",
            {"input.value": "test input", "output.value": "test output"},
            False,
            None,
            False,
            None,
        ),
        (
            None,
            {
                SpanAttributeKey.INPUTS: json.dumps("existing input"),
                SpanAttributeKey.OUTPUTS: json.dumps("existing output"),
                "input.value": "new input",
                "output.value": "new output",
            },
            True,
            "existing input",
            True,
            "existing output",
        ),
    ],
)
def test_translate_inputs_outputs_edge_cases(
    parent_id,
    attributes,
    should_have_inputs,
    expected_inputs,
    should_have_outputs,
    expected_outputs,
):
    span = mock.Mock(spec=Span)
    span.parent_id = parent_id
    span_dict = {"attributes": attributes}
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    if should_have_inputs:
        assert SpanAttributeKey.INPUTS in result["attributes"]
        inputs = json.loads(result["attributes"][SpanAttributeKey.INPUTS])
        assert inputs == expected_inputs
    else:
        assert SpanAttributeKey.INPUTS not in result["attributes"]

    if should_have_outputs:
        assert SpanAttributeKey.OUTPUTS in result["attributes"]
        outputs = json.loads(result["attributes"][SpanAttributeKey.OUTPUTS])
        assert outputs == expected_outputs
    else:
        assert SpanAttributeKey.OUTPUTS not in result["attributes"]
