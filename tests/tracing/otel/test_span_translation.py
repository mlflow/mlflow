import json
from typing import Any
from unittest import mock

import pytest

from mlflow.entities.span import Span, SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.otel.translation import (
    sanitize_attributes,
    translate_loaded_span,
    translate_span_type_from_otel,
    translate_span_when_storing,
)
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator
from mlflow.tracing.otel.translation.genai_semconv import GenAiTranslator
from mlflow.tracing.otel.translation.google_adk import GoogleADKTranslator
from mlflow.tracing.otel.translation.open_inference import OpenInferenceTranslator
from mlflow.tracing.otel.translation.traceloop import TraceloopTranslator
from mlflow.tracing.otel.translation.vercel_ai import VercelAITranslator


@pytest.mark.parametrize(
    ("translator", "otel_kind", "expected_type"),
    [
        (OpenInferenceTranslator, "LLM", SpanType.LLM),
        (OpenInferenceTranslator, "CHAIN", SpanType.CHAIN),
        (OpenInferenceTranslator, "AGENT", SpanType.AGENT),
        (OpenInferenceTranslator, "TOOL", SpanType.TOOL),
        (OpenInferenceTranslator, "RETRIEVER", SpanType.RETRIEVER),
        (OpenInferenceTranslator, "EMBEDDING", SpanType.EMBEDDING),
        (OpenInferenceTranslator, "RERANKER", SpanType.RERANKER),
        (OpenInferenceTranslator, "GUARDRAIL", SpanType.GUARDRAIL),
        (OpenInferenceTranslator, "EVALUATOR", SpanType.EVALUATOR),
        (TraceloopTranslator, "workflow", SpanType.WORKFLOW),
        (TraceloopTranslator, "task", SpanType.TASK),
        (TraceloopTranslator, "agent", SpanType.AGENT),
        (TraceloopTranslator, "tool", SpanType.TOOL),
        (GenAiTranslator, "chat", SpanType.CHAT_MODEL),
        (GenAiTranslator, "create_agent", SpanType.AGENT),
        (GenAiTranslator, "embeddings", SpanType.EMBEDDING),
        (GenAiTranslator, "execute_tool", SpanType.TOOL),
        (GenAiTranslator, "generate_content", SpanType.LLM),
        (GenAiTranslator, "invoke_agent", SpanType.AGENT),
        (GenAiTranslator, "text_completion", SpanType.LLM),
    ],
)
def test_translate_span_type_from_otel(
    translator: OtelSchemaTranslator, otel_kind: str, expected_type: SpanType
):
    attributes = {translator.SPAN_KIND_ATTRIBUTE_KEY: otel_kind}
    result = translate_span_type_from_otel(attributes)
    assert result == expected_type


@pytest.mark.parametrize(
    "attributes",
    [
        {"some.other.attribute": "value"},
        {OpenInferenceTranslator.SPAN_KIND_ATTRIBUTE_KEY: "UNKNOWN_TYPE"},
        {TraceloopTranslator.SPAN_KIND_ATTRIBUTE_KEY: "unknown_type"},
    ],
)
def test_translate_span_type_returns_none(attributes):
    result = translate_span_type_from_otel(attributes)
    assert result is None


@pytest.mark.parametrize(
    ("attr_key", "attr_value", "expected_type"),
    [
        (OpenInferenceTranslator.SPAN_KIND_ATTRIBUTE_KEY, json.dumps("LLM"), SpanType.LLM),
        (TraceloopTranslator.SPAN_KIND_ATTRIBUTE_KEY, json.dumps("agent"), SpanType.AGENT),
        (VercelAITranslator.SPAN_KIND_ATTRIBUTE_KEY, json.dumps("ai.generateText"), SpanType.LLM),
        (VercelAITranslator.SPAN_KIND_ATTRIBUTE_KEY, json.dumps("ai.toolCall"), SpanType.TOOL),
    ],
)
def test_json_serialized_values(attr_key, attr_value, expected_type):
    attributes = {attr_key: attr_value}
    result = translate_span_type_from_otel(attributes)
    assert result == expected_type


@pytest.mark.parametrize(
    ("attr_key", "attr_value", "expected_type"),
    [
        (OpenInferenceTranslator.SPAN_KIND_ATTRIBUTE_KEY, "LLM", SpanType.LLM),
        (TraceloopTranslator.SPAN_KIND_ATTRIBUTE_KEY, "agent", SpanType.AGENT),
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
    "translator", [OpenInferenceTranslator, GenAiTranslator, TraceloopTranslator]
)
@pytest.mark.parametrize("total_token_exists", [True, False])
def test_translate_token_usage_from_otel(translator: OtelSchemaTranslator, total_token_exists):
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    span_dict = {
        "attributes": {
            translator.INPUT_TOKEN_KEY: 100,
            translator.OUTPUT_TOKEN_KEY: 50,
        }
    }
    if total_token_exists and translator.TOTAL_TOKEN_KEY:
        span_dict["attributes"][translator.TOTAL_TOKEN_KEY] = 150

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
    "translator",
    [OpenInferenceTranslator, GenAiTranslator, GoogleADKTranslator],
)
@pytest.mark.parametrize(
    "input_value",
    ["test input", {"query": "test"}, 123],
)
@pytest.mark.parametrize("parent_id", [None, "parent_123"])
def test_translate_inputs_for_spans(
    parent_id: str | None, translator: OtelSchemaTranslator, input_value: Any
):
    span = mock.Mock(spec=Span)
    span.parent_id = parent_id
    for input_key in translator.INPUT_VALUE_KEYS:
        span_dict = {"attributes": {input_key: json.dumps(input_value)}}
        span.to_dict.return_value = span_dict

        result = translate_span_when_storing(span)

        assert result["attributes"][SpanAttributeKey.INPUTS] == json.dumps(input_value)


@pytest.mark.parametrize(
    "input_key",
    [
        "traceloop.entity.input",
        "gen_ai.prompt.0.content",
        "gen_ai.prompt.1.content",
        "gen_ai.completion.0.tool_calls.0.arguments",
        "gen_ai.completion.1.tool_calls.1.arguments",
    ],
)
@pytest.mark.parametrize(
    "input_value",
    ["test input", {"query": "test"}, 123],
)
def test_translate_inputs_for_spans_traceloop(input_key: str, input_value: Any):
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    span_dict = {"attributes": {input_key: json.dumps(input_value)}}
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)
    assert result["attributes"][SpanAttributeKey.INPUTS] == json.dumps(input_value)


@pytest.mark.parametrize(
    "translator",
    [OpenInferenceTranslator, GenAiTranslator, GoogleADKTranslator],
)
@pytest.mark.parametrize("parent_id", [None, "parent_123"])
def test_translate_outputs_for_spans(parent_id: str | None, translator: OtelSchemaTranslator):
    output_value = "test output"
    span = mock.Mock(spec=Span)
    span.parent_id = parent_id
    for output_key in translator.OUTPUT_VALUE_KEYS:
        span_dict = {"attributes": {output_key: json.dumps(output_value)}}
        span.to_dict.return_value = span_dict

        result = translate_span_when_storing(span)

        assert result["attributes"][SpanAttributeKey.OUTPUTS] == json.dumps(output_value)


@pytest.mark.parametrize(
    "output_key",
    [
        "traceloop.entity.output",
        "gen_ai.completion.0.content",
        "gen_ai.completion.1.content",
    ],
)
@pytest.mark.parametrize(
    "output_value",
    ["test input", {"query": "test"}, 123],
)
def test_translate_outputs_for_spans_traceloop(output_key: str, output_value: Any):
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    span_dict = {"attributes": {output_key: json.dumps(output_value)}}
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)
    assert result["attributes"][SpanAttributeKey.OUTPUTS] == json.dumps(output_value)


@pytest.mark.parametrize(
    (
        "parent_id",
        "attributes",
        "expected_inputs",
        "expected_outputs",
    ),
    [
        (
            "parent_123",
            {
                OpenInferenceTranslator.INPUT_VALUE_KEYS[0]: json.dumps("test input"),
                OpenInferenceTranslator.OUTPUT_VALUE_KEYS[0]: json.dumps("test output"),
            },
            "test input",
            "test output",
        ),
        (
            None,
            {
                SpanAttributeKey.INPUTS: json.dumps("existing input"),
                SpanAttributeKey.OUTPUTS: json.dumps("existing output"),
                OpenInferenceTranslator.INPUT_VALUE_KEYS[0]: json.dumps("new input"),
                OpenInferenceTranslator.OUTPUT_VALUE_KEYS[0]: json.dumps("new output"),
            },
            "existing input",
            "existing output",
        ),
    ],
)
def test_translate_inputs_outputs_edge_cases(
    parent_id,
    attributes,
    expected_inputs,
    expected_outputs,
):
    span = mock.Mock(spec=Span)
    span.parent_id = parent_id
    span_dict = {"attributes": attributes}
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    assert SpanAttributeKey.INPUTS in result["attributes"]
    inputs = json.loads(result["attributes"][SpanAttributeKey.INPUTS])
    assert inputs == expected_inputs
    assert SpanAttributeKey.OUTPUTS in result["attributes"]
    outputs = json.loads(result["attributes"][SpanAttributeKey.OUTPUTS])
    assert outputs == expected_outputs


@pytest.mark.parametrize(
    ("attributes", "expected_attributes"),
    [
        ({"some.attribute": json.dumps("value")}, {"some.attribute": json.dumps("value")}),
        (
            {"some.attribute": json.dumps(json.dumps("value"))},
            {"some.attribute": json.dumps("value")},
        ),
        (
            {"key": json.dumps(json.dumps({"x": "y"}))},
            {"key": json.dumps({"x": "y"})},
        ),
        (
            {"key": "string"},
            {"key": "string"},
        ),
        (
            {"key": json.dumps(True)},
            {"key": "true"},
        ),
        (
            {"key": json.dumps(123)},
            {"key": "123"},
        ),
        (
            {"key": json.dumps([1, 2, 3])},
            {"key": "[1, 2, 3]"},
        ),
    ],
)
def test_sanitize_attributes(attributes: dict[str, Any], expected_attributes: dict[str, Any]):
    result = sanitize_attributes(attributes)
    assert result == expected_attributes


@pytest.mark.parametrize(
    ("translator", "model_value"),
    [
        (GenAiTranslator, "gpt-4o-mini"),
        (OpenInferenceTranslator, "claude-3-5-sonnet-20241022"),
    ],
)
def test_translate_model_name_from_otel(translator: OtelSchemaTranslator, model_value: str):
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    # Test with the first MODEL_NAME_KEY from the translator
    model_attr_key = translator.MODEL_NAME_KEYS[0]
    span_dict = {"attributes": {model_attr_key: json.dumps(model_value)}}
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    assert SpanAttributeKey.MODEL in result["attributes"]
    model = json.loads(result["attributes"][SpanAttributeKey.MODEL])
    assert model == model_value


@pytest.mark.parametrize(
    ("translator", "provider_value"),
    [
        (GenAiTranslator, "openai"),
        (OpenInferenceTranslator, "anthropic"),
        (TraceloopTranslator, "azure"),
    ],
)
def test_translate_model_provider_from_otel(translator: OtelSchemaTranslator, provider_value: str):
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    span_dict = {"attributes": {translator.LLM_PROVIDER_KEY: json.dumps(provider_value)}}
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    assert SpanAttributeKey.MODEL_PROVIDER in result["attributes"]
    provider = json.loads(result["attributes"][SpanAttributeKey.MODEL_PROVIDER])
    assert provider == provider_value


@pytest.mark.parametrize(
    ("inputs_outputs_key", "model_value"),
    [
        (
            SpanAttributeKey.INPUTS,
            {"model": "mistral-large-latest", "temperature": 0.7, "messages": []},
        ),
        (SpanAttributeKey.OUTPUTS, {"model": "gpt-3.5-turbo", "choices": []}),
    ],
)
def test_translate_model_name_from_inputs_outputs(
    inputs_outputs_key: str, model_value: dict[str, Any]
):
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    span_dict = {"attributes": {inputs_outputs_key: json.dumps(model_value)}}
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    assert SpanAttributeKey.MODEL in result["attributes"]
    model = json.loads(result["attributes"][SpanAttributeKey.MODEL])
    assert model == model_value["model"]


@pytest.mark.parametrize(
    ("attributes", "expected_model"),
    [
        (
            {
                "gen_ai.response.model": '"gpt-4o-mini"',
                SpanAttributeKey.INPUTS: json.dumps({"model": "different-model"}),
            },
            "gpt-4o-mini",
        ),
        (
            {
                SpanAttributeKey.MODEL: json.dumps("existing-model"),
                "gen_ai.response.model": '"new-model"',
            },
            "existing-model",
        ),
        (
            {SpanAttributeKey.INPUTS: json.dumps({"temperature": 0.7, "messages": []})},
            None,
        ),
        ({}, None),
    ],
)
def test_translate_model_name_edge_cases(attributes: dict[str, Any], expected_model: str | None):
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    span_dict = {"attributes": attributes}
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    if expected_model:
        assert SpanAttributeKey.MODEL in result["attributes"]
        model = json.loads(result["attributes"][SpanAttributeKey.MODEL])
        assert model == expected_model
    else:
        assert SpanAttributeKey.MODEL not in result["attributes"]


@pytest.mark.parametrize(
    "translator",
    [OpenInferenceTranslator, GenAiTranslator],
)
def test_translate_cost_from_otel(translator: OtelSchemaTranslator, mock_litellm_cost):
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    model_attr_key = translator.MODEL_NAME_KEYS[0]
    span_dict = {
        "attributes": {
            translator.INPUT_TOKEN_KEY: 10,
            translator.OUTPUT_TOKEN_KEY: 20,
            model_attr_key: '"gpt-4o-mini"',
        }
    }
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    assert SpanAttributeKey.LLM_COST in result["attributes"]
    cost = json.loads(result["attributes"][SpanAttributeKey.LLM_COST])
    assert cost == {
        "input_cost": 10.0,
        "output_cost": 40.0,
        "total_cost": 50.0,
    }


@pytest.mark.parametrize(
    "translator",
    [OpenInferenceTranslator, GenAiTranslator],
)
def test_translate_cost_with_model_provider(translator: OtelSchemaTranslator, mock_litellm_cost):
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    model_attr_key = translator.MODEL_NAME_KEYS[0]
    span_dict = {
        "attributes": {
            translator.INPUT_TOKEN_KEY: 10,
            translator.OUTPUT_TOKEN_KEY: 20,
            model_attr_key: '"gpt-4o-mini"',
            translator.LLM_PROVIDER_KEY: '"openai"',
        }
    }
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    # Both model and provider should be stored separately
    assert SpanAttributeKey.MODEL in result["attributes"]
    assert SpanAttributeKey.MODEL_PROVIDER in result["attributes"]
    model = json.loads(result["attributes"][SpanAttributeKey.MODEL])
    provider = json.loads(result["attributes"][SpanAttributeKey.MODEL_PROVIDER])
    assert model == "gpt-4o-mini"
    assert provider == "openai"

    # Cost should still be calculated
    assert SpanAttributeKey.LLM_COST in result["attributes"]
    cost = json.loads(result["attributes"][SpanAttributeKey.LLM_COST])
    assert cost == {
        "input_cost": 10.0,
        "output_cost": 40.0,
        "total_cost": 50.0,
    }


@pytest.mark.parametrize(
    ("attributes", "should_have_cost"),
    [
        (
            {
                "gen_ai.usage.input_tokens": 5,
                "gen_ai.usage.output_tokens": 10,
                "gen_ai.response.model": '"claude-3-5-sonnet-20241022"',
            },
            True,
        ),
        (
            {
                "gen_ai.usage.input_tokens": 5,
                "gen_ai.usage.output_tokens": 10,
            },
            False,
        ),
        (
            {
                "gen_ai.response.model": '"gpt-4o-mini"',
            },
            False,
        ),
        ({}, False),
    ],
)
def test_translate_cost_edge_cases(
    attributes: dict[str, Any], should_have_cost: bool, mock_litellm_cost
):
    span = mock.Mock(spec=Span)
    span.parent_id = "parent_123"
    span_dict = {"attributes": attributes}
    span.to_dict.return_value = span_dict

    result = translate_span_when_storing(span)

    if should_have_cost:
        assert SpanAttributeKey.LLM_COST in result["attributes"]
        cost = json.loads(result["attributes"][SpanAttributeKey.LLM_COST])
        input_cost = attributes.get("gen_ai.usage.input_tokens", 0) * 1.0
        output_cost = attributes.get("gen_ai.usage.output_tokens", 0) * 2.0
        assert cost == {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        }
    else:
        assert SpanAttributeKey.LLM_COST not in result["attributes"]
