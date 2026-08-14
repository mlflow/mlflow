import json

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags
from opentelemetry.trace.status import Status, StatusCode

from mlflow.tracing.constant import GenAiSemconvKey, SpanAttributeKey
from mlflow.tracing.export.genai_semconv.translator import (
    _translate_universal_attributes,
    translate_span_to_genai,
)


def _make_span(
    name="test_span",
    attributes=None,
    kind=SpanKind.INTERNAL,
    start_time=1000000000,
    end_time=2000000000,
):
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
        start_time=start_time,
        end_time=end_time,
        status=Status(StatusCode.OK),
    )


# --- _translate_universal_attributes ---


@pytest.mark.parametrize(
    ("span_type", "expected_operation"),
    [
        ("CHAT_MODEL", "chat"),
        ("LLM", "generate_content"),
        ("EMBEDDING", "embeddings"),
        ("TOOL", "execute_tool"),
        ("AGENT", "invoke_agent"),
    ],
)
def test_translate_span_type_to_operation(span_type, expected_operation):
    span = _make_span(attributes={SpanAttributeKey.SPAN_TYPE: json.dumps(span_type)})
    result = _translate_universal_attributes(span)
    assert result[GenAiSemconvKey.OPERATION_NAME] == expected_operation


@pytest.mark.parametrize(
    "span_type",
    ["CHAIN", "WORKFLOW", "PARSER", "MEMORY", "GUARDRAIL", "EVALUATOR", "RETRIEVER", "RERANKER"],
)
def test_translate_unmapped_span_type_passes_through_value(span_type):
    span = _make_span(attributes={SpanAttributeKey.SPAN_TYPE: json.dumps(span_type)})
    result = _translate_universal_attributes(span)
    assert result[GenAiSemconvKey.OPERATION_NAME] == span_type


def test_translate_model_name():
    span = _make_span(
        attributes={
            SpanAttributeKey.SPAN_TYPE: json.dumps("CHAT_MODEL"),
            SpanAttributeKey.MODEL: json.dumps("gpt-4o"),
        }
    )
    result = _translate_universal_attributes(span)
    assert result[GenAiSemconvKey.REQUEST_MODEL] == "gpt-4o"


def test_translate_provider():
    span = _make_span(
        attributes={
            SpanAttributeKey.SPAN_TYPE: json.dumps("CHAT_MODEL"),
            SpanAttributeKey.MODEL_PROVIDER: json.dumps("openai"),
        }
    )
    result = _translate_universal_attributes(span)
    assert result[GenAiSemconvKey.PROVIDER_NAME] == "openai"


def test_translate_token_usage():
    usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    span = _make_span(
        attributes={
            SpanAttributeKey.SPAN_TYPE: json.dumps("CHAT_MODEL"),
            SpanAttributeKey.CHAT_USAGE: json.dumps(usage),
        }
    )
    result = _translate_universal_attributes(span)
    assert result[GenAiSemconvKey.USAGE_INPUT_TOKENS] == 100
    assert result[GenAiSemconvKey.USAGE_OUTPUT_TOKENS] == 50


def test_translate_tool_span_with_inputs_outputs():
    tool_input = {"query": "what is MLflow?"}
    tool_output = {"result": "MLflow is a platform..."}
    span = _make_span(
        attributes={
            SpanAttributeKey.SPAN_TYPE: json.dumps("TOOL"),
            SpanAttributeKey.INPUTS: json.dumps(tool_input),
            SpanAttributeKey.OUTPUTS: json.dumps(tool_output),
        }
    )
    result = _translate_universal_attributes(span)
    assert result[GenAiSemconvKey.OPERATION_NAME] == "execute_tool"
    assert json.loads(result[GenAiSemconvKey.TOOL_CALL_ARGUMENTS]) == tool_input
    assert json.loads(result[GenAiSemconvKey.TOOL_CALL_RESULT]) == tool_output


def test_translate_missing_attributes():
    span = _make_span(attributes={})
    result = _translate_universal_attributes(span)
    assert result == {}


def test_translate_malformed_json_attributes():
    span = _make_span(
        attributes={
            SpanAttributeKey.SPAN_TYPE: "not valid json {",
            SpanAttributeKey.MODEL: "also not valid {",
        }
    )
    result = _translate_universal_attributes(span)
    assert GenAiSemconvKey.OPERATION_NAME not in result


# --- _build_genai_span_name / _get_genai_span_kind (tested via translate_span_to_genai) ---


@pytest.mark.parametrize(
    ("operation", "model", "expected_name", "expected_kind"),
    [
        ("chat", "gpt-4o", "chat gpt-4o", SpanKind.CLIENT),
        (
            "embeddings",
            "text-embedding-3-small",
            "embeddings text-embedding-3-small",
            SpanKind.CLIENT,
        ),
        ("generate_content", "gemini-pro", "generate_content gemini-pro", SpanKind.CLIENT),
        ("execute_tool", None, "execute_tool", SpanKind.INTERNAL),
        ("invoke_agent", None, "invoke_agent", SpanKind.INTERNAL),
    ],
)
def test_span_name_and_kind(operation, model, expected_name, expected_kind):
    operation_to_type = {
        "chat": "CHAT_MODEL",
        "generate_content": "LLM",
        "embeddings": "EMBEDDING",
        "execute_tool": "TOOL",
        "invoke_agent": "AGENT",
    }
    attrs = {SpanAttributeKey.SPAN_TYPE: json.dumps(operation_to_type[operation])}
    if model:
        attrs[SpanAttributeKey.MODEL] = json.dumps(model)

    span = _make_span(name="original", attributes=attrs)
    result = translate_span_to_genai(span)
    assert result.name == expected_name
    assert result.kind == expected_kind


def test_span_name_unmapped_type_uses_span_type():
    span = _make_span(name="my_chain", attributes={SpanAttributeKey.SPAN_TYPE: json.dumps("CHAIN")})
    result = translate_span_to_genai(span)
    assert result.name == "CHAIN"


# --- translate_span_to_genai (end-to-end) ---


def test_full_chat_span():
    attrs = {
        SpanAttributeKey.SPAN_TYPE: json.dumps("CHAT_MODEL"),
        SpanAttributeKey.MODEL: json.dumps("gpt-4o"),
        SpanAttributeKey.MODEL_PROVIDER: json.dumps("openai"),
        SpanAttributeKey.CHAT_USAGE: json.dumps({
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }),
    }
    span = _make_span(name="ChatCompletion.create", attributes=attrs)
    result = translate_span_to_genai(span)

    assert result.name == "chat gpt-4o"
    assert result.kind == SpanKind.CLIENT
    assert result.attributes[GenAiSemconvKey.OPERATION_NAME] == "chat"
    assert result.attributes[GenAiSemconvKey.REQUEST_MODEL] == "gpt-4o"
    assert result.attributes[GenAiSemconvKey.PROVIDER_NAME] == "openai"
    assert result.attributes[GenAiSemconvKey.USAGE_INPUT_TOKENS] == 100
    assert result.attributes[GenAiSemconvKey.USAGE_OUTPUT_TOKENS] == 50
    assert not any(k.startswith("mlflow.") for k in result.attributes)


def test_unmapped_span_type_passes_through():
    attrs = {
        SpanAttributeKey.SPAN_TYPE: json.dumps("CHAIN"),
        "custom.attribute": "preserved",
    }
    span = _make_span(name="my_chain", attributes=attrs)
    result = translate_span_to_genai(span)

    assert result.attributes[GenAiSemconvKey.OPERATION_NAME] == "CHAIN"
    assert result.attributes["custom.attribute"] == "preserved"
    assert not any(k.startswith("mlflow.") for k in result.attributes)
    assert result.name == "CHAIN"


def test_non_mlflow_attributes_preserved():
    attrs = {
        SpanAttributeKey.SPAN_TYPE: json.dumps("CHAT_MODEL"),
        "http.method": "POST",
        "http.url": "https://api.openai.com/v1/chat/completions",
    }
    span = _make_span(attributes=attrs)
    result = translate_span_to_genai(span)

    assert result.attributes["http.method"] == "POST"
    assert result.attributes["http.url"] == "https://api.openai.com/v1/chat/completions"


def test_span_context_preserved():
    attrs = {SpanAttributeKey.SPAN_TYPE: json.dumps("CHAT_MODEL")}
    span = _make_span(attributes=attrs)
    result = translate_span_to_genai(span)

    assert result.context == span.context
    assert result.start_time == span.start_time
    assert result.end_time == span.end_time
    assert result.status == span.status


def test_empty_attributes():
    span = _make_span(attributes={})
    result = translate_span_to_genai(span)
    assert result.attributes == {}
