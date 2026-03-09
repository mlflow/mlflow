import json

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags
from opentelemetry.trace.status import Status, StatusCode

from mlflow.tracing.constant import GenAiSemconvKey, SpanAttributeKey
from mlflow.tracing.export.genai_semconv.translator import (
    _build_genai_span_name,
    _build_readable_span,
    _get_genai_span_kind,
    _parse_json_attr,
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


# --- _parse_json_attr ---


def test_parse_json_attr_none():
    assert _parse_json_attr(None) is None


def test_parse_json_attr_json_string():
    assert _parse_json_attr('"gpt-4o"') == "gpt-4o"


def test_parse_json_attr_json_dict():
    assert _parse_json_attr('{"input_tokens": 10}') == {"input_tokens": 10}


def test_parse_json_attr_plain_string():
    assert _parse_json_attr("not json {") == "not json {"


def test_parse_json_attr_int():
    assert _parse_json_attr(42) == 42


def test_parse_json_attr_dict():
    assert _parse_json_attr({"key": "value"}) == {"key": "value"}


# --- _translate_universal_attributes ---


def test_translate_chat_model_span_type():
    attrs = {SpanAttributeKey.SPAN_TYPE: json.dumps("CHAT_MODEL")}
    result = _translate_universal_attributes(attrs)
    assert result[GenAiSemconvKey.OPERATION_NAME] == "chat"


def test_translate_llm_span_type():
    attrs = {SpanAttributeKey.SPAN_TYPE: json.dumps("LLM")}
    result = _translate_universal_attributes(attrs)
    assert result[GenAiSemconvKey.OPERATION_NAME] == "generate_content"


def test_translate_embedding_span_type():
    attrs = {SpanAttributeKey.SPAN_TYPE: json.dumps("EMBEDDING")}
    result = _translate_universal_attributes(attrs)
    assert result[GenAiSemconvKey.OPERATION_NAME] == "embeddings"


def test_translate_tool_span_type():
    attrs = {SpanAttributeKey.SPAN_TYPE: json.dumps("TOOL")}
    result = _translate_universal_attributes(attrs)
    assert result[GenAiSemconvKey.OPERATION_NAME] == "execute_tool"


def test_translate_agent_span_type():
    attrs = {SpanAttributeKey.SPAN_TYPE: json.dumps("AGENT")}
    result = _translate_universal_attributes(attrs)
    assert result[GenAiSemconvKey.OPERATION_NAME] == "invoke_agent"


def test_translate_unmapped_span_type_returns_empty():
    attrs = {SpanAttributeKey.SPAN_TYPE: json.dumps("CHAIN")}
    result = _translate_universal_attributes(attrs)
    assert GenAiSemconvKey.OPERATION_NAME not in result


def test_translate_workflow_span_type_returns_empty():
    attrs = {SpanAttributeKey.SPAN_TYPE: json.dumps("WORKFLOW")}
    result = _translate_universal_attributes(attrs)
    assert GenAiSemconvKey.OPERATION_NAME not in result


def test_translate_model_name():
    attrs = {
        SpanAttributeKey.SPAN_TYPE: json.dumps("CHAT_MODEL"),
        SpanAttributeKey.MODEL: json.dumps("gpt-4o"),
    }
    result = _translate_universal_attributes(attrs)
    assert result[GenAiSemconvKey.REQUEST_MODEL] == "gpt-4o"


def test_translate_provider():
    attrs = {
        SpanAttributeKey.SPAN_TYPE: json.dumps("CHAT_MODEL"),
        SpanAttributeKey.MODEL_PROVIDER: json.dumps("openai"),
    }
    result = _translate_universal_attributes(attrs)
    assert result[GenAiSemconvKey.PROVIDER_NAME] == "openai"


def test_translate_token_usage():
    usage = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    attrs = {
        SpanAttributeKey.SPAN_TYPE: json.dumps("CHAT_MODEL"),
        SpanAttributeKey.CHAT_USAGE: json.dumps(usage),
    }
    result = _translate_universal_attributes(attrs)
    assert result[GenAiSemconvKey.USAGE_INPUT_TOKENS] == 100
    assert result[GenAiSemconvKey.USAGE_OUTPUT_TOKENS] == 50


def test_translate_tool_span_with_inputs_outputs():
    tool_input = {"query": "what is MLflow?"}
    tool_output = {"result": "MLflow is a platform..."}
    attrs = {
        SpanAttributeKey.SPAN_TYPE: json.dumps("TOOL"),
        SpanAttributeKey.INPUTS: json.dumps(tool_input),
        SpanAttributeKey.OUTPUTS: json.dumps(tool_output),
    }
    result = _translate_universal_attributes(attrs)
    assert result[GenAiSemconvKey.OPERATION_NAME] == "execute_tool"
    assert json.loads(result[GenAiSemconvKey.TOOL_CALL_ARGUMENTS]) == tool_input
    assert json.loads(result[GenAiSemconvKey.TOOL_CALL_RESULT]) == tool_output


def test_translate_missing_attributes():
    result = _translate_universal_attributes({})
    assert result == {}


def test_translate_malformed_json_attributes():
    attrs = {
        SpanAttributeKey.SPAN_TYPE: "not valid json {",
        SpanAttributeKey.MODEL: "also not valid {",
    }
    result = _translate_universal_attributes(attrs)
    assert GenAiSemconvKey.OPERATION_NAME not in result


# --- _build_genai_span_name ---


def test_span_name_operation_and_model():
    attrs = {GenAiSemconvKey.OPERATION_NAME: "chat", GenAiSemconvKey.REQUEST_MODEL: "gpt-4o"}
    assert _build_genai_span_name("original", attrs) == "chat gpt-4o"


def test_span_name_operation_only():
    attrs = {GenAiSemconvKey.OPERATION_NAME: "chat"}
    assert _build_genai_span_name("original", attrs) == "chat"


def test_span_name_no_operation():
    assert _build_genai_span_name("original", {}) == "original"


# --- _get_genai_span_kind ---


def test_span_kind_chat_is_client():
    attrs = {GenAiSemconvKey.OPERATION_NAME: "chat"}
    assert _get_genai_span_kind(attrs, SpanKind.INTERNAL) == SpanKind.CLIENT


def test_span_kind_embeddings_is_client():
    attrs = {GenAiSemconvKey.OPERATION_NAME: "embeddings"}
    assert _get_genai_span_kind(attrs, SpanKind.INTERNAL) == SpanKind.CLIENT


def test_span_kind_generate_content_is_client():
    attrs = {GenAiSemconvKey.OPERATION_NAME: "generate_content"}
    assert _get_genai_span_kind(attrs, SpanKind.INTERNAL) == SpanKind.CLIENT


def test_span_kind_execute_tool_is_internal():
    attrs = {GenAiSemconvKey.OPERATION_NAME: "execute_tool"}
    assert _get_genai_span_kind(attrs, SpanKind.CLIENT) == SpanKind.INTERNAL


def test_span_kind_invoke_agent_is_internal():
    attrs = {GenAiSemconvKey.OPERATION_NAME: "invoke_agent"}
    assert _get_genai_span_kind(attrs, SpanKind.CLIENT) == SpanKind.INTERNAL


def test_span_kind_no_operation_keeps_original():
    assert _get_genai_span_kind({}, SpanKind.INTERNAL) == SpanKind.INTERNAL


# --- translate_span_to_genai (end-to-end) ---


def test_full_chat_span():
    attrs = {
        SpanAttributeKey.SPAN_TYPE: json.dumps("CHAT_MODEL"),
        SpanAttributeKey.MODEL: json.dumps("gpt-4o"),
        SpanAttributeKey.MODEL_PROVIDER: json.dumps("openai"),
        SpanAttributeKey.CHAT_USAGE: json.dumps(
            {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        ),
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

    assert GenAiSemconvKey.OPERATION_NAME not in result.attributes
    assert result.attributes["custom.attribute"] == "preserved"
    assert not any(k.startswith("mlflow.") for k in result.attributes)
    assert result.name == "my_chain"


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


# --- _build_readable_span ---


def test_build_readable_span_creates_new_with_overrides():
    span = _make_span(name="original", attributes={"key": "value"}, kind=SpanKind.INTERNAL)
    new_span = _build_readable_span(
        span,
        name="new_name",
        attributes={"new_key": "new_value"},
        kind=SpanKind.CLIENT,
    )

    assert new_span.name == "new_name"
    assert new_span.attributes == {"new_key": "new_value"}
    assert new_span.kind == SpanKind.CLIENT
    assert new_span.context == span.context
    assert new_span.start_time == span.start_time
    assert new_span.end_time == span.end_time
