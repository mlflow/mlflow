import json

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags
from opentelemetry.trace.status import Status, StatusCode

from mlflow.openai.genai_semconv_converter import OpenAiSemconvConverter
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


# --- translate_span_to_genai edge cases ---


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
