import inspect
from types import SimpleNamespace
from unittest import mock

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import SpanExportResult

from mlflow.entities import LiveSpan, Span
from mlflow.entities.trace_info import TraceInfo, TraceLocation, TraceState
from mlflow.tracing.constant import (
    MCP_SPAN_ATTRIBUTE_KEYS,
    TRACE_SCHEMA_VERSION_KEY,
    SpanAttributeKey,
)
from mlflow.tracing.otel.translation import strip_otlp_double_encoding
from mlflow.tracing.processor.otel import OtelSpanProcessor
from mlflow.tracing.trace_manager import ManagerTrace
from mlflow.tracing.utils import get_mlflow_span_for_otel_span
from mlflow.tracing.utils.otlp import _decode_otel_proto_anyvalue

from tests.tracing.helper import create_mock_otel_span


@pytest.fixture
def encode_spans():
    return pytest.importorskip(
        "opentelemetry.exporter.otlp.proto.common.trace_encoder"
    ).encode_spans


def _make_processor():
    exporter = mock.MagicMock()
    exporter.export.return_value = SpanExportResult.SUCCESS
    processor = OtelSpanProcessor(span_exporter=exporter, export_metrics=False)
    return processor, exporter


def _make_manager_trace(tags: dict[str, str]) -> ManagerTrace:
    from mlflow.entities import Trace, TraceData

    info = TraceInfo(
        trace_id="tr-abc123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=0,
        execution_duration=None,
        state=TraceState.OK,
        trace_metadata={TRACE_SCHEMA_VERSION_KEY: "3"},
        tags=tags,
    )
    return ManagerTrace(trace=Trace(info=info, data=TraceData()), prompts=[], is_remote_trace=False)


def test_on_end_emits_per_tag_attributes():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    root_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x1234)
    assert root_span.parent is None

    manager_trace = _make_manager_trace(tags={"env": "production", "version": "1.0"})

    with mock.patch.object(processor._trace_manager, "pop_trace", return_value=manager_trace):
        processor.on_end(root_span)

    assert root_span._attributes.get(SpanAttributeKey.TRACE_TAG_PREFIX + "env") == "production"
    assert root_span._attributes.get(SpanAttributeKey.TRACE_TAG_PREFIX + "version") == "1.0"


def test_on_end_skips_tag_emission_when_no_tags():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    root_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x1234)
    manager_trace = _make_manager_trace(tags={})

    with mock.patch.object(processor._trace_manager, "pop_trace", return_value=manager_trace):
        processor.on_end(root_span)

    assert not any(k.startswith(SpanAttributeKey.TRACE_TAG_PREFIX) for k in root_span._attributes)


def test_on_end_skips_tag_emission_for_child_span():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    child_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x5678, parent_id=0x1234)
    assert child_span.parent is not None

    with mock.patch.object(processor._trace_manager, "pop_trace") as mock_pop:
        processor.on_end(child_span)
        mock_pop.assert_not_called()

    assert not any(k.startswith(SpanAttributeKey.TRACE_TAG_PREFIX) for k in child_span._attributes)


def test_on_end_handles_missing_trace_gracefully():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    root_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x1234)

    with mock.patch.object(processor._trace_manager, "pop_trace", return_value=None):
        processor.on_end(root_span)

    assert not any(k.startswith(SpanAttributeKey.TRACE_TAG_PREFIX) for k in root_span._attributes)


def test_on_end_filters_mlflow_prefixed_tags():
    processor, _ = _make_processor()
    otel_trace_id = 0xDEADBEEF

    root_span = create_mock_otel_span(trace_id=otel_trace_id, span_id=0x1234)
    tags = {
        "user_tag": "keep_me",
        "mlflow.traceName": "my-trace",
        "mlflow.trace.spansLocation": "TRACKING_STORE",
        "mlflow.user": "alice",
        "mlflow.artifactLocation": "s3://bucket",
    }
    manager_trace = _make_manager_trace(tags=tags)

    with mock.patch.object(processor._trace_manager, "pop_trace", return_value=manager_trace):
        processor.on_end(root_span)

    assert root_span._attributes.get(SpanAttributeKey.TRACE_TAG_PREFIX + "user_tag") == "keep_me"
    mlflow_tag_attrs = [
        k
        for k in root_span._attributes
        if k.startswith(SpanAttributeKey.TRACE_TAG_PREFIX)
        and k != SpanAttributeKey.TRACE_TAG_PREFIX + "user_tag"
    ]
    assert mlflow_tag_attrs == []


@pytest.mark.parametrize("genai_enabled", [False, True])
@pytest.mark.parametrize("native_otel", [False, True])
@pytest.mark.parametrize("dual_export", [False, True])
@pytest.mark.parametrize(
    "version",
    [None, "1.0.0", "1", "true", "null", "[]", "{}", '"quoted"', "发布🧪", 1, 0, True, False, 1.5],
)
def test_export_preserves_mcp_attributes(
    genai_enabled, native_otel, dual_export, version, monkeypatch, encode_spans
):
    monkeypatch.setenv("MLFLOW_ENABLE_OTEL_GENAI_SEMCONV", str(genai_enabled))
    monkeypatch.setenv("MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT", str(dual_export))
    exporter = mock.Mock(export=mock.Mock(return_value=SpanExportResult.SUCCESS))
    provider = TracerProvider()
    provider.add_span_processor(OtelSpanProcessor(span_exporter=exporter, export_metrics=False))
    try:
        original_scope_attributes = (
            {"custom": "keep"}
            if "attributes" in inspect.signature(provider.get_tracer).parameters
            else {}
        )
        tracer = provider.get_tracer(
            __name__,
            "1.0",
            schema_url="https://example.com/schema",
            **({"attributes": original_scope_attributes} if original_scope_attributes else {}),
        )
        with tracer.start_as_current_span("forecast") as otel_span:
            span = (
                LiveSpan(otel_span, trace_id=f"tr-{otel_span.context.trace_id:032x}")
                if dual_export
                else get_mlflow_span_for_otel_span(otel_span)
            )
            assert span is not None
            span.set_span_type("TOOL")
            if native_otel:
                otel_span.set_attribute(SpanAttributeKey.MCP_SERVER_NAME, "com.example/weather")
                otel_span.set_attribute(SpanAttributeKey.MCP_TOOL_NAME, "forecast")
                if version is not None:
                    otel_span.set_attribute(SpanAttributeKey.MCP_SERVER_VERSION, version)
            else:
                attributes = {
                    SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
                    SpanAttributeKey.MCP_TOOL_NAME: "forecast",
                }
                if version is not None:
                    attributes[SpanAttributeKey.MCP_SERVER_VERSION] = version
                span.set_attributes(attributes)
            original_attributes = dict(otel_span.attributes)
        assert provider.force_flush()
    finally:
        provider.shutdown()

    exporter.export.assert_called_once()
    exporter.shutdown.assert_called_once()
    exported_spans = exporter.export.call_args.args[0]
    assert len(exported_spans) == 1
    assert dict(otel_span.attributes) == original_attributes
    original_scope = otel_span.instrumentation_scope
    exported_scope = exported_spans[0].instrumentation_scope
    assert dict(getattr(original_scope, "attributes", {})) == original_scope_attributes
    assert exported_scope.name == original_scope.name
    assert exported_scope.version == original_scope.version
    assert exported_scope.schema_url == original_scope.schema_url
    assert exported_scope is original_scope
    original_resource_attributes = dict(otel_span.resource.attributes)
    expected_resource_attributes = original_resource_attributes
    assert dict(exported_spans[0].resource.attributes) == expected_resource_attributes
    assert exported_spans[0].attributes[SpanAttributeKey.MCP_SERVER_NAME] == "com.example/weather"
    if version is not None:
        assert exported_spans[0].attributes[SpanAttributeKey.MCP_SERVER_VERSION] == version
    resource_spans = encode_spans(exported_spans).resource_spans[0]
    scope_spans = resource_spans.scope_spans[0]
    resource_attributes = {
        attr.key: _decode_otel_proto_anyvalue(attr.value)
        for attr in resource_spans.resource.attributes
    }
    assert resource_attributes == expected_resource_attributes
    # Older encoders omit scope metadata; preserve the encoder's existing behavior.
    original_scope_spans = encode_spans([otel_span]).resource_spans[0].scope_spans[0]
    assert scope_spans.scope == original_scope_spans.scope
    assert scope_spans.schema_url == original_scope_spans.schema_url
    proto = scope_spans.spans[0]
    assert {attr.key: _decode_otel_proto_anyvalue(attr.value) for attr in proto.attributes} == dict(
        exported_spans[0].attributes
    )
    received = Span.from_otel_proto(proto)
    received._span._attributes = strip_otlp_double_encoding(
        received._span._attributes,
        raw_attribute_keys=MCP_SPAN_ATTRIBUTE_KEYS,
    )
    assert received.get_attribute(SpanAttributeKey.MCP_SERVER_NAME) == "com.example/weather"
    assert received.get_attribute(SpanAttributeKey.MCP_SERVER_VERSION) == version
    assert type(received.get_attribute(SpanAttributeKey.MCP_SERVER_VERSION)) is type(version)
    assert received.get_attribute(SpanAttributeKey.MCP_TOOL_NAME) == "forecast"


@pytest.mark.parametrize("max_attributes", [4, 16])
@pytest.mark.parametrize("max_attribute_length", [None, 32])
@pytest.mark.parametrize(
    "mcp_attributes",
    [
        {},
        {SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather"},
        {SpanAttributeKey.MCP_SERVER_VERSION: "发布🧪" * 3},
        {
            SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
            SpanAttributeKey.MCP_SERVER_VERSION: "发布🧪" * 3,
        },
    ],
)
def test_mcp_export_preserves_dropped_attribute_count(
    max_attributes, max_attribute_length, mcp_attributes, monkeypatch, encode_spans
):
    monkeypatch.setenv("MLFLOW_ENABLE_OTEL_GENAI_SEMCONV", "false")
    monkeypatch.setenv("MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT", "true")
    exporter = mock.Mock(export=mock.Mock(return_value=SpanExportResult.SUCCESS))
    provider = TracerProvider(
        span_limits=SpanLimits(
            max_attributes=max_attributes, max_attribute_length=max_attribute_length
        )
    )
    provider.add_span_processor(OtelSpanProcessor(span_exporter=exporter, export_metrics=False))
    try:
        with provider.get_tracer(__name__).start_as_current_span("bounded") as otel_span:
            for i in range(6):
                otel_span.set_attribute(f"custom.{i}", i)
            otel_span.set_attributes(mcp_attributes)
            original_attributes = dict(otel_span.attributes)
        assert provider.force_flush()
    finally:
        provider.shutdown()

    exporter.export.assert_called_once()
    exporter.shutdown.assert_called_once()
    exported_spans = exporter.export.call_args.args[0]
    assert len(exported_spans) == 1
    expected_dropped = max(0, 6 + len(mcp_attributes) - max_attributes)
    assert otel_span.dropped_attributes == expected_dropped
    assert dict(otel_span.attributes) == original_attributes
    assert exported_spans[0].dropped_attributes == expected_dropped
    proto = encode_spans(exported_spans).resource_spans[0].scope_spans[0].spans[0]
    assert proto.dropped_attributes_count == expected_dropped
    received = Span.from_otel_proto(proto)
    received._span._attributes = strip_otlp_double_encoding(received._span._attributes)
    received_attributes = received.attributes
    assert received_attributes.pop(SpanAttributeKey.REQUEST_ID) == received.trace_id
    assert received_attributes == original_attributes


@pytest.mark.parametrize("has_scope", [False, True])
@pytest.mark.parametrize("dual_export", [False, True])
@pytest.mark.parametrize(
    "mcp_attributes",
    [
        {},
        {SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather"},
        {SpanAttributeKey.MCP_SERVER_VERSION: "[]"},
        {
            SpanAttributeKey.MCP_SERVER_NAME: "com.example/weather",
            SpanAttributeKey.MCP_SERVER_VERSION: '"quoted"',
        },
        {SpanAttributeKey.MCP_TOOL_NAME: '"forecast"'},
    ],
)
def test_mcp_export_without_scope_attributes(has_scope, dual_export, mcp_attributes, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_OTEL_GENAI_SEMCONV", "false")
    monkeypatch.setenv("MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT", str(dual_export))
    resource = Resource({"custom": "keep"}, schema_url="https://example.com/resource")
    exporter = mock.Mock(export=mock.Mock(return_value=SpanExportResult.SUCCESS))
    provider = TracerProvider(resource=resource, span_limits=SpanLimits(max_attributes=4))
    provider.add_span_processor(OtelSpanProcessor(span_exporter=exporter, export_metrics=False))
    scope = (
        SimpleNamespace(name="legacy", version="1.0", schema_url="https://example.com/scope")
        if has_scope
        else None
    )
    try:
        with provider.get_tracer(__name__).start_as_current_span("mcp") as span:
            span._instrumentation_scope = scope
            for index in range(6):
                span.set_attribute(f"custom.{index}", index)
            span.set_attributes(mcp_attributes)
            original_attributes = dict(span.attributes)
        assert provider.force_flush()
    finally:
        provider.shutdown()
    exporter.export.assert_called_once()
    exporter.shutdown.assert_called_once()
    exported = exporter.export.call_args.args[0][0]
    assert dict(span.attributes) == original_attributes
    assert dict(exported.attributes) == original_attributes
    assert exported.dropped_attributes == span.dropped_attributes
    assert exported.instrumentation_scope is scope
    assert exported.resource.schema_url == resource.schema_url
    expected_resource_attributes = {"custom": "keep"}
    assert dict(exported.resource.attributes) == expected_resource_attributes
    assert dict(resource.attributes) == {"custom": "keep"}
