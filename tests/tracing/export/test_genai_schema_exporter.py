"""
Tests for the GenAI schema span exporter wrapper.

This module tests the GenAISchemaSpanExporter which converts
span attributes to OpenTelemetry GenAI semantic conventions.
"""

from unittest import mock

import pytest

from mlflow.tracing.export.genai_schema_exporter import (
    GenAISchemaSpanExporter,
    _ReadableSpanWithAttributes,
)
from mlflow.tracing.utils.otlp import (
    SCHEMA_DEFAULT,
    SCHEMA_GENAI,
    SUPPORTED_SCHEMAS,
    _get_otlp_traces_schema,
)

# OTLP exporters are not installed in some CI jobs
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GrpcExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HttpExporter,
    )

    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class MockSpanExporter(SpanExporter):
    """Mock exporter for testing."""

    def __init__(self):
        self.exported_spans = []
        self.shutdown_called = False
        self.force_flush_called = False

    def export(self, spans):
        self.exported_spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.shutdown_called = True

    def force_flush(self, timeout_millis=30000):
        self.force_flush_called = True
        return True


class MockReadableSpan:
    """Mock ReadableSpan for testing."""

    def __init__(self, attributes=None, name="test-span"):
        self._attributes = attributes or {}
        self._name = name
        self._context = mock.MagicMock()
        self._kind = mock.MagicMock()
        self._parent = None
        self._start_time = 1000000000
        self._end_time = 2000000000
        self._status = mock.MagicMock()
        self._events = []
        self._links = []
        self._resource = mock.MagicMock()
        self._instrumentation_scope = mock.MagicMock()
        self.dropped_attributes = 0
        self.dropped_events = 0
        self.dropped_links = 0

    @property
    def name(self):
        return self._name

    @property
    def context(self):
        return self._context

    @property
    def kind(self):
        return self._kind

    @property
    def parent(self):
        return self._parent

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def status(self):
        return self._status

    @property
    def attributes(self):
        return self._attributes

    @property
    def events(self):
        return self._events

    @property
    def links(self):
        return self._links

    @property
    def resource(self):
        return self._resource

    @property
    def instrumentation_scope(self):
        return self._instrumentation_scope

    def get_span_context(self):
        return self._context

    def is_recording(self):
        return False


class TestGenAISchemaSpanExporter:
    """Tests for GenAISchemaSpanExporter."""

    def test_export_converts_attributes(self):
        """Test that export converts MLflow attributes to GenAI schema."""
        mock_exporter = MockSpanExporter()
        genai_exporter = GenAISchemaSpanExporter(mock_exporter)

        span = MockReadableSpan(
            attributes={
                "mlflow.spanInputs": "Hello, world!",
                "mlflow.spanOutputs": "Hi there!",
                "mlflow.spanType": "LLM",
            }
        )

        result = genai_exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_exporter.exported_spans) == 1

        exported_span = mock_exporter.exported_spans[0]
        attrs = exported_span.attributes

        assert attrs["gen_ai.request.input"] == "Hello, world!"
        assert attrs["gen_ai.response.output"] == "Hi there!"
        assert attrs["gen_ai.operation.name"] == "text_completion"

    def test_export_preserves_unmapped_attributes(self):
        """Test that unmapped attributes are preserved."""
        mock_exporter = MockSpanExporter()
        genai_exporter = GenAISchemaSpanExporter(mock_exporter)

        span = MockReadableSpan(
            attributes={
                "mlflow.spanInputs": "input",
                "custom.attribute": "preserved",
            }
        )

        genai_exporter.export([span])

        exported_span = mock_exporter.exported_spans[0]
        attrs = exported_span.attributes

        assert attrs["gen_ai.request.input"] == "input"
        assert attrs["custom.attribute"] == "preserved"

    def test_export_converts_token_usage(self):
        """Test that token usage is converted correctly."""
        mock_exporter = MockSpanExporter()
        genai_exporter = GenAISchemaSpanExporter(mock_exporter)

        span = MockReadableSpan(
            attributes={
                "mlflow.chat.tokenUsage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                }
            }
        )

        genai_exporter.export([span])

        exported_span = mock_exporter.exported_spans[0]
        attrs = exported_span.attributes

        assert attrs["gen_ai.usage.input_tokens"] == 100
        assert attrs["gen_ai.usage.output_tokens"] == 50
        assert attrs["gen_ai.usage.total_tokens"] == 150

    def test_export_handles_empty_attributes(self):
        """Test export with empty attributes."""
        mock_exporter = MockSpanExporter()
        genai_exporter = GenAISchemaSpanExporter(mock_exporter)

        span = MockReadableSpan(attributes={})

        result = genai_exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_exporter.exported_spans) == 1

    def test_export_handles_none_attributes(self):
        """Test export with None attributes."""
        mock_exporter = MockSpanExporter()
        genai_exporter = GenAISchemaSpanExporter(mock_exporter)

        span = MockReadableSpan(attributes=None)

        result = genai_exporter.export([span])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_exporter.exported_spans) == 1

    def test_export_multiple_spans(self):
        """Test exporting multiple spans."""
        mock_exporter = MockSpanExporter()
        genai_exporter = GenAISchemaSpanExporter(mock_exporter)

        spans = [
            MockReadableSpan(attributes={"mlflow.spanInputs": "input1"}),
            MockReadableSpan(attributes={"mlflow.spanInputs": "input2"}),
            MockReadableSpan(attributes={"mlflow.spanInputs": "input3"}),
        ]

        result = genai_exporter.export(spans)

        assert result == SpanExportResult.SUCCESS
        assert len(mock_exporter.exported_spans) == 3

        for i, exported_span in enumerate(mock_exporter.exported_spans):
            assert exported_span.attributes["gen_ai.request.input"] == f"input{i + 1}"

    def test_shutdown_delegates_to_wrapped_exporter(self):
        """Test that shutdown is delegated."""
        mock_exporter = MockSpanExporter()
        genai_exporter = GenAISchemaSpanExporter(mock_exporter)

        genai_exporter.shutdown()

        assert mock_exporter.shutdown_called

    def test_force_flush_delegates_to_wrapped_exporter(self):
        """Test that force_flush is delegated."""
        mock_exporter = MockSpanExporter()
        genai_exporter = GenAISchemaSpanExporter(mock_exporter)

        result = genai_exporter.force_flush(timeout_millis=5000)

        assert result is True
        assert mock_exporter.force_flush_called

    def test_export_handles_conversion_error_gracefully(self):
        """Test that export falls back to original span if conversion fails."""
        mock_exporter = MockSpanExporter()
        genai_exporter = GenAISchemaSpanExporter(mock_exporter)

        # Create a span with attributes that might cause conversion issues
        # Mock the attributes property to raise an exception
        problematic_span = MockReadableSpan(attributes={"valid": "attr"})

        # Mock convert_to_genai_schema to raise an exception for this test
        with mock.patch(
            "mlflow.tracing.export.genai_schema_exporter.convert_to_genai_schema",
            side_effect=Exception("Conversion failed"),
        ):
            result = genai_exporter.export([problematic_span])

            # Should still succeed, but use original span
            assert result == SpanExportResult.SUCCESS
            assert len(mock_exporter.exported_spans) == 1
            # The span should be the original (not converted)
            assert mock_exporter.exported_spans[0] == problematic_span

    def test_export_handles_wrapped_exporter_failure(self):
        """Test that export propagates wrapped exporter failures."""
        mock_exporter = MockSpanExporter()

        # Make the wrapped exporter fail
        def failing_export(spans):
            return SpanExportResult.FAILURE

        mock_exporter.export = failing_export

        genai_exporter = GenAISchemaSpanExporter(mock_exporter)
        span = MockReadableSpan(attributes={"mlflow.spanInputs": "test"})

        result = genai_exporter.export([span])

        assert result == SpanExportResult.FAILURE

    def test_export_handles_empty_spans_list(self):
        """Test that export handles empty spans list."""
        mock_exporter = MockSpanExporter()
        genai_exporter = GenAISchemaSpanExporter(mock_exporter)

        result = genai_exporter.export([])

        assert result == SpanExportResult.SUCCESS
        assert len(mock_exporter.exported_spans) == 0


class TestReadableSpanWithAttributes:
    """Tests for _ReadableSpanWithAttributes wrapper."""

    def test_attributes_overridden(self):
        """Test that attributes are overridden."""
        original = MockReadableSpan(attributes={"old": "value"})
        converted = {"new": "value"}

        wrapped = _ReadableSpanWithAttributes(original, converted)

        assert wrapped.attributes == converted
        assert wrapped.attributes["new"] == "value"

    def test_other_properties_delegated(self):
        """Test that other properties are delegated to original span."""
        original = MockReadableSpan(name="test-span")
        converted = {"key": "value"}

        wrapped = _ReadableSpanWithAttributes(original, converted)

        assert wrapped.name == "test-span"
        assert wrapped.context == original.context
        assert wrapped.kind == original.kind
        assert wrapped.parent == original.parent
        assert wrapped.start_time == original.start_time
        assert wrapped.end_time == original.end_time
        assert wrapped.status == original.status
        assert wrapped.events == original.events
        assert wrapped.links == original.links
        assert wrapped.resource == original.resource
        assert wrapped.instrumentation_scope == original.instrumentation_scope

    def test_is_recording_returns_false(self):
        """Test that is_recording returns False for ReadableSpan."""
        original = MockReadableSpan()
        wrapped = _ReadableSpanWithAttributes(original, {})

        assert wrapped.is_recording() is False

    def test_dropped_attributes_preserved(self):
        """Test that dropped_attributes count is preserved."""
        original = MockReadableSpan()
        original.dropped_attributes = 5
        wrapped = _ReadableSpanWithAttributes(original, {})

        assert wrapped.dropped_attributes == 5

    def test_dropped_events_preserved(self):
        """Test that dropped_events count is preserved."""
        original = MockReadableSpan()
        original.dropped_events = 3
        wrapped = _ReadableSpanWithAttributes(original, {})

        assert wrapped.dropped_events == 3

    def test_dropped_links_preserved(self):
        """Test that dropped_links count is preserved."""
        original = MockReadableSpan()
        original.dropped_links = 2
        wrapped = _ReadableSpanWithAttributes(original, {})

        assert wrapped.dropped_links == 2

    def test_dropped_properties_default_to_zero(self):
        """Test that dropped properties default to 0 if not set."""
        original = MockReadableSpan()
        # Don't set dropped_* properties
        wrapped = _ReadableSpanWithAttributes(original, {})

        assert wrapped.dropped_attributes == 0
        assert wrapped.dropped_events == 0
        assert wrapped.dropped_links == 0


class TestGetOtlpTracesSchema:
    """Tests for _get_otlp_traces_schema function."""

    def test_default_schema(self, monkeypatch):
        """Test default schema when env var not set."""
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_SCHEMA", raising=False)

        schema = _get_otlp_traces_schema()

        assert schema == SCHEMA_DEFAULT

    def test_genai_schema(self, monkeypatch):
        """Test genai schema when env var set."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_SCHEMA", "genai")

        schema = _get_otlp_traces_schema()

        assert schema == SCHEMA_GENAI

    def test_genai_schema_uppercase(self, monkeypatch):
        """Test genai schema with uppercase value."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_SCHEMA", "GENAI")

        schema = _get_otlp_traces_schema()

        assert schema == SCHEMA_GENAI

    def test_default_schema_explicit(self, monkeypatch):
        """Test explicit default schema."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_SCHEMA", "default")

        schema = _get_otlp_traces_schema()

        assert schema == SCHEMA_DEFAULT

    def test_invalid_schema_falls_back_to_default(self, monkeypatch):
        """Test invalid schema falls back to default."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_SCHEMA", "invalid_schema")

        schema = _get_otlp_traces_schema()

        assert schema == SCHEMA_DEFAULT

    def test_supported_schemas_contains_expected_values(self):
        """Test that SUPPORTED_SCHEMAS contains expected values."""
        assert SCHEMA_DEFAULT in SUPPORTED_SCHEMAS
        assert SCHEMA_GENAI in SUPPORTED_SCHEMAS


@pytest.mark.skipif(not OTLP_AVAILABLE, reason="OTLP exporters not installed")
class TestGetOtlpExporterWithSchema:
    """Integration tests for get_otlp_exporter with schema configuration."""

    def test_default_schema_returns_unwrapped_exporter(self, monkeypatch):
        """Test that default schema returns unwrapped OTLP exporter."""
        from mlflow.tracing.utils.otlp import get_otlp_exporter

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4317")
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_SCHEMA", raising=False)

        exporter = get_otlp_exporter()

        assert isinstance(exporter, GrpcExporter)

    def test_genai_schema_returns_wrapped_exporter(self, monkeypatch):
        """Test that genai schema returns wrapped OTLP exporter."""
        from mlflow.tracing.utils.otlp import get_otlp_exporter

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_SCHEMA", "genai")

        exporter = get_otlp_exporter()

        assert isinstance(exporter, GenAISchemaSpanExporter)
        assert isinstance(exporter._wrapped_exporter, GrpcExporter)

    def test_genai_schema_with_http_protocol(self, monkeypatch):
        """Test genai schema with HTTP protocol."""
        from mlflow.tracing.utils.otlp import get_otlp_exporter

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_SCHEMA", "genai")

        exporter = get_otlp_exporter()

        assert isinstance(exporter, GenAISchemaSpanExporter)
        assert isinstance(exporter._wrapped_exporter, HttpExporter)
