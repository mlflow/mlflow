"""
GenAI Schema Span Exporter Wrapper.

This module provides a wrapper around any SpanExporter that converts
span attributes to OpenTelemetry GenAI semantic conventions before export.
"""

import logging
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from mlflow.tracing.genai_schema import convert_to_genai_schema

_logger = logging.getLogger(__name__)


class GenAISchemaSpanExporter(SpanExporter):
    """
    A wrapper SpanExporter that converts span attributes to OpenTelemetry
    GenAI semantic conventions before delegating to the wrapped exporter.

    This exporter enables interoperability with OTEL-compatible observability
    tools that expect GenAI semantic convention attribute keys.

    Example:
        >>> from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        >>> base_exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
        >>> genai_exporter = GenAISchemaSpanExporter(base_exporter)
    """

    def __init__(self, wrapped_exporter: SpanExporter) -> None:
        """
        Initialize the GenAI schema exporter wrapper.

        Args:
            wrapped_exporter: The underlying SpanExporter to delegate to
                after attribute conversion.
        """
        self._wrapped_exporter = wrapped_exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Export spans after converting attributes to GenAI schema.

        Args:
            spans: Sequence of ReadableSpan objects to export.

        Returns:
            SpanExportResult indicating success or failure.
        """
        converted_spans = []
        for span in spans:
            try:
                converted_span = self._convert_span_attributes(span)
                converted_spans.append(converted_span)
            except Exception as e:
                _logger.debug(f"Failed to convert span attributes to GenAI schema: {e}")
                # Fall back to original span if conversion fails
                converted_spans.append(span)

        return self._wrapped_exporter.export(converted_spans)

    def _convert_span_attributes(self, span: ReadableSpan) -> ReadableSpan:
        """
        Convert a span's attributes to GenAI semantic conventions.

        Creates a new ReadableSpan with converted attributes while
        preserving all other span properties.

        Args:
            span: The original ReadableSpan.

        Returns:
            A new ReadableSpan with converted attributes.
        """
        # Get the original attributes as a dict
        original_attrs = dict(span.attributes) if span.attributes else {}

        # Convert to GenAI schema
        converted_attrs = convert_to_genai_schema(original_attrs)

        # Create a new span with converted attributes
        # We use the _ReadableSpan internal class to create a modified copy
        return _ReadableSpanWithAttributes(span, converted_attrs)

    def shutdown(self) -> None:
        """Shutdown the wrapped exporter."""
        self._wrapped_exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the wrapped exporter."""
        return self._wrapped_exporter.force_flush(timeout_millis)


class _ReadableSpanWithAttributes(ReadableSpan):
    """
    A ReadableSpan wrapper that overrides the attributes property.

    This class wraps an existing ReadableSpan and substitutes its
    attributes with converted GenAI schema attributes while delegating
    all other properties to the original span.
    """

    def __init__(self, original_span: ReadableSpan, converted_attributes: dict) -> None:
        """
        Initialize the wrapper.

        Args:
            original_span: The original ReadableSpan to wrap.
            converted_attributes: The converted attributes dict.
        """
        self._original_span = original_span
        self._converted_attributes = converted_attributes
        # Required by OTLP exporter which accesses _attributes directly
        self._attributes = converted_attributes

    @property
    def name(self):
        return self._original_span.name

    @property
    def context(self):
        return self._original_span.context

    @property
    def kind(self):
        return self._original_span.kind

    @property
    def parent(self):
        return self._original_span.parent

    @property
    def start_time(self):
        return self._original_span.start_time

    @property
    def end_time(self):
        return self._original_span.end_time

    @property
    def status(self):
        return self._original_span.status

    @property
    def attributes(self):
        return self._converted_attributes

    @property
    def events(self):
        return self._original_span.events

    @property
    def links(self):
        return self._original_span.links

    @property
    def resource(self):
        return self._original_span.resource

    @property
    def instrumentation_scope(self):
        return self._original_span.instrumentation_scope

    def get_span_context(self):
        return self._original_span.get_span_context()

    def is_recording(self):
        return False  # ReadableSpan is not recording

    @property
    def dropped_attributes(self):
        """Return dropped attributes count from original span."""
        if hasattr(self._original_span, "dropped_attributes"):
            return self._original_span.dropped_attributes
        return 0

    @property
    def dropped_events(self):
        """Return dropped events count from original span."""
        if hasattr(self._original_span, "dropped_events"):
            return self._original_span.dropped_events
        return 0

    @property
    def dropped_links(self):
        """Return dropped links count from original span."""
        if hasattr(self._original_span, "dropped_links"):
            return self._original_span.dropped_links
        return 0

    def to_json(self, indent=4):
        """Return a JSON representation of the span."""
        if hasattr(self._original_span, "to_json"):
            return self._original_span.to_json(indent)
        return None
