"""
Mixin class for OpenTelemetry span processors that provides metrics recording functionality.

This mixin allows different span processor implementations to share common metrics logic
while maintaining their own inheritance hierarchies (BatchSpanProcessor, SimpleSpanProcessor).
"""

import json
import logging
from typing import Any

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities.span import SpanType
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import get_experiment_id_for_trace
from mlflow.tracing.utils.otlp import _get_otlp_metrics_endpoint, _get_otlp_metrics_protocol

_logger = logging.getLogger(__name__)


class OtelMetricsMixin:
    """
    Mixin class that provides metrics recording capabilities for span processors.

    This mixin is designed to be used with OpenTelemetry span processors to record
    span-related metrics (e.g. duration) and metadata.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the mixin and pass through to parent classes."""
        super().__init__(*args, **kwargs)
        self._duration_histogram = None
        self._trace_manager = InMemoryTraceManager.get_instance()

    def _setup_metrics_if_necessary(self) -> None:
        """
        Set up OpenTelemetry metrics if not already configured previously.
        """
        if self._duration_histogram is not None:
            return

        endpoint = _get_otlp_metrics_endpoint()
        if not endpoint:
            return

        protocol = _get_otlp_metrics_protocol()
        if protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
        elif protocol == "http/protobuf":
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
        else:
            _logger.warning(
                f"Unsupported OTLP metrics protocol '{protocol}'. "
                "Supported protocols are 'grpc' and 'http/protobuf'. "
                "Metrics export will be skipped."
            )
            return

        metric_exporter = OTLPMetricExporter(endpoint=endpoint)
        reader = PeriodicExportingMetricReader(metric_exporter)
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)
        meter = metrics.get_meter("mlflow.tracing")
        self._duration_histogram = meter.create_histogram(
            name="mlflow.trace.span.duration",
            description="Duration of spans in milliseconds",
            unit="ms",
        )

    def record_metrics_for_span(self, span: OTelReadableSpan) -> None:
        """
        Record metrics for a completed span.

        This method should be called at the beginning of the on_end() method
        to record span duration and associated metadata.

        Args:
            span: The completed OpenTelemetry span to record metrics for.
        """
        self._setup_metrics_if_necessary()

        if self._duration_histogram is None:
            return

        span_type = span.attributes.get(SpanAttributeKey.SPAN_TYPE, SpanType.UNKNOWN)
        try:
            # Span attributes are JSON encoded by default; decode them for metric label readability
            span_type = json.loads(span_type)
        except (json.JSONDecodeError, TypeError):
            pass

        attributes = {
            "root": span.parent is None,
            "span_type": span_type,
            "span_status": span.status.status_code.name if span.status else "UNSET",
            "experiment_id": get_experiment_id_for_trace(span),
        }

        # Add trace tags and metadata if trace is available
        # Get MLflow trace ID from OpenTelemetry trace ID
        mlflow_trace_id = self._trace_manager.get_mlflow_trace_id_from_otel_id(
            span.context.trace_id
        )
        if mlflow_trace_id is not None:
            with self._trace_manager.get_trace(mlflow_trace_id) as trace:
                if trace is not None:
                    for key, value in trace.info.tags.items():
                        attributes[f"tags.{key}"] = str(value)
                    if trace.info.trace_metadata:
                        for meta_key, meta_value in trace.info.trace_metadata.items():
                            attributes[f"metadata.{meta_key}"] = str(meta_value)

        self._duration_histogram.record(
            amount=(span.end_time - span.start_time) / 1e6, attributes=attributes
        )
