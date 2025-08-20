import json
import os

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import StatusCode

from mlflow.entities.trace_info import TraceInfo, TraceLocation, TraceState
from mlflow.environment_variables import (
    MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT,
)
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY, SpanAttributeKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import (
    generate_trace_id_v3,
    get_experiment_id_for_trace,
    get_otel_attribute,
)


class OtelSpanProcessor(BatchSpanProcessor):
    """
    SpanProcessor implementation to export MLflow traces to a OpenTelemetry collector.

    Extending OpenTelemetry BatchSpanProcessor to add some custom hooks to be executed when a span
    is started or ended (before exporting).
    """

    def __init__(self, span_exporter: SpanExporter, export_spans: bool, export_metrics: bool):
        """
        Initialize the OtelSpanProcessor.

        Args:
            span_exporter: The OpenTelemetry span exporter to use for span export.
            export_spans: Whether to export spans to the OTLP collector.
            export_metrics: Whether to export metrics to the OTLP collector. When True,
                metrics setup will be initialized regardless of export_spans value.
        """
        super().__init__(span_exporter)

        self._export_spans = export_spans
        self._export_metrics = export_metrics

        # In opentelemetry-sdk 1.34.0, the `span_exporter` field was removed from the
        # `BatchSpanProcessor` class.
        # https://github.com/open-telemetry/opentelemetry-python/issues/4616
        #
        # The `span_exporter` field was restored as a property in 1.34.1
        # https://github.com/open-telemetry/opentelemetry-python/pull/4621
        #
        # We use a try-except block to maintain compatibility with both versions.
        try:
            self.span_exporter = span_exporter
        except AttributeError:
            pass

        # Only register traces when NOT in dual export mode AND spans are exported
        # In dual export mode, MLflow span processors handle trace registration
        self._should_register_traces = (
            self._export_spans and not MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT.get()
        )
        if self._should_register_traces:
            self._trace_manager = InMemoryTraceManager.get_instance()

        # Initialize metrics if enabled
        self._duration_histogram = None
        if self._export_metrics:
            self._duration_histogram = self._setup_metrics()

    def on_start(self, span: OTelReadableSpan, parent_context=None):
        if self._should_register_traces and not span.parent:
            trace_info = self._create_trace_info(span)
            span.set_attribute(SpanAttributeKey.REQUEST_ID, json.dumps(trace_info.trace_id))
            self._trace_manager.register_trace(trace_info.trace_id, trace_info)

        # Only call parent on_start if we're exporting spans
        if self._export_spans:
            super().on_start(span, parent_context)

    def on_end(self, span: OTelReadableSpan):
        # Handle metrics logging if enabled
        if self._duration_histogram:
            # Calculate duration in milliseconds
            duration_ms = (span.end_time - span.start_time) / 1e6

            # Determine if this is a root span
            is_root = span.parent is None

            # Get span type from attributes if available
            span_type = (
                span.attributes.get("mlflow.spanType", "unknown") if span.attributes else "unknown"
            )

            # Get span status
            status = "UNSET"
            if span.status and span.status.status_code == StatusCode.OK:
                status = "OK"
            elif span.status and span.status.status_code == StatusCode.ERROR:
                status = "ERROR"

            # Get experiment ID for the span
            experiment_id = get_experiment_id_for_trace(span)

            # Start with basic attributes
            attributes = {
                "root": str(is_root),
                "span_type": span_type,
                "span_status": status,
                "experiment_id": experiment_id,
            }

            # Add trace tags and metadata if trace is available
            if is_root and self._should_register_traces:
                trace_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)
                if trace_id:
                    with self._trace_manager.get_trace(trace_id) as trace:
                        if trace:
                            # Add trace tags as attributes (prefixed with 'tag_')
                            for key, value in trace.info.tags.items():
                                # Sanitize key names for metric labels
                                safe_key = f"tag_{key.replace('.', '_').replace('-', '_')}"
                                attributes[safe_key] = str(value)

                            # Add select trace metadata (prefixed with 'meta_')
                            if trace.info.trace_metadata:
                                # Only include specific metadata to avoid overwhelming the metrics
                                for meta_key in ["source_run", "model_id"]:
                                    if meta_key in trace.info.trace_metadata:
                                        safe_key = f"meta_{meta_key}"
                                        attributes[safe_key] = str(
                                            trace.info.trace_metadata[meta_key]
                                        )

            # Record the histogram metric with all attributes
            self._duration_histogram.record(duration_ms, attributes=attributes)

        # Handle trace registration cleanup
        if self._should_register_traces and not span.parent:
            self._trace_manager.pop_trace(span.context.trace_id)

        # Only call parent on_end if we're exporting spans
        if self._export_spans:
            super().on_end(span)

    def _create_trace_info(self, span: OTelReadableSpan) -> TraceInfo:
        """Create a TraceInfo object from an OpenTelemetry span."""
        experiment_id = get_experiment_id_for_trace(span)
        return TraceInfo(
            trace_id=generate_trace_id_v3(span),
            trace_location=TraceLocation.from_experiment_id(experiment_id),
            request_time=span.start_time // 1_000_000,  # nanosecond to millisecond
            execution_duration=None,
            state=TraceState.IN_PROGRESS,
            trace_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
            tags={},
        )

    def _setup_metrics(self):
        """Set up OpenTelemetry metrics and return histogram, or None if setup fails."""
        # Get OTLP endpoint and protocol
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT") or os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT"
        )
        protocol = os.environ.get("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL") or os.environ.get(
            "OTEL_EXPORTER_OTLP_PROTOCOL", "grpc"
        )

        if not endpoint:
            return None

        # Get appropriate metric exporter based on protocol
        # Valid protocols per OpenTelemetry spec: 'grpc' and 'http/protobuf'
        if protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
        elif protocol == "http/protobuf":
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter,
            )
        else:
            raise MlflowException.invalid_parameter_value(
                f"Unsupported OTLP metrics protocol '{protocol}'. "
                "Supported protocols are 'grpc' and 'http/protobuf'."
            )

        metric_exporter = OTLPMetricExporter(endpoint=endpoint)

        # Set up metrics provider
        reader = PeriodicExportingMetricReader(metric_exporter)
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)

        # Create and return histogram
        meter = metrics.get_meter("mlflow.tracing")
        return meter.create_histogram(
            name="mlflow.trace.span.duration",
            description="Duration of spans in milliseconds",
            unit="ms",
        )
