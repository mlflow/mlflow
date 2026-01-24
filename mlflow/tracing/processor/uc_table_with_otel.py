import logging

from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from mlflow.entities.span import create_mlflow_span
from mlflow.entities.telemetry_profile import TelemetryProfile
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION_KEY
from mlflow.tracing.processor.otel_metrics_mixin import OtelMetricsMixin
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import generate_trace_id_v4
from mlflow.tracing.utils.environment import resolve_env_metadata

_logger = logging.getLogger(__name__)


class DatabricksUCTableWithOtelSpanProcessor(OtelMetricsMixin, BatchSpanProcessor):
    """
    Span processor for exporting spans via OTLP while persisting TraceInfo to UC tables.

    This processor extends BatchSpanProcessor (like OtelSpanProcessor) for proper
    span batching and OTLP export, while also handling TraceInfo registration and
    persistence via the backend REST API.

    The processor creates TraceInfo with UcTablePrefixLocation derived from
    the TelemetryProfile configuration.
    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        telemetry_profile: TelemetryProfile,
        export_metrics: bool = False,
    ) -> None:
        super().__init__(span_exporter)
        self._telemetry_profile = telemetry_profile
        self._export_metrics = export_metrics
        self._trace_manager = InMemoryTraceManager.get_instance()
        self._env_metadata = resolve_env_metadata()

        # Compatibility with different opentelemetry-sdk versions
        # See OtelSpanProcessor for details
        try:
            self.span_exporter = span_exporter
        except AttributeError:
            pass

    def on_start(self, span: OTelSpan, parent_context=None):
        """Handle span start - register trace and span with trace manager."""
        if not span.parent:
            # Root span - create and register TraceInfo
            trace_info = self._create_trace_info(span)
            trace_id = trace_info.trace_id
            self._trace_manager.register_trace(span.context.trace_id, trace_info)
        else:
            # Child span - get trace_id from trace manager
            trace_id = self._trace_manager.get_mlflow_trace_id_from_otel_id(span.context.trace_id)

        if trace_id:
            self._trace_manager.register_span(create_mlflow_span(span, trace_id))

        super().on_start(span, parent_context)

    def on_end(self, span: OTelReadableSpan):
        """Handle span end - record metrics and export via BatchSpanProcessor."""
        if self._export_metrics:
            self.record_metrics_for_span(span)

        # Note: We don't pop the trace here like OtelSpanProcessor does.
        # The exporter will handle TraceInfo persistence and trace cleanup
        # when processing root spans in _export_traces().

        super().on_end(span)

    def _create_trace_info(self, root_span: OTelSpan) -> TraceInfo:
        """
        Create a TraceInfo with UcTablePrefixLocation from TelemetryProfile.
        """
        uc_tables_config = self._telemetry_profile.get_uc_tables_config()
        if not uc_tables_config:
            raise MlflowException(
                "TelemetryProfile does not contain a UnityCatalogTablesConfig. "
                "Cannot export traces without UC table configuration."
            )

        catalog_name = uc_tables_config.uc_catalog
        schema_name = uc_tables_config.uc_schema
        table_prefix = uc_tables_config.uc_table_prefix or ""

        if not catalog_name or not schema_name:
            raise MlflowException(
                "TelemetryProfile UnityCatalogTablesConfig is missing uc_catalog or uc_schema. "
                "Cannot export traces without complete UC configuration."
            )

        # Create TraceLocation with UcTablePrefixLocation
        trace_location = TraceLocation.from_databricks_uc_table_prefix(
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_prefix=table_prefix,
        )

        # Generate trace ID using the schema location (catalog.schema)
        location_id = f"{catalog_name}.{schema_name}"
        trace_id = generate_trace_id_v4(root_span, location_id)

        metadata = self._env_metadata.copy()
        # Override the schema version to 4 for UC table prefix
        metadata[TRACE_SCHEMA_VERSION_KEY] = "4"

        return TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location,
            request_time=root_span.start_time // 1_000_000,  # nanosecond to millisecond
            execution_duration=None,
            state=TraceState.IN_PROGRESS,
            trace_metadata=metadata,
            tags={},
        )
