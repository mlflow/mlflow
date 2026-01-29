import logging

from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation, UcTablePrefixLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION_KEY
from mlflow.tracing.processor.base_mlflow import BaseMlflowSpanProcessor
from mlflow.tracing.utils import generate_trace_id_v4

_logger = logging.getLogger(__name__)


class DatabricksUCTableFromStorageLocationSpanProcessor(BaseMlflowSpanProcessor):
    """
    Span processor that uses REST API for span export based on UcTablePrefixLocation.

    Unlike DatabricksUCTableWithOtelSpanProcessor which uses OTLP for span export,
    this processor uses REST API for both spans and TraceInfo persistence.

    The UC location (catalog, schema, table_prefix) is provided via UcTablePrefixLocation.
    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        uc_location: UcTablePrefixLocation,
    ) -> None:
        # metrics export is not supported for UC table yet
        super().__init__(span_exporter, export_metrics=False)
        self._uc_location = uc_location

    def _start_trace(self, root_span: OTelSpan) -> TraceInfo:
        """
        Create a TraceInfo with UcTablePrefixLocation.
        """
        catalog_name = self._uc_location.catalog_name
        schema_name = self._uc_location.schema_name
        table_prefix = self._uc_location.table_prefix or ""

        if not catalog_name or not schema_name:
            raise MlflowException(
                "UcTablePrefixLocation is missing catalog_name or schema_name. "
                "Cannot export traces without complete UC configuration."
            )

        # Create TraceLocation with UcTablePrefixLocation
        trace_location = TraceLocation.from_databricks_uc_table_prefix(
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_prefix=table_prefix,
            spans_table_name=self._uc_location.spans_table_name,
            logs_table_name=self._uc_location.logs_table_name,
            metrics_table_name=self._uc_location.metrics_table_name,
        )

        # Generate trace ID using the schema location (catalog.schema)
        location_id = f"{catalog_name}.{schema_name}"
        trace_id = generate_trace_id_v4(root_span, location_id)

        metadata = self._get_basic_trace_metadata()
        # Override the schema version to 4 for UC table prefix
        metadata[TRACE_SCHEMA_VERSION_KEY] = "4"

        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=trace_location,
            request_time=root_span.start_time // 1_000_000,  # nanosecond to millisecond
            execution_duration=None,
            state=TraceState.IN_PROGRESS,
            trace_metadata=metadata,
            tags=self._get_basic_trace_tags(root_span),
        )
        self._trace_manager.register_trace(root_span.context.trace_id, trace_info)

        return trace_info
