import logging

from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION_KEY, SpanAttributeKey, TraceMetadataKey
from mlflow.tracing.processor.base_mlflow import BaseMlflowSpanProcessor
from mlflow.tracing.utils import (
    _bypass_attribute_guard,
    generate_trace_id_v4,
    get_active_spans_table_name,
    get_mlflow_span_for_otel_span,
)

_logger = logging.getLogger(__name__)


class DatabricksUCTableSpanProcessor(BaseMlflowSpanProcessor):
    """
    Defines custom hooks to be executed when a span is started or ended (before exporting).

    This processor is used for exporting traces to Databricks Unity Catalog table.
    """

    def __init__(self, span_exporter: SpanExporter):
        # metrics export is not supported for UC table yet
        super().__init__(span_exporter, export_metrics=False)

    def _start_trace(self, root_span: OTelSpan) -> TraceInfo:
        """
        Create a new TraceInfo object and register it with the trace manager.

        This method is called in the on_start method of the base class.
        """
        if uc_spans_table_name := get_active_spans_table_name():
            catalog_name, schema_name, spans_table_name = uc_spans_table_name.split(".")
            trace_location = TraceLocation.from_databricks_uc_schema(catalog_name, schema_name)
            trace_location.uc_schema._otel_spans_table_name = spans_table_name
            trace_id = generate_trace_id_v4(root_span, trace_location.uc_schema.schema_location)
        else:
            raise MlflowException(
                "Unity Catalog spans table name is not set for trace. It can not be exported to "
                "Databricks Unity Catalog table."
            )

        metadata = self._get_basic_trace_metadata()
        # Override the schema version to 4 for UC table
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

    def on_end(self, span: OTelReadableSpan) -> None:
        if span._parent is None:
            self._set_user_session_span_attributes(span)
        super().on_end(span)

    def _set_user_session_span_attributes(self, root_span: OTelReadableSpan) -> None:
        mlflow_span = get_mlflow_span_for_otel_span(root_span)
        if mlflow_span is None:
            return
        with self._trace_manager.get_trace(mlflow_span.trace_id) as trace:
            if trace is None:
                return
            for meta_key, attr_key in (
                (TraceMetadataKey.TRACE_USER, SpanAttributeKey.USER_ID),
                (TraceMetadataKey.TRACE_SESSION, SpanAttributeKey.SESSION_ID),
            ):
                if value := trace.info.trace_metadata.get(meta_key):
                    with _bypass_attribute_guard(mlflow_span._span):
                        mlflow_span._span.set_attribute(attr_key, value)
