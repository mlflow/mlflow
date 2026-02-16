import logging
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan

from mlflow.entities.span import Span
from mlflow.entities.trace_info import TraceInfo
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_TRACE_LOGGING
from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter
from mlflow.tracing.export.span_batcher import SpanBatcher
from mlflow.tracing.utils import get_active_spans_table_name

_logger = logging.getLogger(__name__)


class DatabricksUCTableSpanExporter(MlflowV3SpanExporter):
    """
    An exporter implementation that logs the traces to Databricks Unity Catalog table.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        super().__init__(tracking_uri)

        # Track if we've raised an error for span export to avoid raising it multiple times.
        self._has_raised_span_export_error = False

        if hasattr(self, "_async_queue"):
            self._span_batcher = SpanBatcher(
                async_task_queue=self._async_queue,
                log_spans_func=self._log_spans,
            )

    def _export_spans_incrementally(self, spans: Sequence[ReadableSpan]) -> None:
        """
        Export spans incrementally as they complete.

        Args:
            spans: Sequence of ReadableSpan objects to export.
        """
        location = get_active_spans_table_name()

        if not location:
            # this should not happen since this exporter is only used when a destination
            # is set to UCSchemaLocation
            _logger.debug("No active spans table name found. Skipping span export.")
            return

        # Wrapping with MLflow span interface for easier downstream handling
        spans = [Span(span) for span in spans]
        if self._should_log_async():
            for span in spans:
                self._span_batcher.add_span(location=location, span=span)
        else:
            self._log_spans(location, spans)

    def _log_spans(self, location: str, spans: list[Span]) -> None:
        try:
            self._client.log_spans(location, spans)
        except Exception as e:
            if self._has_raised_span_export_error:
                _logger.debug(f"Failed to log spans to the trace server: {e}", exc_info=True)
            else:
                _logger.warning(f"Failed to log spans to the trace server: {e}")
                self._has_raised_span_export_error = True

    def _should_enable_async_logging(self) -> bool:
        return MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.get()

    # Override this to False since spans are logged to UC table instead of artifacts.
    def _should_log_spans_to_artifacts(self, trace_info: TraceInfo) -> bool:
        return False

    def flush(self) -> None:
        self._span_batcher.shutdown()
        self._async_queue.flush(terminate=True)
