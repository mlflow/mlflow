import logging
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.trace_info import TraceInfo
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_TRACE_LOGGING
from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter

_logger = logging.getLogger(__name__)


class DatabricksUCTableWithOtelSpanExporter(MlflowV3SpanExporter):
    """
    An exporter that combines OTLP span export with TraceInfo persistence to UC tables.

    This exporter extends MlflowV3SpanExporter to:
    1. Export spans via OTLP (delegating to an underlying OTLP SpanExporter)
    2. Persist TraceInfo via start_trace() REST API (inherited from MlflowV3SpanExporter)

    Unlike DatabricksUCTableSpanExporter which logs spans to UC table,
    this exporter exports spans via OTLP protocol.
    """

    def __init__(
        self,
        otlp_exporter: SpanExporter,
        tracking_uri: str | None = None,
    ) -> None:
        super().__init__(tracking_uri)
        self._otlp_exporter = otlp_exporter

        # Track if we've raised an error for span export to avoid raising it multiple times.
        self._has_raised_span_export_error = False

    def _export_spans_incrementally(self, spans: Sequence[ReadableSpan]) -> None:
        """
        Export spans incrementally via OTLP.

        Args:
            spans: Sequence of ReadableSpan objects to export.
        """
        try:
            self._otlp_exporter.export(spans)
        except Exception as e:
            if self._has_raised_span_export_error:
                _logger.debug(f"Failed to export spans via OTLP: {e}", exc_info=True)
            else:
                _logger.warning(f"Failed to export spans via OTLP: {e}")
                self._has_raised_span_export_error = True

    def _should_enable_async_logging(self) -> bool:
        return MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.get()

    def _should_log_spans_to_artifacts(self, trace_info: TraceInfo) -> bool:
        # Spans are exported via OTLP, not to artifacts
        return False

    def shutdown(self) -> None:
        """Shutdown both the OTLP exporter and async queue."""
        self._otlp_exporter.shutdown()
        if hasattr(self, "_async_queue"):
            self._async_queue.flush(terminate=True)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush the underlying OTLP exporter."""
        return self._otlp_exporter.force_flush(timeout_millis)
