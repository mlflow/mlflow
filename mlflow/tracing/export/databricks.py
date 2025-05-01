import logging
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.trace import Trace
from mlflow.environment_variables import (
    MLFLOW_ENABLE_ASYNC_TRACE_LOGGING,
)
from mlflow.tracing.client import TracingClient
from mlflow.tracing.export.async_export_queue import AsyncTraceExportQueue, Task
from mlflow.tracing.fluent import _set_last_active_trace_id
from mlflow.tracing.trace_manager import InMemoryTraceManager

_logger = logging.getLogger(__name__)


class DatabricksSpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to Databricks Tracing Server.
    """

    def __init__(self):
        self._is_async = MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.get()
        if self._is_async:
            _logger.info("MLflow is configured to log traces asynchronously.")
            self._async_queue = AsyncTraceExportQueue()
        self._client = TracingClient()

    def export(self, spans: Sequence[ReadableSpan]):
        """
        Export the spans to the destination.

        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects passed from
                a span processor. Only root spans for each trace should be exported.
        """
        for span in spans:
            if span._parent is not None:
                _logger.debug("Received a non-root span. Skipping export.")
                continue

            trace = InMemoryTraceManager.get_instance().pop_trace(span.context.trace_id)
            if trace is None:
                _logger.debug(f"Trace for span {span} not found. Skipping export.")
                continue

            _set_last_active_trace_id(trace.info.request_id)

            if self._is_async:
                self._async_queue.put(
                    task=Task(
                        handler=self._log_trace,
                        args=(trace,),
                        error_msg="Failed to log trace to the trace server.",
                    )
                )
            else:
                self._log_trace(trace)

    def _log_trace(self, trace: Trace):
        """
        Handles exporting a trace to MLflow using the V3 API and blob storage.
        Steps:
        1. Create the trace in MLflow
        2. Upload the trace data to blob storage using the returned trace info.
        """
        try:
            if trace:
                returned_trace_info = self._client.start_trace_v3(trace)
                self._client._upload_trace_data(returned_trace_info, trace.data)
            else:
                _logger.warning("No trace or trace info provided, unable to export")
        except Exception as e:
            _logger.warning(f"Failed to send trace to MLflow backend: {e}")
