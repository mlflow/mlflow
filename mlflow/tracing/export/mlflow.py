import logging
from typing import Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.trace import Trace
from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_LOGGING
from mlflow.tracing.constant import TraceTagKey
from mlflow.tracing.display import get_display_handler
from mlflow.tracing.display.display_handler import IPythonTraceDisplayHandler
from mlflow.tracing.export.async_export_queue import AsyncTraceExportQueue, Task
from mlflow.tracing.fluent import _EVAL_REQUEST_ID_TO_TRACE_ID, _set_last_active_trace_id
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import maybe_get_request_id
from mlflow.tracking.client import MlflowClient

_logger = logging.getLogger(__name__)


class MlflowSpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to MLflow.

    MLflow backend (will) only support logging the complete trace, not incremental updates
    for spans, so this exporter is designed to aggregate the spans into traces in memory.
    Therefore, this only works within a single process application and not intended to work
    in a distributed environment. For the same reason, this exporter should only be used with
    SimpleSpanProcessor.

    If we want to support distributed tracing, we should first implement an incremental trace
    logging in MLflow backend, then we can get rid of the in-memory trace aggregation.

    :meta private:
    """

    def __init__(
        self,
        client: Optional[MlflowClient] = None,
        display_handler: Optional[IPythonTraceDisplayHandler] = None,
    ):
        self._client = client or MlflowClient()
        self._display_handler = display_handler or get_display_handler()
        self._trace_manager = InMemoryTraceManager.get_instance()
        self._async_queue = AsyncTraceExportQueue()

    def export(self, spans: Sequence[ReadableSpan]):
        """
        Export the spans to MLflow backend.

        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects passed from
                a span processor. Only root spans for each trace should be exported.
        """
        for span in spans:
            if span._parent is not None:
                _logger.debug("Received a non-root span. Skipping export.")
                continue

            trace = self._trace_manager.pop_trace(span.context.trace_id)
            if trace is None:
                _logger.debug(f"TraceInfo for span {span} not found. Skipping export.")
                continue

            _set_last_active_trace_id(trace.info.request_id)

            # Store mapping from eval request ID to trace ID so that the evaluation
            # harness can access to the trace using mlflow.get_trace(eval_request_id)
            if eval_request_id := trace.info.tags.get(TraceTagKey.EVAL_REQUEST_ID):
                _EVAL_REQUEST_ID_TO_TRACE_ID[eval_request_id] = trace.info.request_id

            if not maybe_get_request_id(is_evaluate=True):
                # Display the trace in the UI if the trace is not generated from within
                # an MLflow model evaluation context
                self._display_handler.display_traces([trace])

            self._log_trace(trace)

    def _log_trace(self, trace: Trace):
        """Log the trace to MLflow backend."""
        upload_trace_data_task = Task(
            handler=self._client._upload_trace_data,
            args=(trace.info, trace.data),
            error_msg="Failed to log trace to MLflow backend.",
        )

        upload_ended_trace_info_task = Task(
            handler=self._client._upload_ended_trace_info,
            args=(trace.info,),
            error_msg="Failed to log trace to MLflow backend.",
        )

        # TODO: Use MLFLOW_ENABLE_ASYNC_TRACE_LOGGING instead and default to async
        # logging once the async logging implementation becomes stable.
        if MLFLOW_ENABLE_ASYNC_LOGGING.get():
            self._async_queue.put(upload_trace_data_task)
            self._async_queue.put(upload_ended_trace_info_task)
        else:
            upload_trace_data_task.handle()
            upload_ended_trace_info_task.handle()
