import logging
from typing import Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.trace import Trace
from mlflow.tracing.constant import TraceTagKey
from mlflow.tracing.display import get_display_handler
from mlflow.tracing.display.display_handler import IPythonTraceDisplayHandler
from mlflow.tracing.fluent import TRACE_BUFFER
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

    def export(self, root_spans: Sequence[ReadableSpan]):
        """
        Export the spans to MLflow backend.

        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects to be exported.
                Only root spans for each trace are passed to this method.
        """
        for span in root_spans:
            if span._parent is not None:
                _logger.debug("Received a non-root span. Skipping export.")
                continue

            trace = self._trace_manager.pop_trace(span.context.trace_id)
            if trace is None:
                _logger.debug(f"TraceInfo for span {span} not found. Skipping export.")
                continue

            # Add the trace to the in-memory buffer
            TRACE_BUFFER[trace.info.request_id] = trace
            # Add evaluation trace to the in-memory buffer with eval_request_id key
            if eval_request_id := trace.info.tags.get(TraceTagKey.EVAL_REQUEST_ID):
                TRACE_BUFFER[eval_request_id] = trace

            if not maybe_get_request_id(is_evaluate=True):
                # Display the trace in the UI if the trace is not generated from within
                # an MLflow model evaluation context
                self._display_handler.display_traces([trace])

            # Log the trace to MLflow
            self._log_trace(trace)

    def _log_trace(self, trace: Trace):
        try:
            self._client._upload_trace_spans_as_tag(trace.info, trace.data)
        except Exception as e:
            _logger.debug(f"Failed to log trace spans as tag to MLflow backend: {e}", exc_info=True)

        # TODO: Make this async
        # The trace is already updated in processor.on_end method
        # so we just log to backend store here
        try:
            self._client._upload_trace_data(trace.info, trace.data)
            self._client._upload_ended_trace_info(trace.info)
        except Exception as e:
            # avoid silent failures
            _logger.warning(
                f"Failed to log trace to MLflow backend: {e}",
                exc_info=_logger.isEnabledFor(logging.DEBUG),
            )
