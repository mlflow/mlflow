import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.environment_variables import (
    MLFLOW_TRACE_BUFFER_MAX_SIZE_BYTES,
    MLFLOW_TRACE_BUFFER_TTL_SECONDS,
)
from mlflow.tracing.cache import SizedTTLCache
from mlflow.tracing.trace_manager import InMemoryTraceManager

_logger = logging.getLogger(__name__)


def pop_trace(request_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Pop the completed trace data from the buffer. This method is used in
    the Databricks model serving so please be careful when modifying it.

    Args:
        request_id: The request ID of the trace.

    Returns:
        The list of trace dictionaries if the trace is found, otherwise None.
        It can contain multiple traces if the model generated multiple traces
        for a single prediction call.
    """
    return _TRACE_BUFFER.pop(request_id, None)


def _initialize_trace_buffer():  # Define as a function for testing purposes
    """
    Create trace buffer that stores the mapping request_id (str) -> List[Dict[str, Any]]

    For Inference Table, we use special TTLCache to store the finished traces
    so that they can be retrieved by Databricks model serving. The values
    in the buffer are not Trace dataclass, but rather list of a dictionary with
    the schema that is used within Databricks model serving.

    TODO: In model serving, the request_id of trace will be Flask request ID for the
      prediction request. However, the model may generated multiple traces for a single
      prediction call depending on how it is instrumented. In such case, the request ID
      is no longer unique for a trace. In the interest of time, we don't handle this and
      simply store *list of traces* for each request ID. This should be revisited in the
      future to ensure the uniqueness of the request ID.
    """
    return SizedTTLCache(
        maxsize_bytes=MLFLOW_TRACE_BUFFER_MAX_SIZE_BYTES.get(),
        ttl=MLFLOW_TRACE_BUFFER_TTL_SECONDS.get(),
        serializer=lambda x: json.dumps(x, default=str),
    )


_TRACE_BUFFER = _initialize_trace_buffer()


class InferenceTableSpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to Inference Table.

    Currently the Inference Table does not use collector to receive the traces,
    but rather actively fetches the trace during the prediction process. In the
    future, we may consider using collector-based approach and this exporter should
    send the traces instead of storing them in the buffer.
    """

    def __init__(self):
        self._trace_manager = InMemoryTraceManager.get_instance()

    def export(self, root_spans: Sequence[ReadableSpan]):
        """
        Export the spans to MLflow backend.

        Args:
            root_spans: A sequence of OpenTelemetry ReadableSpan objects to be exported.
                Only root spans for each trace are passed to this method.
        """
        for span in root_spans:
            if span._parent is not None:
                _logger.debug("Received a non-root span. Skipping export.")
                continue

            trace = self._trace_manager.pop_trace(span.context.trace_id)
            if trace is None:
                _logger.debug(f"Trace for span {span} not found. Skipping export.")
                continue

            # Add the trace to the in-memory buffer so it can be retrieved by upstream
            trace_dict = trace.to_dict()
            if trace.info.request_id not in _TRACE_BUFFER:
                _TRACE_BUFFER[trace.info.request_id] = [trace_dict]
            else:
                _TRACE_BUFFER[trace.info.request_id].append(trace_dict)
