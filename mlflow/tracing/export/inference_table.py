import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

from cachetools import TTLCache
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.span import LiveSpan
from mlflow.environment_variables import (
    MLFLOW_TRACE_BUFFER_MAX_SIZE,
    MLFLOW_TRACE_BUFFER_TTL_SECONDS,
)
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY
from mlflow.tracing.trace_manager import InMemoryTraceManager

_logger = logging.getLogger(__name__)


def pop_trace(request_id: str) -> Optional[Dict[str, Any]]:
    """
    Pop the completed trace data from the buffer. This method is used in
    the Databricks model serving so please be careful when modifying it.
    """
    return _TRACE_BUFFER.pop(request_id, None)


# For Inference Table, we use special TTLCache to store the finished traces
# so that they can be retrieved by Databricks model serving. The values
# in the buffer are not Trace dataclass, but rather a dictionary with the schema
# that is used within Databricks model serving.
def _initialize_trace_buffer():  # Define as a function for testing purposes
    return TTLCache(
        maxsize=MLFLOW_TRACE_BUFFER_MAX_SIZE.get(),
        ttl=MLFLOW_TRACE_BUFFER_TTL_SECONDS.get(),
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
            spans: A sequence of OpenTelemetry ReadableSpan objects to be exported.
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
            _TRACE_BUFFER[trace.info.request_id] = {
                TRACE_SCHEMA_VERSION_KEY: TRACE_SCHEMA_VERSION,
                "start_timestamp": self._nanoseconds_to_datetime(span._start_time),
                "end_timestamp": self._nanoseconds_to_datetime(span._end_time),
                "spans": [self._format_spans(span) for span in trace.data.spans],
            }

    def _format_spans(self, mlflow_span: LiveSpan) -> Dict[str, Any]:
        """
        Format the MLflow span to the format that can be stored in the Inference Table.

        The schema is mostly the same, but the attributes field is stored as a JSON
        string instead of a dictionary. Therefore, the attributes are converted from
        Dict[str, str(json)] to str(json).
        """
        span_dict = mlflow_span.to_dict(dump_events=True)
        attributes = span_dict["attributes"]
        # deserialize each attribute value and then serialize the whole dictionary
        attributes = {k: self._decode_attribute(v) for k, v in attributes.items()}
        span_dict["attributes"] = json.dumps(attributes)
        return span_dict

    def _decode_attribute(self, value: str) -> Any:
        # All attribute values should be JSON encoded strings if they set via the MLflow Span
        # property, but there may be some attributes set directly to the OpenTelemetry span.
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    def _nanoseconds_to_datetime(self, t: Optional[int]) -> Optional[datetime]:
        return datetime.fromtimestamp(t / 1e9) if t is not None else None
