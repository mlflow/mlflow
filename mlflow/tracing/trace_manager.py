import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Generator, Optional

from cachetools import TTLCache

from mlflow.entities import LiveSpan, Trace, TraceData, TraceInfo
from mlflow.environment_variables import (
    MLFLOW_TRACE_BUFFER_MAX_SIZE,
    MLFLOW_TRACE_BUFFER_TTL_SECONDS,
)
from mlflow.tracing.constant import SpanAttributeKey

_logger = logging.getLogger(__name__)


# Internal representation to keep the state of a trace.
# Dict[str, Span] is used instead of TraceData to allow access by span_id.
@dataclass
class _Trace:
    info: TraceInfo
    span_dict: Dict[str, LiveSpan] = field(default_factory=dict)

    def to_mlflow_trace(self) -> Trace:
        trace_data = TraceData()
        for span in self.span_dict.values():
            trace_data.spans.append(span)
            if span.parent_id is None:
                # Accessing the OTel span directly get serialized value directly.
                trace_data.request = span._span.attributes.get(SpanAttributeKey.INPUTS)
                trace_data.response = span._span.attributes.get(SpanAttributeKey.OUTPUTS)
        return Trace(self.info, trace_data)


class InMemoryTraceManager:
    """
    Manage spans and traces created by the tracing system in memory.

    :meta private:
    """

    _instance_lock = threading.Lock()
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = InMemoryTraceManager()
        return cls._instance

    def __init__(self):
        # Storing request_id -> _Trace mapping
        self._traces: Dict[str, _Trace] = TTLCache(
            maxsize=MLFLOW_TRACE_BUFFER_MAX_SIZE.get(),
            ttl=MLFLOW_TRACE_BUFFER_TTL_SECONDS.get(),
        )
        # Store mapping between OpenTelemetry trace ID and MLflow request ID
        self._trace_id_to_request_id: Dict[int, str] = {}
        self._lock = threading.Lock()  # Lock for _traces

    def register_trace(self, trace_id: int, trace_info: TraceInfo):
        """
        Register a new trace info object to the in-memory trace registry.

        Args:
            trace_id: The trace ID for the new trace.
            trace_info: The trace info object to be stored.
        """
        with self._lock:
            self._traces[trace_info.request_id] = _Trace(trace_info)
            self._trace_id_to_request_id[trace_id] = trace_info.request_id

    def update_trace_info(self, trace_info: TraceInfo):
        """
        Update the trace info object in the in-memory trace registry.

        Args:
            trace_info: The updated trace info object to be stored.
        """
        with self._lock:
            if trace_info.request_id not in self._traces:
                _logger.warning(f"Trace data with request ID {trace_info.request_id} not found.")
                return
            self._traces[trace_info.request_id].info = trace_info

    def register_span(self, span: LiveSpan):
        """
        Store the given span in the in-memory trace data.

        Args:
            span: The span to be stored.
        """
        if not isinstance(span, LiveSpan):
            _logger.warning(f"Invalid span object {type(span)} is passed. Skipping.")
            return

        with self._lock:
            trace_data_dict = self._traces[span.request_id].span_dict
            trace_data_dict[span.span_id] = span

    @contextlib.contextmanager
    def get_trace(self, request_id: str) -> Generator[Optional[_Trace], None, None]:
        """
        Yield the trace info for the given request_id.
        This is designed to be used as a context manager to ensure the trace info is accessed
        with the lock held.
        """
        with self._lock:
            yield self._traces.get(request_id)

    def get_span_from_id(self, request_id: str, span_id: str) -> Optional[LiveSpan]:
        """
        Get a span object for the given request_id and span_id.
        """
        with self._lock:
            trace = self._traces.get(request_id)

        return trace.span_dict.get(span_id) if trace else None

    def get_root_span_id(self, request_id) -> Optional[str]:
        """
        Get the root span ID for the given trace ID.
        """
        with self._lock:
            trace = self._traces.get(request_id)

        if trace:
            for span in trace.span_dict.values():
                if span.parent_id is None:
                    return span.span_id

        return None

    def get_request_id_from_trace_id(self, trace_id: int) -> Optional[str]:
        """
        Get the request ID for the given trace ID.
        """
        return self._trace_id_to_request_id.get(trace_id)

    def get_mlflow_trace(self, request_id: int) -> Optional[Trace]:
        """
        Get the trace data for the given trace ID and return it as a ready-to-publish Trace object.
        """
        with self._lock:
            trace = self._traces.get(request_id)

        return trace.to_mlflow_trace() if trace else None

    def pop_trace(self, trace_id: int) -> Optional[Trace]:
        """
        Pop the trace data for the given id and return it as a ready-to-publish Trace object.
        """
        with self._lock:
            request_id = self._trace_id_to_request_id.pop(trace_id, None)
            trace = self._traces.pop(request_id, None)
        return trace.to_mlflow_trace() if trace else None

    def flush(self):
        """Clear all the aggregated trace data. This should only be used for testing."""
        with self._lock:
            self._traces.clear()
