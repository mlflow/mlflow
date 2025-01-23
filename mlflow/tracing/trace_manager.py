import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Generator, Optional

from mlflow.entities import LiveSpan, Trace, TraceData, TraceInfo
from mlflow.environment_variables import MLFLOW_TRACE_TIMEOUT_SECONDS
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils.timeout import get_trace_cache_with_timeout

_logger = logging.getLogger(__name__)


# Internal representation to keep the state of a trace.
# Dict[str, Span] is used instead of TraceData to allow access by span_id.
@dataclass
class _Trace:
    info: TraceInfo
    span_dict: dict[str, LiveSpan] = field(default_factory=dict)

    def to_mlflow_trace(self) -> Trace:
        trace_data = TraceData()
        for span in self.span_dict.values():
            # Convert LiveSpan, mutable objects, into immutable Span objects before persisting.
            trace_data.spans.append(span.to_immutable_span())
            if span.parent_id is None:
                # Accessing the OTel span directly get serialized value directly.
                trace_data.request = span._span.attributes.get(SpanAttributeKey.INPUTS)
                trace_data.response = span._span.attributes.get(SpanAttributeKey.OUTPUTS)
        return Trace(self.info, trace_data)

    def get_root_span(self) -> Optional[LiveSpan]:
        for span in self.span_dict.values():
            if span.parent_id is None:
                return span
        return None


class InMemoryTraceManager:
    """
    Manage spans and traces created by the tracing system in memory.
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
        # In-memory cache to store request_id -> _Trace mapping.
        self._traces = get_trace_cache_with_timeout()

        # Store mapping between OpenTelemetry trace ID and MLflow request ID
        self._trace_id_to_request_id: dict[int, str] = {}
        self._lock = threading.Lock()  # Lock for _traces

    def register_trace(self, trace_id: int, trace_info: TraceInfo):
        """
        Register a new trace info object to the in-memory trace registry.

        Args:
            trace_id: The trace ID for the new trace.
            trace_info: The trace info object to be stored.
        """
        # Check for a new timeout setting whenever a new trace is created.
        self._check_timeout_update()
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
                _logger.debug(f"Trace data with request ID {trace_info.request_id} not found.")
                return
            self._traces[trace_info.request_id].info = trace_info

    def register_span(self, span: LiveSpan):
        """
        Store the given span in the in-memory trace data.

        Args:
            span: The span to be stored.
        """
        if not isinstance(span, LiveSpan):
            _logger.debug(f"Invalid span object {type(span)} is passed. Skipping.")
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

    def set_request_metadata(self, request_id: str, key: str, value: str):
        """
        Set the request metadata for the given request ID.
        """
        with self.get_trace(request_id) as trace:
            if trace:
                trace.info.request_metadata[key] = value

    def pop_trace(self, trace_id: int) -> Optional[Trace]:
        """
        Pop the trace data for the given id and return it as a ready-to-publish Trace object.
        """
        with self._lock:
            request_id = self._trace_id_to_request_id.pop(trace_id, None)
            trace = self._traces.pop(request_id, None)
        return trace.to_mlflow_trace() if trace else None

    def _check_timeout_update(self):
        """
        TTL/Timeout may be updated by users after initial cache creation. This method checks
        for the update and create a new cache instance with the updated timeout.
        """
        new_timeout = MLFLOW_TRACE_TIMEOUT_SECONDS.get()
        if new_timeout != getattr(self._traces, "timeout", None):
            if len(self._traces) > 0:
                _logger.warning(
                    f"The timeout of the trace buffer has been updated to {new_timeout} seconds. "
                    "This operation discards all in-progress traces at the moment. Please make "
                    "sure to update the timeout when there are no in-progress traces."
                )

            with self._lock:
                # We need to check here again in case this method runs in parallel
                if new_timeout != getattr(self._traces, "timeout", None):
                    self._traces = get_trace_cache_with_timeout()

    @classmethod
    def reset(self):
        """Clear all the aggregated trace data. This should only be used for testing."""
        if self._instance:
            with self._instance._lock:
                self._instance._traces.clear()
            self._instance = None
