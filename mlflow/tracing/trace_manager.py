import atexit
from concurrent.futures import ThreadPoolExecutor
import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from cachetools import TTLCache

from mlflow.entities import LiveSpan, Trace, TraceData, TraceInfo
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.environment_variables import (
    MLFLOW_TRACE_BUFFER_MAX_SIZE,
    MLFLOW_TRACE_BUFFER_TTL_SECONDS,
)
from mlflow.exceptions import MlflowTracingException
from mlflow.tracing.constant import SpanAttributeKey

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
        # If TTL is set, use TTLCache to automatically expire the trace data that breaches the TTL.
        self._traces: dict[str, _Trace] = _TTLCacheWithLogging(
            maxsize=MLFLOW_TRACE_BUFFER_MAX_SIZE.get(),
            ttl=MLFLOW_TRACE_BUFFER_TTL_SECONDS.get(),
        )

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
        # Trigger the TTL cache expiration (non-blocking) when any new span is added.
        # NB: The expire() method in the original TTLCache is blocking and only triggered when
        # __setitem__ is called for some item. This corresponds to a new trace creation in
        # our case, which is not enough refresh rate.
        self._traces.expire()

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

    def flush(self):
        """Clear all the aggregated trace data."""
        with self._lock:
            self._traces.expire(block=True)
            self._traces.clear()


class _TTLCacheWithLogging(TTLCache):
    def __init__(self, maxsize: int, ttl: int):
        super().__init__(maxsize=maxsize, ttl=ttl)
        self._expire_traces_lock = threading.Lock()

    def expire(self, time: Optional[int] = None, block=False):
        """
        Trigger the TTL cache expiration (non-blocking).

        In addition to removing the expired traces from the cache, this method also mark it completed
        with an error status, and log it to the backend, so users can see the timeout traces on the UI.
        Since logging takes time, it is done in a background thread and this method returns immediately.
        """
        expired = self._get_expired_traces()
        if not expired:
            return

        # End the expired traces and set the status to ERROR in background thread
        def _expire_traces():
            for request_id, trace in expired:
                root_span = next((s for s in trace.span_dict.values() if s.parent_id is None), None)
                if root_span is not None:
                    try:
                        root_span.set_status(SpanStatusCode.ERROR)
                        exception_event = SpanEvent.from_exception(MlflowTracingException(
                            "This trace is automatically halted by MLflow due to the time-to-live (TTL) expiration. "
                            "The operation may be stuck or taking too long to complete. To increase the TTL duration, "
                            "set the environment variable MLFLOW_TRACE_BUFFER_TTL_SECONDS to a larger value. "
                            f"Current: {MLFLOW_TRACE_BUFFER_TTL_SECONDS.get()} seconds."
                        ))
                        root_span.add_event(exception_event)
                        root_span.end()

                        _logger.debug(f"Trace with request ID {request_id} is automatically aborted due to TTL expiration.")
                        print(f"Trace with request ID {request_id} is automatically aborted due to TTL expiration.")

                    except Exception as e:
                        print(f"Failed to expire a trace with request ID {request_id}: {e}")

            # Call the original expire method to remove from the cache
            TTLCache.expire(self, time)

        if block:
            _expire_traces()
        else:
            threading.Thread(target=_expire_traces, daemon=True).start()


    def _get_expired_traces(self) -> list[tuple[str, _Trace]]:
        """
        Fine all TTL expired traces.
        Ref: https://github.com/tkem/cachetools/blob/d44c98407030d2e91cbe82c3997be042d9c2f0de/src/cachetools/__init__.py#L469-L489
        """
        time = self.timer()
        root = self._TTLCache__root
        curr = root.next

        # Quick check when no trace is expired (most of the time)
        if curr.expires and time < curr.expires:
            return []

        expired = []
        # Require a lock to ensure only one thread is checking the linked list at a time
        with self._expire_traces_lock:
            while curr is not root and not (time < curr.expires):
                # Set the expiration time to a far future to avoid expiring it twice
                # NB: Not removing them from the cache here because
                # span exporter may access the trace stored in the cache
                curr.expires = 1e9
                self._TTLCache__links.move_to_end(curr.key)
                expired.append((curr.key, self[curr.key]))
                curr = curr.next
        return expired