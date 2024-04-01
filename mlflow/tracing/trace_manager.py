import logging
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional

from cachetools import TTLCache
from opentelemetry import trace as trace_api

from mlflow.entities import SpanType, Trace, TraceData, TraceInfo, TraceStatus
from mlflow.environment_variables import (
    MLFLOW_TRACE_BUFFER_MAX_SIZE,
    MLFLOW_TRACE_BUFFER_TTL_SECONDS,
)
from mlflow.tracing.types.wrapper import MlflowSpanWrapper, NoOpMlflowSpanWrapper

_logger = logging.getLogger(__name__)


# Internal representation to keep the state of a trace.
# Dict[str, Span] is used instead of TraceData to allow access by span_id.
@dataclass
class _Trace:
    trace_info: TraceInfo
    span_dict: Dict[str, MlflowSpanWrapper] = field(default_factory=dict)

    def to_mlflow_trace(self) -> Trace:
        span_list = [span.to_mlflow_span() for span in self.span_dict.values()]
        return Trace(self.trace_info, TraceData(spans=span_list))


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
        self._traces: Dict[str, _Trace] = TTLCache(
            maxsize=MLFLOW_TRACE_BUFFER_MAX_SIZE.get(),
            ttl=MLFLOW_TRACE_BUFFER_TTL_SECONDS.get(),
        )
        self._lock = threading.Lock()  # Lock for _traces

    def start_detached_span(
        self,
        name: str,
        request_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        span_type: str = SpanType.UNKNOWN,
    ) -> MlflowSpanWrapper:
        """
        Start a new OpenTelemetry span that is not part of the current trace context, but with the
        explicit parent span ID if provided.

        Args:
            name: The name of the span.
            request_id: The request (trace) ID for the span. Only used for getting the parent span
                for the given parent_span_id. If not provided, a new trace will be created.
            parent_span_id: The parent span ID of the span. If None, the span will be a root span.
            span_type: The type of the span.

        Returns:
            The newly created span (wrapped in MlflowSpanWrapper). If any error occurs, returns a
            NoOpMlflowSpanWrapper that has exact same interface but no-op implementations.
        """
        from mlflow.tracing.provider import get_tracer

        try:
            tracer = get_tracer(__name__)
            if parent_span_id:
                parent_span = self.get_span_from_id(request_id, parent_span_id)._span
                context = trace_api.set_span_in_context(parent_span)
            else:
                context = None

            span = MlflowSpanWrapper(tracer.start_span(name, context=context), span_type=span_type)

            self.add_or_update_span(span)
            return span
        except Exception as e:
            _logger.warning(f"Failed to start span {name}: {e}")
            return NoOpMlflowSpanWrapper()

    def add_or_update_span(self, span: MlflowSpanWrapper):
        """
        Store the given span in the in-memory trace data. If the trace does not exist, create a new
        trace with the request_id of the span. If the span ID already exists in the trace, update
        the span with the new data.

        Args:
            span: The span to be stored.
        """
        if not isinstance(span, MlflowSpanWrapper):
            _logger.warning(f"Invalid span object {type(span)} is passed. Skipping.")
            return

        request_id = span.request_id
        if request_id not in self._traces:
            # NB: the first span might not be a root span, so we can only
            # set trace_id here. Other information will be propagated from
            # the root span when it ends.
            self._create_empty_trace(request_id, span.start_time)

        trace_data_dict = self._traces[request_id].span_dict
        trace_data_dict[span.span_id] = span

    def _create_empty_trace(
        self,
        request_id: str,
        start_time_ns: int,
        experiment_id: Optional[str] = None,
        request_metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        trace_info = TraceInfo(
            request_id=request_id,
            experiment_id=experiment_id or "EXPERIMENT",  # TODO: Fetch this from global state
            timestamp_ms=start_time_ns // 1_000_000,
            execution_time_ms=None,
            status=TraceStatus.UNSPECIFIED,
            request_metadata=request_metadata or {},
            tags=tags or {},
        )
        with self._lock:
            if request_id in self._traces:
                _logger.warning(f"Trace with ID {request_id} already exists.")
                return
            self._traces[request_id] = _Trace(trace_info)

    def get_trace_info(self, request_id: str) -> Optional[TraceInfo]:
        """
        Get the trace info for the given request_id.
        """
        trace = self._traces.get(request_id)
        return trace.trace_info if trace else None

    def get_span_from_id(self, request_id: str, span_id: str) -> Optional[MlflowSpanWrapper]:
        """
        Get a span object for the given request_id and span_id.
        """
        with self._lock:
            trace = self._traces.get(request_id)

        return trace.span_dict.get(span_id) if trace else None

    # NB: Caching as this requires a linear search over all spans in the trace and
    #   the return value should not change for the same request_id.
    @lru_cache(maxsize=128)
    def get_root_span_id(self, request_id) -> Optional[str]:
        """
        Get the root span ID for the given trace ID.
        """
        with self._lock:
            trace = self._traces.get(request_id)

        if trace:
            for span in trace.span_dict.values():
                if span.parent_span_id is None:
                    return span.span_id

        return None

    def pop_trace(self, request_id) -> Optional[Trace]:
        """
        Pop the trace data for the given id and return it as a ready-to-publish Trace object.
        """
        with self._lock:
            trace: _Trace = self._traces.pop(request_id, None)
        return trace.to_mlflow_trace() if trace else None

    def flush(self):
        """Clear all the aggregated trace data. This should only be used for testing."""
        with self._lock:
            self._traces.clear()
