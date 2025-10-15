import contextlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Generator, Sequence

from mlflow.entities import LiveSpan, Trace, TraceData, TraceInfo
from mlflow.entities.model_registry import PromptVersion
from mlflow.environment_variables import MLFLOW_TRACE_TIMEOUT_SECONDS
from mlflow.prompt.constants import LINKED_PROMPTS_TAG_KEY
from mlflow.tracing.utils.prompt import update_linked_prompts_tag
from mlflow.tracing.utils.timeout import get_trace_cache_with_timeout
from mlflow.tracing.utils.truncation import set_request_response_preview

_logger = logging.getLogger(__name__)


# Internal representation to keep the state of a trace.
# Dict[str, Span] is used instead of TraceData to allow access by span_id.
@dataclass
class _Trace:
    info: TraceInfo
    span_dict: dict[str, LiveSpan] = field(default_factory=dict)
    prompts: list[PromptVersion] = field(default_factory=list)

    def to_mlflow_trace(self) -> Trace:
        trace_data = TraceData()
        for span in self.span_dict.values():
            # Convert LiveSpan, mutable objects, into immutable Span objects before persisting.
            trace_data.spans.append(span.to_immutable_span())

        set_request_response_preview(self.info, trace_data)
        return Trace(self.info, trace_data)

    def get_root_span(self) -> LiveSpan | None:
        for span in self.span_dict.values():
            if span.parent_id is None:
                return span
        return None


@dataclass
class ManagerTrace:
    """
    Wrapper around a trace and its associated prompts.
    """

    trace: Trace
    prompts: Sequence[PromptVersion]


class InMemoryTraceManager:
    """
    Manage spans and traces created by the tracing system in memory.
    """

    _instance_lock = threading.RLock()
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = InMemoryTraceManager()
        return cls._instance

    def __init__(self):
        # In-memory cache to store trace_od -> _Trace mapping.
        self._traces = get_trace_cache_with_timeout()

        # Store mapping between OpenTelemetry trace ID and MLflow trace ID
        self._otel_id_to_mlflow_trace_id: dict[int, str] = {}
        self._lock = threading.RLock()  # Lock for _traces

    def register_trace(self, otel_trace_id: int, trace_info: TraceInfo):
        """
        Register a new trace info object to the in-memory trace registry.

        Args:
            otel_trace_id: The OpenTelemetry trace ID for the new trace.
            trace_info: The trace info object to be stored.
        """
        # Check for a new timeout setting whenever a new trace is created.
        self._check_timeout_update()
        with self._lock:
            self._traces[trace_info.trace_id] = _Trace(trace_info)
            self._otel_id_to_mlflow_trace_id[otel_trace_id] = trace_info.trace_id

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

    def register_prompt(self, trace_id: str, prompt: PromptVersion):
        """
        Register a prompt to link to the trace with the given trace ID.

        Args:
            trace_id: The ID of the trace to which the prompt belongs.
            prompt: The prompt version to be registered.
        """
        with self._lock:
            if prompt not in self._traces[trace_id].prompts:
                self._traces[trace_id].prompts.append(prompt)

            # NB: Set prompt URIs in trace tags for linking. This is a short-term solution until
            # LinkPromptsToTraces endpoint is implemented in the backend.
            # TODO: Remove this once LinkPromptsToTraces endpoint is implemented in the backend.
            try:
                current_tag = self._traces[trace_id].info.tags.get(LINKED_PROMPTS_TAG_KEY)
                updated_tag = update_linked_prompts_tag(current_tag, [prompt])
                self._traces[trace_id].info.tags[LINKED_PROMPTS_TAG_KEY] = updated_tag
            except Exception:
                _logger.debug(f"Failed to update prompts tag for trace {trace_id}", exc_info=True)
                raise

    @contextlib.contextmanager
    def get_trace(self, trace_id: str) -> Generator[_Trace | None, None, None]:
        """
        Yield the trace info for the given trace ID..
        This is designed to be used as a context manager to ensure the trace info is accessed
        with the lock held.
        """
        with self._lock:
            yield self._traces.get(trace_id)

    def get_span_from_id(self, trace_id: str, span_id: str) -> LiveSpan | None:
        """
        Get a span object for the given trace_id and span_id.
        """
        with self._lock:
            trace = self._traces.get(trace_id)

        return trace.span_dict.get(span_id) if trace else None

    def get_root_span_id(self, trace_id) -> str | None:
        """
        Get the root span ID for the given trace ID.
        """
        with self._lock:
            trace = self._traces.get(trace_id)

        if trace:
            for span in trace.span_dict.values():
                if span.parent_id is None:
                    return span.span_id

        return None

    def get_mlflow_trace_id_from_otel_id(self, otel_trace_id: int) -> str | None:
        """
        Get the MLflow trace ID for the given OpenTelemetry trace ID.
        """
        return self._otel_id_to_mlflow_trace_id.get(otel_trace_id)

    def set_trace_metadata(self, trace_id: str, key: str, value: str):
        """
        Set the trace metadata for the given request ID.
        """
        with self.get_trace(trace_id) as trace:
            if trace:
                trace.info.trace_metadata[key] = value

    def pop_trace(self, otel_trace_id: int) -> ManagerTrace | None:
        """
        Pop trace data for the given OpenTelemetry trace ID and
        return it as a ManagerTrace wrapper containing the trace and prompts.
        """
        with self._lock:
            mlflow_trace_id = self._otel_id_to_mlflow_trace_id.pop(otel_trace_id, None)
            internal_trace = self._traces.pop(mlflow_trace_id, None) if mlflow_trace_id else None
            if internal_trace is None:
                return None
            return ManagerTrace(
                trace=internal_trace.to_mlflow_trace(), prompts=internal_trace.prompts
            )

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
    def reset(cls):
        """Clear all the aggregated trace data. This should only be used for testing."""
        if cls._instance:
            with cls._instance._lock:
                cls._instance._traces.clear()
            cls._instance = None
