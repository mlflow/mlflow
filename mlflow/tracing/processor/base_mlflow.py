import json
import logging
import threading
import weakref
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)

from mlflow.entities.span import create_mlflow_span
from mlflow.entities.trace_info import TraceInfo
from mlflow.environment_variables import (
    MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS,
    MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE,
)
from mlflow.tracing.constant import (
    MAX_CHARS_IN_TRACE_INFO_METADATA,
    TRACE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION_KEY,
    TRUNCATION_SUFFIX,
    SpanAttributeKey,
    TraceMetadataKey,
    TraceTagKey,
)
from mlflow.tracing.context import get_configured_trace_metadata, get_configured_trace_tags
from mlflow.tracing.fluent import _set_last_active_trace_id
from mlflow.tracing.processor.otel_metrics_mixin import OtelMetricsMixin
from mlflow.tracing.trace_manager import InMemoryTraceManager, _Trace
from mlflow.tracing.utils import (
    aggregate_cost_from_spans,
    aggregate_usage_from_spans,
    get_otel_attribute,
    maybe_get_dependencies_schemas,
    maybe_get_logged_model_id,
    maybe_get_request_id,
    should_compute_cost_client_side,
    update_trace_state_from_span_conditionally,
)
from mlflow.tracing.utils.environment import resolve_env_metadata
from mlflow.tracking.fluent import (
    _get_active_model_id_global,
    _get_latest_active_run,
)

_logger = logging.getLogger(__name__)


# Default max_queue_size in OTel's BatchSpanProcessor.
# https://opentelemetry.io/docs/specs/otel/trace/sdk/#batching-processor
_DEFAULT_OTEL_MAX_QUEUE_SIZE = 2048

# Registry of all BaseMlflowSpanProcessor instances that have a batch delegate.
# When set_destination() creates a new tracer provider, the old processor is orphaned
# but its BatchSpanProcessor background thread keeps running with queued spans.
# This registry allows flush_all_batch_processors() to drain all of them.
# Uses WeakSet so processors that are garbage-collected (e.g., when the tracer
# provider is replaced) are automatically removed without unbounded growth.
_batch_processor_registry: weakref.WeakSet["BaseMlflowSpanProcessor"] = weakref.WeakSet()
_batch_processor_registry_lock = threading.Lock()


def flush_all_batch_processors(timeout_millis: float = 30000, terminate: bool = False) -> None:
    """Flush all registered batch processors and their exporters' async queues.

    Two-layer flush:
      1. force_flush each BatchSpanProcessor (drains span queue → exporter.export())
      2. flush each exporter's _async_queue (drains DB write queue → tracking store)

    Only after both layers are drained do we optionally shutdown.

    Args:
        timeout_millis: Timeout per processor for force_flush.
        terminate: If True, also shutdown all processors and clear the registry.
    """
    with _batch_processor_registry_lock:
        processors = list(_batch_processor_registry)
        # Clear immediately so any new processors created during flush
        # go into a fresh registry.
        if terminate:
            _batch_processor_registry.clear()
    # Wait for all in-flight on_end calls to finish before flushing.
    # This guarantees every span is in the BSP queue before force_flush() is
    # called, preventing the race where a span arrives just after the flush
    # signal is sent to the BSP worker thread.
    # Note: wait_for() always evaluates the predicate before blocking, so even
    # if notify_all() fires before wait_for() is entered (counter already 0),
    # the predicate is true and wait_for() returns immediately.
    timeout_secs = timeout_millis / 1000
    for processor in processors:
        with processor._pending_on_end_condition:
            processor._pending_on_end_condition.wait_for(
                lambda: processor._pending_on_end_count == 0,
                timeout=timeout_secs,
            )

    # Layer 1: drain span queues into exporters
    for processor in processors:
        try:
            processor.force_flush(timeout_millis)
        except Exception:
            _logger.debug(f"Failed to flush processor {processor}", exc_info=True)
    # Layer 2: drain all exporters' async queues into the tracking store
    for processor in processors:
        try:
            exporter = processor.span_exporter
            if hasattr(exporter, "_async_queue"):
                exporter._async_queue.flush(terminate=terminate)
        except Exception:
            _logger.debug(f"Failed to flush exporter queue for {processor}", exc_info=True)
    if terminate:
        for processor in processors:
            try:
                processor.shutdown()
                # Null out the delegate so future on_end calls fall through
                # to SimpleSpanProcessor instead of going to the dead batch
                # processor. This is critical for test isolation: the tracer
                # provider may outlive the shutdown and reuse the processor.
                processor._batch_delegate = None
            except Exception:
                _logger.debug(f"Failed to shutdown processor {processor}", exc_info=True)


def _create_batch_span_processor(exporter: SpanExporter) -> BatchSpanProcessor:
    max_export_batch_size = MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE.get()
    # OTel requires max_export_batch_size <= max_queue_size (raises ValueError otherwise).
    max_queue_size = max(max_export_batch_size, _DEFAULT_OTEL_MAX_QUEUE_SIZE)
    return BatchSpanProcessor(
        exporter,
        schedule_delay_millis=MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS.get(),
        max_queue_size=max_queue_size,
        max_export_batch_size=max_export_batch_size,
    )


class BaseMlflowSpanProcessor(OtelMetricsMixin, SimpleSpanProcessor):
    """
    Defines custom hooks to be executed when a span is started or ended (before exporting).

    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        export_metrics: bool,
        use_batch_processor: bool = False,
    ):
        # Always call the full MRO __init__ chain (OtelMetricsMixin ->
        # SimpleSpanProcessor) so _trace_manager and other state is set up.
        super().__init__(span_exporter)
        self._batch_delegate = (
            _create_batch_span_processor(span_exporter) if use_batch_processor else None
        )
        if self._batch_delegate is not None:
            with _batch_processor_registry_lock:
                _batch_processor_registry.add(self)
        self.span_exporter = span_exporter
        self._export_metrics = export_metrics
        self._env_metadata = resolve_env_metadata()
        # Lock to prevent race conditions during concurrent span name deduplication
        # This ensures that when multiple spans end simultaneously, their names are
        # deduplicated atomically without interference
        self._deduplication_lock = threading.RLock()
        # Counter tracking in-flight on_end calls. flush_all_batch_processors()
        # waits for this to reach 0 before calling force_flush(), ensuring every
        # span is in the BSP queue before the flush starts.
        self._pending_on_end_count = 0
        self._pending_on_end_condition = threading.Condition(threading.Lock())

    def on_start(self, span: OTelSpan, parent_context: Context | None = None):
        """
        Handle the start of a span. This method is called when an OpenTelemetry span is started.

        Args:
            span: An OpenTelemetry Span object that is started.
            parent_context: The context of the span. Note that this is only passed when the context
                object is explicitly specified to OpenTelemetry start_span call. If the parent span
                is obtained from the global context, it won't be passed here so we should not rely
                on it.
        """
        trace_id = self._trace_manager.get_mlflow_trace_id_from_otel_id(span.context.trace_id)

        if not trace_id and span.parent is not None:
            _logger.debug(
                "Received a non-root span but the trace ID is not found."
                "The trace has likely been halted due to a timeout expiration."
            )
            return

        if span.parent is None:
            trace_info = self._start_trace(span)
            if trace_info is None:
                return
            trace_id = trace_info.trace_id

        InMemoryTraceManager.get_instance().register_span(create_mlflow_span(span, trace_id))

    def _start_trace(self, root_span: OTelSpan) -> TraceInfo:
        raise NotImplementedError("Subclasses must implement this method.")

    def on_end(self, span: OTelReadableSpan) -> None:
        """
        Handle the end of a span. This method is called when an OpenTelemetry span is ended.

        Args:
            span: An OpenTelemetry ReadableSpan object that is ended.
        """
        with self._pending_on_end_condition:
            self._pending_on_end_count += 1
        try:
            self._on_end_impl(span)
        finally:
            with self._pending_on_end_condition:
                self._pending_on_end_count -= 1
                if self._pending_on_end_count == 0:
                    self._pending_on_end_condition.notify_all()

    def _on_end_impl(self, span: OTelReadableSpan) -> None:
        if self._export_metrics:
            self.record_metrics_for_span(span)

        trace_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)

        # Acquire lock before accessing and modifying trace data to prevent race conditions
        # during concurrent span endings. This ensures span name deduplication happens
        # atomically without interference from other threads
        with self._deduplication_lock:
            with self._trace_manager.get_trace(trace_id) as trace:
                if trace is not None:
                    if span._parent is None:
                        self._update_trace_info(trace, span)
                        # Set the last active trace ID immediately so that
                        # mlflow.get_trace() returns the correct trace even in batch mode.
                        _set_last_active_trace_id(trace_id)
                else:
                    _logger.debug(f"Trace data with request ID {trace_id} not found.")

        # During evaluation, bypass batch mode to ensure traces are available
        # synchronously for the evaluation harness.
        if self._batch_delegate is not None and not maybe_get_request_id(is_evaluate=True):
            self._batch_delegate.on_end(span)
        else:
            super().on_end(span)

    def shutdown(self) -> None:
        if self._batch_delegate is not None:
            self._batch_delegate.shutdown()
        super().shutdown()

    def force_flush(self, timeout_millis: float = 30000) -> bool:
        if self._batch_delegate is not None:
            return self._batch_delegate.force_flush(timeout_millis)
        return super().force_flush(timeout_millis)

    def _get_basic_trace_metadata(self) -> dict[str, Any]:
        metadata = self._env_metadata.copy()

        metadata[TRACE_SCHEMA_VERSION_KEY] = str(TRACE_SCHEMA_VERSION)

        # If the span is started within an active MLflow run, we should record it as a trace tag
        # Note `mlflow.active_run()` can only get thread-local active run,
        # but tracing routine might be applied to model inference worker threads
        # in the following cases:
        #  - langchain model `chain.batch` which uses thread pool to spawn workers.
        #  - MLflow langchain pyfunc model `predict` which calls `api_request_parallel_processor`.
        # Therefore, we use `_get_global_active_run()` instead to get the active run from
        # all threads and set it as the tracing source run.
        if run := _get_latest_active_run():
            metadata[TraceMetadataKey.SOURCE_RUN] = run.info.run_id

        # The order is:
        # 1. model_id of the current active model set by `set_active_model`
        # 2. model_id from the current prediction context
        #   (set by mlflow pyfunc predict, or explicitly using set_prediction_context)
        if active_model_id := _get_active_model_id_global():
            metadata[TraceMetadataKey.MODEL_ID] = active_model_id
        elif model_id := maybe_get_logged_model_id():
            metadata[TraceMetadataKey.MODEL_ID] = model_id

        # Append metadata from context() scope (caller-declared, wins on conflict)
        if ctx_metadata := get_configured_trace_metadata():
            metadata.update(ctx_metadata)

        return metadata

    def _get_basic_trace_tags(self, span: OTelReadableSpan) -> dict[str, Any]:
        # If the trace is created in the context of MLflow model evaluation, we extract the request
        # ID from the prediction context. Otherwise, we create a new trace info by calling the
        # backend API.
        tags = {}
        if request_id := maybe_get_request_id(is_evaluate=True):
            tags.update({TraceTagKey.EVAL_REQUEST_ID: request_id})
        if dependencies_schema := maybe_get_dependencies_schemas():
            tags.update(dependencies_schema)

        # Append tags from context() scope before trace name
        # (trace name tag always wins because it comes last)
        if ctx_tags := get_configured_trace_tags():
            tags.update(ctx_tags)

        tags.update({TraceTagKey.TRACE_NAME: span.name})
        return tags

    def _update_trace_info(self, trace: _Trace, root_span: OTelReadableSpan):
        """Update the trace info with the final values from the root span."""
        # The trace/span start time needs adjustment to exclude the latency of
        # the backend API call. We already adjusted the span start time in the
        # on_start method, so we reflect the same to the trace start time here.
        trace.info.request_time = root_span.start_time // 1_000_000  # nanosecond to millisecond
        trace.info.execution_duration = (root_span.end_time - root_span.start_time) // 1_000_000

        # Update trace state from span status, but only if the user hasn't explicitly set
        # a different trace status
        update_trace_state_from_span_conditionally(trace, root_span)

        # TODO: Remove this once the new trace table UI is available that is based on V3 trace.
        # Until then, these two are still used to render the "request" and "response" columns.
        trace.info.trace_metadata.update({
            TraceMetadataKey.INPUTS: self._truncate_metadata(
                root_span.attributes.get(SpanAttributeKey.INPUTS)
            ),
            TraceMetadataKey.OUTPUTS: self._truncate_metadata(
                root_span.attributes.get(SpanAttributeKey.OUTPUTS)
            ),
        })

        spans = trace.span_dict.values()
        # Aggregate token usage information from all spans
        if usage := aggregate_usage_from_spans(spans):
            trace.info.request_metadata[TraceMetadataKey.TOKEN_USAGE] = json.dumps(usage)

        if should_compute_cost_client_side() and (cost := aggregate_cost_from_spans(spans)):
            trace.info.request_metadata[TraceMetadataKey.COST] = json.dumps(cost)

    def _truncate_metadata(self, value: str | None) -> str:
        """Get truncated value of the attribute if it exceeds the maximum length."""
        if not value:
            return ""

        if len(value) > MAX_CHARS_IN_TRACE_INFO_METADATA:
            trunc_length = MAX_CHARS_IN_TRACE_INFO_METADATA - len(TRUNCATION_SUFFIX)
            value = value[:trunc_length] + TRUNCATION_SUFFIX
        return value
