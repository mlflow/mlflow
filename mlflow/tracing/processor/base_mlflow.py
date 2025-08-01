import json
import logging
from typing import Any, Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

from mlflow.entities.trace_info import TraceInfo
from mlflow.tracing.constant import (
    MAX_CHARS_IN_TRACE_INFO_METADATA,
    TRUNCATION_SUFFIX,
    SpanAttributeKey,
    TraceMetadataKey,
    TraceTagKey,
)
from mlflow.tracing.trace_manager import InMemoryTraceManager, _Trace
from mlflow.tracing.utils import (
    aggregate_usage_from_spans,
    deduplicate_span_names_in_place,
    get_otel_attribute,
    maybe_get_dependencies_schemas,
    maybe_get_logged_model_id,
    maybe_get_request_id,
    update_trace_state_from_span_conditionally,
)
from mlflow.tracing.utils.environment import resolve_env_metadata
from mlflow.tracking.fluent import (
    _get_active_model_id_global,
    _get_experiment_id,
    _get_latest_active_run,
)

_logger = logging.getLogger(__name__)


class BaseMlflowSpanProcessor(SimpleSpanProcessor):
    """
    Defines custom hooks to be executed when a span is started or ended (before exporting).

    """

    def __init__(
        self,
        span_exporter: SpanExporter,
    ):
        self.span_exporter = span_exporter
        self._trace_manager = InMemoryTraceManager.get_instance()
        self._env_metadata = resolve_env_metadata()

    def on_start(self, span: OTelSpan, parent_context: Optional[Context] = None):
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
            trace_id = trace_info.trace_id

        span.set_attribute(SpanAttributeKey.REQUEST_ID, json.dumps(trace_id))

    def _start_trace(self, root_span: OTelSpan) -> TraceInfo:
        raise NotImplementedError("Subclasses must implement this method.")

    def on_end(self, span: OTelReadableSpan) -> None:
        """
        Handle the end of a span. This method is called when an OpenTelemetry span is ended.

        Args:
            span: An OpenTelemetry ReadableSpan object that is ended.
        """
        # Processing the trace only when the root span is found.
        if span._parent is not None:
            return

        trace_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)
        with self._trace_manager.get_trace(trace_id) as trace:
            if trace is None:
                _logger.debug(f"Trace data with request ID {trace_id} not found.")
                return

            self._update_trace_info(trace, span)
            deduplicate_span_names_in_place(list(trace.span_dict.values()))

        super().on_end(span)

    def _get_experiment_id_for_trace(self, span: OTelReadableSpan) -> str:
        """
        Determine the experiment ID to associate with the trace.

        The experiment ID can be configured in multiple ways, in order of precedence:
          1. An experiment ID specified via the span creation API i.e. MlflowClient().start_trace()
          2. An experiment ID specified via `mlflow.tracing.set_destination`
          3. An experiment ID of an active run.
          4. The default experiment ID
        """
        from mlflow.tracing.provider import _MLFLOW_TRACE_USER_DESTINATION
        from mlflow.tracking.fluent import _get_latest_active_run

        if experiment_id := get_otel_attribute(span, SpanAttributeKey.EXPERIMENT_ID):
            return experiment_id

        if destination := _MLFLOW_TRACE_USER_DESTINATION.get():
            if exp_id := getattr(destination, "experiment_id"):
                return exp_id

        if run := _get_latest_active_run():
            return run.info.experiment_id

        return _get_experiment_id()

    def _get_basic_trace_metadata(self) -> dict[str, Any]:
        metadata = self._env_metadata.copy()

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
        trace.info.trace_metadata.update(
            {
                TraceMetadataKey.INPUTS: self._truncate_metadata(
                    root_span.attributes.get(SpanAttributeKey.INPUTS)
                ),
                TraceMetadataKey.OUTPUTS: self._truncate_metadata(
                    root_span.attributes.get(SpanAttributeKey.OUTPUTS)
                ),
            }
        )

        # Aggregate token usage information from all spans
        if usage := aggregate_usage_from_spans(trace.span_dict.values()):
            trace.info.request_metadata[TraceMetadataKey.TOKEN_USAGE] = json.dumps(usage)

    def _truncate_metadata(self, value: Optional[str]) -> str:
        """Get truncated value of the attribute if it exceeds the maximum length."""
        if not value:
            return ""

        if len(value) > MAX_CHARS_IN_TRACE_INFO_METADATA:
            trunc_length = MAX_CHARS_IN_TRACE_INFO_METADATA - len(TRUNCATION_SUFFIX)
            value = value[:trunc_length] + TRUNCATION_SUFFIX
        return value
