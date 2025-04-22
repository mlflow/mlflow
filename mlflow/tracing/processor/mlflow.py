import json
import logging
import time
from typing import Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.tracing.constant import (
    MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS,
    TRACE_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION_KEY,
    TRUNCATION_SUFFIX,
    SpanAttributeKey,
    TraceMetadataKey,
    TraceTagKey,
)
from mlflow.tracing.trace_manager import InMemoryTraceManager, _Trace
from mlflow.tracing.utils import (
    deduplicate_span_names_in_place,
    get_otel_attribute,
    maybe_get_dependencies_schemas,
    maybe_get_request_id,
)
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context.databricks_repo_context import DatabricksRepoRunContext
from mlflow.tracking.context.git_context import GitRunContext
from mlflow.tracking.context.registry import resolve_tags
from mlflow.tracking.default_experiment import DEFAULT_EXPERIMENT_ID
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.mlflow_tags import TRACE_RESOLVE_TAGS_ALLOWLIST

_logger = logging.getLogger(__name__)


class MlflowSpanProcessor(SimpleSpanProcessor):
    """
    Defines custom hooks to be executed when a span is started or ended (before exporting).

    This processor is used when the tracing destination is MLflow Tracking Server.
    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        client: Optional[MlflowClient] = None,
        experiment_id: Optional[str] = None,
    ):
        self.span_exporter = span_exporter
        self._client = client or MlflowClient()
        self._experiment_id = experiment_id
        self._trace_manager = InMemoryTraceManager.get_instance()

        # We issue a warning when a trace is created under the default experiment.
        # We only want to issue it once, and typically it can be achieved by using
        # warnings.warn() with filterwarnings setting. However, the de-duplication does
        # not work in notebooks (https://github.com/ipython/ipython/issues/11207),
        # so we instead keep track of the warning issuance state manually.
        self._issued_default_exp_warning = False

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
        request_id = self._trace_manager.get_request_id_from_trace_id(span.context.trace_id)

        if not request_id and span.parent is not None:
            _logger.debug(
                "Received a non-root span but the request ID is not found."
                "The trace has likely been halted due to a timeout expiration."
            )
            return

        if not request_id:
            # If the user started trace/span with fixed start time, this attribute is set
            start_time_ns = get_otel_attribute(span, SpanAttributeKey.START_TIME_NS)

            trace_info = self._start_trace(span, start_time_ns)
            self._trace_manager.register_trace(span.context.trace_id, trace_info)
            request_id = trace_info.request_id

            # NB: This is a workaround to exclude the latency of backend StartTrace API call (within
            #   _create_trace_info()) from the execution time of the span. The API call takes ~1 sec
            #   and significantly skews the span duration.
            if not start_time_ns:
                span._start_time = time.time_ns()

        span.set_attribute(SpanAttributeKey.REQUEST_ID, json.dumps(request_id))

    def _start_trace(self, span: OTelSpan, start_time_ns: Optional[int]) -> TraceInfo:
        from mlflow.tracking.fluent import _get_latest_active_run

        metadata = {TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)}

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

        experiment_id = self._get_experiment_id_for_trace(span)
        if experiment_id == DEFAULT_EXPERIMENT_ID and not self._issued_default_exp_warning:
            _logger.warning(
                "Creating a trace within the default experiment with id "
                f"'{DEFAULT_EXPERIMENT_ID}'. It is strongly recommended to not use "
                "the default experiment to log traces due to ambiguous search results and "
                "probable performance issues over time due to directory table listing performance "
                "degradation with high volumes of directories within a specific path. "
                "To avoid performance and disambiguation issues, set the experiment for "
                "your environment using `mlflow.set_experiment()` API."
            )
            self._issued_default_exp_warning = True

        # Avoid running unnecessary context providers to avoid overhead
        unfiltered_tags = resolve_tags(ignore=[DatabricksRepoRunContext, GitRunContext])
        tags = {
            key: value
            for key, value in unfiltered_tags.items()
            if key in TRACE_RESOLVE_TAGS_ALLOWLIST
        }

        # If the trace is created in the context of MLflow model evaluation, we extract the request
        # ID from the prediction context. Otherwise, we create a new trace info by calling the
        # backend API.
        if request_id := maybe_get_request_id(is_evaluate=True):
            tags.update({TraceTagKey.EVAL_REQUEST_ID: request_id})
        if dependencies_schema := maybe_get_dependencies_schemas():
            tags.update(dependencies_schema)
        tags.update({TraceTagKey.TRACE_NAME: span.name})

        return self._client._start_tracked_trace(
            experiment_id=experiment_id,
            # TODO: This timestamp is not accurate because it is not adjusted to exclude the
            #   latency of the backend API call. We do this adjustment for span start time
            #   above, but can't do it for trace start time until the backend API supports
            #   updating the trace start time.
            timestamp_ms=(start_time_ns or span.start_time) // 1_000_000,  # ns to ms
            request_metadata=metadata,
            tags=tags,
        )

    def on_end(self, span: OTelReadableSpan) -> None:
        """
        Handle the end of a span. This method is called when an OpenTelemetry span is ended.

        Args:
            span: An OpenTelemetry ReadableSpan object that is ended.
        """
        # Processing the trace only when the root span is found.
        if span._parent is not None:
            return

        request_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)
        with self._trace_manager.get_trace(request_id) as trace:
            if trace is None:
                _logger.debug(f"Trace data with request ID {request_id} not found.")
                return

            self._update_trace_info(trace, span)
            deduplicate_span_names_in_place(list(trace.span_dict.values()))

        super().on_end(span)

    def _get_experiment_id_for_trace(self, span: OTelReadableSpan) -> str:
        """
        Determine the experiment ID to associate with the trace.

        The experiment ID can be configured in multiple ways, in order of precedence:
          1. An experiment ID specified via the span creation API i.e. MlflowClient().start_trace()
          2. An experiment ID specified via the processor constructor
          3. An experiment ID of an active run.
          4. The default experiment ID
        """
        from mlflow.tracking.fluent import _get_latest_active_run

        if experiment_id := get_otel_attribute(span, SpanAttributeKey.EXPERIMENT_ID):
            return experiment_id

        if self._experiment_id:
            return self._experiment_id

        if run := _get_latest_active_run():
            return run.info.experiment_id

        return _get_experiment_id()

    def _update_trace_info(self, trace: _Trace, root_span: OTelReadableSpan):
        """Update the trace info with the final values from the root span."""
        # The trace/span start time needs adjustment to exclude the latency of
        # the backend API call. We already adjusted the span start time in the
        # on_start method, so we reflect the same to the trace start time here.
        trace.info.timestamp_ms = root_span.start_time // 1_000_000  # nanosecond to millisecond
        trace.info.execution_time_ms = (root_span.end_time - root_span.start_time) // 1_000_000
        trace.info.status = TraceStatus.from_otel_status(root_span.status)
        trace.info.request_metadata.update(
            {
                TraceMetadataKey.INPUTS: self._truncate_metadata(
                    root_span.attributes.get(SpanAttributeKey.INPUTS)
                ),
                TraceMetadataKey.OUTPUTS: self._truncate_metadata(
                    root_span.attributes.get(SpanAttributeKey.OUTPUTS)
                ),
            }
        )

    def _truncate_metadata(self, value: Optional[str]) -> str:
        """Get truncated value of the attribute if it exceeds the maximum length."""
        if not value:
            return ""

        if len(value) > MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS:
            trunc_length = MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS - len(TRUNCATION_SUFFIX)
            value = value[:trunc_length] + TRUNCATION_SUFFIX
        return value
