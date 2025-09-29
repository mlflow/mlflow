import logging
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.trace import Trace
from mlflow.environment_variables import (
    MLFLOW_ENABLE_ASYNC_TRACE_LOGGING,
)
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import TraceTagKey
from mlflow.tracing.display import get_display_handler
from mlflow.tracing.export.async_export_queue import AsyncTraceExportQueue, Task
from mlflow.tracing.export.utils import try_link_prompts_to_trace
from mlflow.tracing.fluent import _EVAL_REQUEST_ID_TO_TRACE_ID, _set_last_active_trace_id
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import (
    add_size_stats_to_trace_metadata,
    encode_span_id,
    get_active_spans_table_name,
    maybe_get_request_id,
)

_logger = logging.getLogger(__name__)


class DatabricksUCTableSpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to Databricks Unity Catalog table.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._client = TracingClient(tracking_uri)
        self._is_async_enabled = MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.get()
        if self._is_async_enabled:
            self._async_queue = AsyncTraceExportQueue()

        # Display handler is no-op when running outside of notebooks.
        self._display_handler = get_display_handler()

    def export(self, spans: Sequence[ReadableSpan]) -> None:
        """
        Export the spans to the destination.

        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects passed from
                a span processor. All spans (root and non-root) are exported.
        """

        self._export_spans_incrementally(spans)
        self._export_traces(spans)

    def _export_spans_incrementally(self, spans: Sequence[ReadableSpan]) -> None:
        """
        Export spans incrementally as they complete.

        Args:
            spans: Sequence of ReadableSpan objects to export.
        """
        if active_spans_table_name := get_active_spans_table_name():
            _logger.debug(f"exporting spans to uc table: {active_spans_table_name}")

            manager = InMemoryTraceManager.get_instance()
            spans_to_export = []
            for span in spans:
                mlflow_trace_id = manager.get_mlflow_trace_id_from_otel_id(span.context.trace_id)
                with manager.get_trace(mlflow_trace_id) as internal_trace:
                    if (
                        uc_schema := internal_trace.info.trace_location.uc_schema
                    ) and uc_schema.full_otel_spans_table_name == active_spans_table_name:
                        # Get the LiveSpan from the trace manager
                        span_id = encode_span_id(span.context.span_id)
                        if mlflow_span := manager.get_span_from_id(mlflow_trace_id, span_id):
                            spans_to_export.append(mlflow_span)
                        else:
                            _logger.debug(
                                "Failed to get LiveSpan from the trace manager for span "
                                f"ID: {span_id}"
                            )
                    else:
                        _logger.debug(
                            "Trace is not associated with the active spans table, skipping export."
                        )

            if self._should_log_async():
                self._async_queue.put(
                    task=Task(
                        handler=self._client.log_spans,
                        args=(active_spans_table_name, spans_to_export),
                        error_msg="Failed to log spans to the trace server.",
                    )
                )
            else:
                self._client.log_spans(active_spans_table_name, spans_to_export)
        else:
            # this should not happen since this exporter is only used when a destination
            # is set to DatabricksUnityCatalog
            _logger.debug("No active spans table name found. Skipping span export.")

    def _export_traces(self, spans: Sequence[ReadableSpan]) -> None:
        """
        Export trace info for root spans and handle other trace-level operations.

        Args:
            spans: Sequence of ReadableSpan objects.
        """
        manager = InMemoryTraceManager.get_instance()
        for span in spans:
            if span._parent is not None:
                continue

            manager_trace = manager.pop_trace(span.context.trace_id)
            if manager_trace is None:
                _logger.debug(f"Trace for root span {span} not found. Skipping full export.")
                continue

            trace = manager_trace.trace
            _set_last_active_trace_id(trace.info.trace_id)

            # Store mapping from eval request ID to trace ID so that the evaluation
            # harness can access to the trace using mlflow.get_trace(eval_request_id)
            if eval_request_id := trace.info.tags.get(TraceTagKey.EVAL_REQUEST_ID):
                _EVAL_REQUEST_ID_TO_TRACE_ID[eval_request_id] = trace.info.trace_id

            if not maybe_get_request_id(is_evaluate=True):
                self._display_handler.display_traces([trace])

            if self._should_log_async():
                self._async_queue.put(
                    task=Task(
                        handler=self._log_trace,
                        args=(trace, manager_trace.prompts),
                        error_msg="Failed to log trace to the trace server.",
                    )
                )
            else:
                self._log_trace(trace, prompts=manager_trace.prompts)

    def _log_trace(self, trace: Trace, prompts: Sequence[PromptVersion]) -> None:
        """
        Handles exporting a trace to MLflow using the V4 API, which creates a new trace
        in the MLflow backend.
        """
        try:
            if trace:
                add_size_stats_to_trace_metadata(trace)
                self._client.start_trace(trace.info)
            else:
                _logger.warning("No trace or trace info provided, unable to export")
        except Exception as e:
            _logger.warning(f"Failed to send trace to MLflow backend: {e}")

        try:
            # Always run prompt linking asynchronously since (1) prompt linking API calls
            # would otherwise add latency to the export procedure and (2) prompt linking is not
            # critical for trace export (if the prompt fails to link, the user's workflow is
            # minorly affected), so we don't have to await successful linking
            try_link_prompts_to_trace(
                client=self._client,
                trace_id=trace.info.trace_id,
                prompts=prompts,
                synchronous=False,
            )
        except Exception as e:
            _logger.warning(f"Failed to link prompts to trace: {e}")

    def _should_log_async(self) -> bool:
        # During evaluate, the eval harness relies on the generated trace objects,
        # so we should not log traces asynchronously.
        if maybe_get_request_id(is_evaluate=True):
            return False

        return self._is_async_enabled
