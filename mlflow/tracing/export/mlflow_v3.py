import logging
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.span import Span
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
    maybe_get_request_id,
)
from mlflow.utils.databricks_utils import is_in_databricks_notebook
from mlflow.utils.uri import is_databricks_uri

_logger = logging.getLogger(__name__)


class MlflowV3SpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to MLflow Tracking Server
    using the V3 trace schema and API.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._client = TracingClient(tracking_uri)
        self._is_async_enabled = self._should_enable_async_logging()
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
        manager = InMemoryTraceManager.get_instance()

        self._export_spans_incrementally(spans, manager)
        self._export_traces(spans, manager)

    def _export_spans_incrementally(
        self, spans: Sequence[ReadableSpan], manager: InMemoryTraceManager
    ) -> None:
        """
        Export spans incrementally as they complete.

        Args:
            spans: Sequence of ReadableSpan objects to export.
            manager: The trace manager instance.
        """
        spans_by_experiment = self._collect_mlflow_spans_for_export(spans, manager)

        for experiment_id, spans_to_log in spans_by_experiment.items():
            if self._should_log_async():
                self._async_queue.put(
                    task=Task(
                        handler=self._log_spans,
                        args=(experiment_id, spans_to_log),
                        error_msg="Failed to log spans to the trace server.",
                    )
                )
            else:
                self._log_spans(experiment_id, spans_to_log)

    def _collect_mlflow_spans_for_export(
        self, spans: Sequence[ReadableSpan], manager: InMemoryTraceManager
    ) -> dict[str, list[Span]]:
        """
        Collect MLflow spans from ReadableSpans for export, grouped by experiment ID.

        Args:
            spans: Sequence of ReadableSpan objects.
            manager: The trace manager instance.

        Returns:
            Dictionary mapping experiment_id to list of MLflow Span objects.
        """
        spans_by_experiment = {}

        for span in spans:
            mlflow_trace_id = manager.get_mlflow_trace_id_from_otel_id(span.context.trace_id)
            with manager.get_trace(mlflow_trace_id) as internal_trace:
                experiment_id = internal_trace.info.experiment_id

                # Get the LiveSpan from the trace manager
                span_id = encode_span_id(span.context.span_id)
                if mlflow_span := manager.get_span_from_id(mlflow_trace_id, span_id):
                    if experiment_id not in spans_by_experiment:
                        spans_by_experiment[experiment_id] = []
                    spans_by_experiment[experiment_id].append(mlflow_span)

        return spans_by_experiment

    def _export_traces(self, spans: Sequence[ReadableSpan], manager: InMemoryTraceManager) -> None:
        """
        Export full traces for root spans.

        Args:
            spans: Sequence of ReadableSpan objects.
            manager: The trace manager instance.
        """
        for span in spans:
            if span._parent is not None:
                continue

            manager_trace = manager.pop_trace(span.context.trace_id)
            if manager_trace is None:
                _logger.debug(f"Trace for root span {span} not found. Skipping full export.")
                continue

            trace = manager_trace.trace
            _set_last_active_trace_id(trace.info.request_id)

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

    def _log_spans(self, experiment_id: str, spans: list[Span]) -> None:
        """
        Helper method to log spans with error handling.

        Args:
            experiment_id: The experiment ID to log spans to.
            spans: List of spans to log.
        """
        try:
            self._client.log_spans(experiment_id, spans)
        except NotImplementedError:
            # Silently skip if the store doesn't support log_spans. This is expected for stores that
            # don't implement span-level logging, and we don't want to spam warnings for every span.
            pass
        except Exception as e:
            _logger.warning(f"Failed to log span to MLflow backend: {e}")

    def _log_trace(self, trace: Trace, prompts: Sequence[PromptVersion]) -> None:
        """
        Handles exporting a trace to MLflow using the V3 API and blob storage.
        Steps:
        1. Create the trace in MLflow
        2. Upload the trace data to blob storage using the returned trace info.
        """
        try:
            if trace:
                add_size_stats_to_trace_metadata(trace)
                returned_trace_info = self._client.start_trace(trace.info)
                self._client._upload_trace_data(returned_trace_info, trace.data)
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

    def _should_enable_async_logging(self) -> bool:
        if (
            is_in_databricks_notebook()
            # NB: Not defaulting OSS backend to async logging for now to reduce blast radius.
            or not is_databricks_uri(self._client.tracking_uri)
        ):
            # NB: We don't turn on async logging in Databricks notebook by default
            # until we are confident that the async logging is working on the
            # offline workload on Databricks, to derisk the inclusion to the
            # standard image. When it is enabled explicitly via the env var, we
            # will respect that.
            return (
                MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.get()
                if MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.is_set()
                else False
            )

        return MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.get()

    def _should_log_async(self) -> bool:
        # During evaluate, the eval harness relies on the generated trace objects,
        # so we should not log traces asynchronously.
        if maybe_get_request_id(is_evaluate=True):
            return False

        return self._is_async_enabled
