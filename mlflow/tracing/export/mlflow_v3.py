import logging
from typing import Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.trace import Trace
from mlflow.environment_variables import (
    MLFLOW_ENABLE_ASYNC_TRACE_LOGGING,
    MLFLOW_TRACING_ENABLE_DELTA_ARCHIVAL,
    MLFLOW_TRACING_DELTA_ARCHIVAL_SPANS_TABLE,
    MLFLOW_TRACING_DELTA_ARCHIVAL_TOKEN,
    MLFLOW_TRACING_DELTA_ARCHIVAL_INGESTION_URL,
    MLFLOW_TRACING_DELTA_ARCHIVAL_WORKSPACE_URL,
)
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import TraceTagKey
from mlflow.tracing.display import get_display_handler
from mlflow.tracing.export.async_export_queue import AsyncTraceExportQueue, Task
from mlflow.tracing.export.utils import try_link_prompts_to_trace
from mlflow.tracing.fluent import _EVAL_REQUEST_ID_TO_TRACE_ID, _set_last_active_trace_id
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import add_size_bytes_to_trace_metadata, maybe_get_request_id
from mlflow.utils.databricks_utils import is_in_databricks_notebook
from mlflow.tracing.export.databricks_delta import DatabricksDeltaExporter

_logger = logging.getLogger(__name__)


class MlflowV3SpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to MLflow Tracking Server
    using the V3 trace schema and API.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
    ):
        """
        Initialize the MlflowV3SpanExporter.
        
        Args:
            tracking_uri: The MLflow tracking URI.
        """
        self._is_async_enabled = self._should_enable_async_logging()
        if self._is_async_enabled:
            self._async_queue = AsyncTraceExportQueue()
        self._client = TracingClient(tracking_uri)

        # Only display traces inline in Databricks notebooks
        self._should_display_trace = is_in_databricks_notebook()
        if self._should_display_trace:
            self._display_handler = get_display_handler()

        # Initialize Databricks Delta archival if enabled
        self._delta_exporter = self._get_delta_exporter()

    def export(self, spans: Sequence[ReadableSpan]):
        """
        Export the spans to the destination.

        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects passed from
                a span processor. Only root spans for each trace should be exported.
        """
        for span in spans:
            if span._parent is not None:
                _logger.debug("Received a non-root span. Skipping export.")
                continue

            manager_trace = InMemoryTraceManager.get_instance().pop_trace(span.context.trace_id)
            if manager_trace is None:
                _logger.debug(f"Trace for span {span} not found. Skipping export.")
                continue

            trace = manager_trace.trace
            _set_last_active_trace_id(trace.info.request_id)

            # Store mapping from eval request ID to trace ID so that the evaluation
            # harness can access to the trace using mlflow.get_trace(eval_request_id)
            if eval_request_id := trace.info.tags.get(TraceTagKey.EVAL_REQUEST_ID):
                _EVAL_REQUEST_ID_TO_TRACE_ID[eval_request_id] = trace.info.trace_id

            if self._should_display_trace and not maybe_get_request_id(is_evaluate=True):
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

    def _log_trace(self, trace: Trace, prompts: Sequence[PromptVersion]):
        """
        Handles exporting a trace to MLflow using the V3 API and blob storage.
        Steps:
        1. Create the trace in MLflow
        2. Upload the trace data to blob storage using the returned trace info.
        3. If enabled, export to Databricks Delta table
        """
        try:
            if trace:
                try:
                    add_size_bytes_to_trace_metadata(trace)
                except Exception:
                    _logger.warning("Failed to add size bytes to trace metadata.", exc_info=True)
                returned_trace_info = self._client.start_trace_v3(trace)
                self._client._upload_trace_data(returned_trace_info, trace.data)
                # Always run prompt linking asynchronously since (1) prompt linking API calls
                # would otherwise add latency to the export procedure and (2) prompt linking is not
                # critical for trace export (if the prompt fails to link, the user's workflow is
                # minorly affected), so we don't have to await successful linking
                try_link_prompts_to_trace(
                    client=self._client,
                    trace_id=returned_trace_info.trace_id,
                    prompts=prompts,
                    synchronous=False,
                )
                
                # Export to Databricks Delta if enabled
                if self._delta_exporter:
                    self._log_trace_to_delta(trace)
            else:
                _logger.warning("No trace or trace info provided, unable to export")
        except Exception as e:
            _logger.warning(f"Failed to send trace to MLflow backend: {e}")

    def _should_enable_async_logging(self):
        if is_in_databricks_notebook():
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

    def _should_log_async(self):
        # During evaluate, the eval harness relies on the generated trace objects,
        # so we should not log traces asynchronously.
        if maybe_get_request_id(is_evaluate=True):
            return False

        return self._is_async_enabled

    def _should_enable_delta_archival(self) -> bool:
        """Check if Databricks Delta archival should be enabled."""
        if not MLFLOW_TRACING_ENABLE_DELTA_ARCHIVAL.get():
            return False
        
        # Check if all required environment variables are set
        required_vars = [
            MLFLOW_TRACING_DELTA_ARCHIVAL_SPANS_TABLE,
            MLFLOW_TRACING_DELTA_ARCHIVAL_TOKEN,
            MLFLOW_TRACING_DELTA_ARCHIVAL_INGESTION_URL,
            MLFLOW_TRACING_DELTA_ARCHIVAL_WORKSPACE_URL,
        ]
        
        for var in required_vars:
            if not var.get():
                _logger.warning(
                    f"Delta archival is enabled but {var.name} is not set. "
                    "Disabling delta archival."
                )
                return False
        
        return True

    def _get_delta_exporter(self) -> Optional[DatabricksDeltaExporter]:
        """
        Initialize Databricks Delta archival by creating a DatabricksDeltaExporter.
        
        Returns:
            DatabricksDeltaExporter instance if successful, None if failed.
        """
        if not self._should_enable_delta_archival():
            return None

        try:            
            delta_exporter = DatabricksDeltaExporter(
                spans_table_name=MLFLOW_TRACING_DELTA_ARCHIVAL_SPANS_TABLE.get(),
                ingest_url=MLFLOW_TRACING_DELTA_ARCHIVAL_INGESTION_URL.get(),
                workspace_url=MLFLOW_TRACING_DELTA_ARCHIVAL_WORKSPACE_URL.get(),
                token=MLFLOW_TRACING_DELTA_ARCHIVAL_TOKEN.get(),
            )
            _logger.debug("Databricks Delta archival initialized successfully")
            return delta_exporter
        except Exception as e:
            _logger.warning(
                f"Failed to initialize Databricks Delta exporter: {e}. "
                "Disabling delta archival."
            )
            return None

    def _log_trace_to_delta(self, trace: Trace):
        """
        Delegate trace export to Databricks Delta via DatabricksDeltaExporter.
        
        Args:
            trace: MLflow Trace object containing spans data.
        """
        if self._delta_exporter is None:
            _logger.debug("Delta exporter not initialized, skipping delta archival")
            return
            
        try:
            # Delegate to the DatabricksDeltaExporter's _log_trace method
            self._delta_exporter._log_trace(trace)
        except Exception as e:
            _logger.warning(f"Failed to send trace to Databricks Delta: {e}")

