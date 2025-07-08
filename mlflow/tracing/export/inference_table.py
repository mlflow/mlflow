import logging
from typing import Any, Optional, Sequence

from cachetools import TTLCache
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.trace import Trace
from mlflow.environment_variables import (
    _MLFLOW_ENABLE_TRACE_DUAL_WRITE_IN_MODEL_SERVING,
    MLFLOW_TRACE_BUFFER_MAX_SIZE,
    MLFLOW_TRACE_BUFFER_TTL_SECONDS,
)
from mlflow.tracing.client import TracingClient
from mlflow.tracing.export.async_export_queue import AsyncTraceExportQueue, Task
from mlflow.tracing.export.utils import try_link_prompts_to_trace
from mlflow.tracing.fluent import _set_last_active_trace_id
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import add_size_bytes_to_trace_metadata

_logger = logging.getLogger(__name__)


def pop_trace(request_id: str) -> Optional[dict[str, Any]]:
    """
    Pop the completed trace data from the buffer. This method is used in
    the Databricks model serving so please be careful when modifying it.
    """
    return _TRACE_BUFFER.pop(request_id, None)


# For Inference Table, we use special TTLCache to store the finished traces
# so that they can be retrieved by Databricks model serving. The values
# in the buffer are not Trace dataclass, but rather a dictionary with the schema
# that is used within Databricks model serving.
def _initialize_trace_buffer():  # Define as a function for testing purposes
    return TTLCache(
        maxsize=MLFLOW_TRACE_BUFFER_MAX_SIZE.get(),
        ttl=MLFLOW_TRACE_BUFFER_TTL_SECONDS.get(),
    )


_TRACE_BUFFER = _initialize_trace_buffer()


class InferenceTableSpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to Inference Table.

    Currently the Inference Table does not use collector to receive the traces,
    but rather actively fetches the trace during the prediction process. In the
    future, we may consider using collector-based approach and this exporter should
    send the traces instead of storing them in the buffer.
    """

    def __init__(self):
        self._trace_manager = InMemoryTraceManager.get_instance()

        # NB: When this env var is set to true, MLflow will export traces to both inference
        #  table and the Databricks Tracing Server.
        self._should_write_to_mlflow_backend = (
            _MLFLOW_ENABLE_TRACE_DUAL_WRITE_IN_MODEL_SERVING.get()
        )
        if self._should_write_to_mlflow_backend:
            self._client = TracingClient("databricks")
            self._async_queue = AsyncTraceExportQueue()

    def export(self, spans: Sequence[ReadableSpan]):
        """
        Export the spans to Inference Table via the TTLCache buffer.

        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects passed from
                a span processor. Only root spans for each trace should be exported.
        """
        for span in spans:
            if span._parent is not None:
                _logger.debug("Received a non-root span. Skipping export.")
                continue

            manager_trace = self._trace_manager.pop_trace(span.context.trace_id)
            if manager_trace is None:
                _logger.debug(f"Trace for span {span} not found. Skipping export.")
                continue

            trace = manager_trace.trace
            _set_last_active_trace_id(trace.info.trace_id)

            # Add the trace to the in-memory buffer so it can be retrieved by upstream
            # The key is Databricks request ID.
            _TRACE_BUFFER[trace.info.client_request_id] = trace.to_dict()

            if self._should_write_to_mlflow_backend:
                if trace.info.experiment_id is None:
                    # NB: The experiment ID is set based on the MLFLOW_EXPERIMENT_ID env var
                    #   populated in the scoring server by Agent Framework. If the model is not
                    #   deployed via agents.deploy(), the env var will not be set and the
                    #   experiment will be empty, even if the dual write itself is enabled.
                    _logger.warning(
                        "Dual write to MLflow backend is enabled, but experiment ID is not set "
                        "for the trace. Skipping trace export to MLflow backend."
                    )
                    continue

                try:
                    # Log the trace to the MLflow backend asynchronously
                    self._async_queue.put(
                        task=Task(
                            handler=self._log_trace_to_mlflow_backend,
                            args=(trace, manager_trace.prompts),
                            error_msg=f"Failed to log trace {trace.info.trace_id}.",
                        )
                    )
                except Exception as e:
                    _logger.warning(
                        f"Failed to export trace to MLflow backend. Error: {e}",
                        stack_info=_logger.isEnabledFor(logging.DEBUG),
                    )

    def _log_trace_to_mlflow_backend(self, trace: Trace, prompts: Sequence[PromptVersion]):
        try:
            add_size_bytes_to_trace_metadata(trace)
        except Exception:
            _logger.warning("Failed to add size bytes to trace metadata.", exc_info=True)

        returned_trace_info = self._client.start_trace_v3(trace)
        self._client._upload_trace_data(returned_trace_info, trace.data)

        # Link prompt versions to the trace. Prompt linking is not critical for trace export
        # (if the prompt fails to link, the user's workflow is minorly affected), so we handle
        # errors gracefully without failing the entire trace export
        try_link_prompts_to_trace(
            client=self._client,
            trace_id=returned_trace_info.trace_id,
            prompts=prompts,
            synchronous=True,  # Run synchronously since we're already in an async task
        )
        _logger.debug(
            f"Finished logging trace to MLflow backend. TraceInfo: {returned_trace_info.to_dict()} "
        )
