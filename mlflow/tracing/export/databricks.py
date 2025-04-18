import logging
from typing import Sequence

from google.protobuf.json_format import MessageToDict
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.trace import Trace
from mlflow.environment_variables import (
    MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT,
    MLFLOW_ENABLE_ASYNC_TRACE_LOGGING,
)
from mlflow.protos.databricks_trace_server_pb2 import CreateTrace, DatabricksTracingServerService
from mlflow.tracing.export.async_export_queue import AsyncTraceExportQueue, Task
from mlflow.tracing.fluent import _set_last_active_trace_id
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    extract_api_info_for_service,
    http_request,
)

_logger = logging.getLogger(__name__)


_METHOD_TO_INFO = extract_api_info_for_service(
    DatabricksTracingServerService, _REST_API_PATH_PREFIX
)


class DatabricksSpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to Databricks Tracing Server.
    """

    def __init__(self):
        self._is_async = MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.get()
        if self._is_async:
            _logger.info("MLflow is configured to log traces asynchronously.")
            self._async_queue = AsyncTraceExportQueue()

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

            trace = InMemoryTraceManager.get_instance().pop_trace(span.context.trace_id)
            if trace is None:
                _logger.debug(f"Trace for span {span} not found. Skipping export.")
                continue

            _set_last_active_trace_id(trace.info.request_id)

            if self._is_async:
                self._async_queue.put(
                    task=Task(
                        handler=self._log_trace,
                        args=(trace,),
                        error_msg="Failed to log trace to the trace server.",
                    )
                )
            else:
                self._log_trace(trace)

    def _log_trace(self, trace: Trace):
        """Create a new Trace record in the Databricks Tracing Server."""
        request_body = MessageToDict(trace.to_proto(), preserving_proto_field_name=True)
        endpoint, method = _METHOD_TO_INFO[CreateTrace]

        # NB: Using Databricks SDK's built-in retry logic, which simply retries until the timeout
        #    is reached, with linearly increasing backoff. Since it doesn't expose additional
        #    configuration options, we might want to implement our own retry logic in the future.
        # NB: If async logging is disabled, we don't retry to avoid blocking the application.
        timeout = MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT.get() if self._is_async else 0

        # Use context manager to ensure the request is closed properly
        with http_request(
            host_creds=get_databricks_host_creds(),
            endpoint=endpoint,
            method=method,
            json=request_body,
            retry_timeout_seconds=timeout,
        ) as res:
            if res.status_code != 200:
                _logger.warning(f"Failed to log trace to the trace server. Response: {res.text}")
