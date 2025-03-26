import logging
from typing import Sequence

from google.protobuf.json_format import MessageToDict
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.trace import Trace
from mlflow.environment_variables import MLFLOW_HTTP_REQUEST_TIMEOUT
from mlflow.protos.databricks_trace_server_pb2 import CreateTrace, DatabricksTracingServerService
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

# NB: Setting lower default timeout for trace export to avoid blocking the application
# We can increase this value when the trace export is updated to async.
_DEFAULT_TRACE_EXPORT_TIMEOUT = 5


class DatabricksSpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to Databricks Tracing Server.
    """

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

            self._log_trace(trace)

    def _log_trace(self, trace: Trace):
        """Create a new Trace record in the Databricks Tracing Server."""
        request_body = MessageToDict(trace.to_proto(), preserving_proto_field_name=True)
        endpoint, method = _METHOD_TO_INFO[CreateTrace]

        # Use context manager to ensure the request is closed properly
        with http_request(
            host_creds=get_databricks_host_creds(),
            endpoint=endpoint,
            method=method,
            timeout=self._get_timeout(),
            # Not doing reties here because trace export is currently running synchronously
            # and we don't want to bottleneck the application by retrying.
            json=request_body,
        ) as res:
            if res.status_code != 200:
                _logger.warning(f"Failed to log trace to the trace server. Response: {res.text}")

    def _get_timeout(self) -> int:
        if MLFLOW_HTTP_REQUEST_TIMEOUT.get_raw() is not None:
            return int(MLFLOW_HTTP_REQUEST_TIMEOUT.get_raw())
        return _DEFAULT_TRACE_EXPORT_TIMEOUT
