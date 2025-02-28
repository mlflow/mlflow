import logging
import os
from typing import Sequence

from google.protobuf.json_format import MessageToDict
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.deployments import get_deploy_client
from mlflow.entities.trace import Trace
from mlflow.environment_variables import MLFLOW_HTTP_REQUEST_TIMEOUT
from mlflow.protos.databricks_trace_server_pb2 import CreateTrace, DatabricksTracingServerService
from mlflow.tracing.destination import TraceDestination
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


class DatabricksAgentSpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to Databricks Agent Monitoring.

    Args:
        trace_destination: The destination of the traces.

    TODO: This class will be migrated under databricks-agents package.
    """

    def __init__(self, trace_destination: TraceDestination):
        # TODO: Remove this once the new trace server is fully rolled out.
        self._v3_write = (
            os.environ.get("AGENT_EVAL_TRACE_SERVER_ENABLED", "false").lower() == "true"
        )

        self._trace_manager = InMemoryTraceManager.get_instance()
        self._databricks_monitor_id = trace_destination.databricks_monitor_id

        if not self._v3_write:
            self._deploy_client = get_deploy_client("databricks")

    def export(self, root_spans: Sequence[ReadableSpan]):
        """
        Export the spans to the destination.

        Args:
            root_spans: A sequence of OpenTelemetry ReadableSpan objects to be exported.
                Only root spans for each trace are passed to this method.
        """
        for span in root_spans:
            if span._parent is not None:
                _logger.debug("Received a non-root span. Skipping export.")
                continue

            trace = self._trace_manager.pop_trace(span.context.trace_id)
            if trace is None:
                _logger.debug(f"Trace for span {span} not found. Skipping export.")
                continue

            if self._v3_write:
                self._log_trace_v3(trace)
            else:
                self._log_trace_legacy(trace)

    def _log_trace_legacy(self, trace: Trace):
        """
        Export via a serving endpoint that accepts trace JSON as an input payload,
        and then will be written to the Inference Table.
        """
        self._deploy_client.predict(
            endpoint=self._databricks_monitor_id,
            inputs={"inputs": [trace.to_json()]},
        )

    def _log_trace_v3(self, trace: Trace):
        """Create a new Trace record in the Databricks Agent Monitoring."""
        request_body = MessageToDict(
            CreateTrace(trace=trace.to_proto()),
            preserving_proto_field_name=True,
        )
        endpoint, method = _METHOD_TO_INFO[CreateTrace]

        res = http_request(
            host_creds=get_databricks_host_creds(),
            endpoint=endpoint,
            method=method,
            timeout=MLFLOW_HTTP_REQUEST_TIMEOUT.get(),
            # Not doing reties here because trace export is currently running synchronously
            # and we don't want to bottleneck the application by retrying.
            json=request_body,
        )

        if res.status_code != 200:
            _logger.warning(f"Failed to log trace to the trace server. Response: {res.text}")
