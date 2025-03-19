import json
import logging
from typing import Sequence

from google.protobuf.json_format import MessageToDict
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.trace import Trace
from mlflow.environment_variables import MLFLOW_HTTP_REQUEST_TIMEOUT
from mlflow.protos.databricks_trace_server_pb2 import CreateTrace, DatabricksTracingServerService
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.export.inference_table import _TRACE_BUFFER
from mlflow.tracing.processor.inference_table import get_databricks_request_id
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

    def __init__(self, is_dual_write_enabled: bool = False):
        # Temporary flag to enable dual write to the inference table
        # TODO: Remove this once the backend is fully migrated to the trace server.
        self._is_dual_write_enabled = is_dual_write_enabled

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

            self.log_trace(trace)

            if self._is_dual_write_enabled:
                self._write_trace_to_inference_table(trace)

    def log_trace(self, trace: Trace):
        """Create a new Trace record in the Databricks Tracing Server."""
        request_body = MessageToDict(trace.to_proto(), preserving_proto_field_name=True)
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

    def _write_trace_to_inference_table(self, trace: Trace):
        """
        TODO: Remove this dual write once the backend is fully migrated
        to the trace server.
        """

        # Overwrite the request_id in the trace and spans to have databricks_request_id
        # because it is the original behavior of the trace in DB model serving
        db_request_id = get_databricks_request_id()

        if db_request_id is None:
            _logger.warning(
                "Failed to get Databricks request ID. Skipping dual write to the inference table."
            )
            return

        trace_dict = trace.to_dict()
        trace_dict["info"]["request_id"] = db_request_id
        for span in trace_dict["data"]["spans"]:
            # NB: All span attribute are stored as json dumped string
            span["attributes"][SpanAttributeKey.REQUEST_ID] = json.dumps(db_request_id)

        _TRACE_BUFFER[db_request_id] = trace_dict
