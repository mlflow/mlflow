import logging
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.deployments import get_deploy_client
from mlflow.tracing.destination import TraceDestination
from mlflow.tracing.trace_manager import InMemoryTraceManager

_logger = logging.getLogger(__name__)


class DatabricksAgentSpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to Databricks Agent Monitoring.

    Args:
        trace_destination: The destination of the traces.

    TODO: This class will be migrated under databricks-agents package.
    """

    def __init__(self, trace_destination: TraceDestination):
        self._databricks_monitor_id = trace_destination.databricks_monitor_id
        self._trace_manager = InMemoryTraceManager.get_instance()
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

            # Traces are exported via a serving endpoint that accepts trace JSON as
            # an input payload, and then will be written to the Inference Table.
            self._deploy_client.predict(
                endpoint=self._databricks_monitor_id,
                inputs={"inputs": [trace.to_json()]},
            )
