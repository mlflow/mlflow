import logging
from typing import Dict, Sequence
from mlflow.traces.client import TraceClient
from mlflow.traces.types import MLflowSpanWrapper, Trace, TraceData, TraceInfo

from opentelemetry.sdk.trace.export import SpanExporter


_logger = logging.getLogger(__name__)

class TraceAggregator:
    # Simple in-memory store for trace_id -> Trace
    def __init__(self):
        self._traces: Dict[str] = {}

    def create_empty_trace(self, trace_id: str):
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_name=None,
            start_time=None,
            end_time=None,
        )
        trace_data = TraceData([])
        self._traces[trace_id] = Trace(trace_info=trace_info, trace_data=trace_data)

    def add_span(self, trace_id, span: MLflowSpanWrapper):
        if trace_id not in self._traces:
            _logger.warning(f"Trace with ID {trace_id} not found. Skipping the span.")
            return

        self._traces[trace_id].trace_data.spans.append(span)

    def contains(self, trace_id):
        return trace_id in self._traces

    def pop_trace(self, trace_id):
        return self._traces.pop(trace_id, None)



class MLflowSpanExporter(SpanExporter):
    def __init__(self, client: TraceClient, trace_aggregator: TraceAggregator):
        self._client = client
        self._traces = trace_aggregator

    def export(self, spans: Sequence[MLflowSpanWrapper]):
        for span in spans:
            self._add_span_to_trace(span)

        # Call this after processing all spans because the parent-child order might
        # not be preserved in the input spans
        for span in spans:
            if isinstance(span, MLflowSpanWrapper) and span.is_root():
                self._export_trace(span)

    def _add_span_to_trace(self, span: MLflowSpanWrapper):
        if not isinstance(span, MLflowSpanWrapper):
            _logger.warning("Span exporter expected MLflowSpanWrapper, but got "
                            f"{type(span)}. Skipping the span.")
            return

        trace_id = span.context.trace_id
        mlflow_span = span.to_mlflow_span()
        if not self._traces.contains(trace_id):
            # Note that the span might not be a root span, so we can only
            # set trace_id here.
            self._traces.create_empty_trace(trace_id)

        self._traces.add_span(trace_id, mlflow_span)


    def _export_trace(self, root_span: MLflowSpanWrapper):
        trace = self._traces.pop_trace(root_span.trace_id)
        if trace is None:
            _logger.warning(f"Trace with ID {root_span.trace_id} not found.")
            return

        trace.trace_info.trace_name = root_span.name
        trace.trace_info.start_time = root_span.start_time
        trace.trace_info.end_time = root_span.end_time

        self._client.log_trace(trace)
