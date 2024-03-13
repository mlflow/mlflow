import json
import logging
import threading
from contextvars import ContextVar
from typing import Any, Dict, Optional, Sequence

from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.tracing.client import TraceClient
from mlflow.tracing.types.model import Span, Trace, TraceData, TraceInfo
from mlflow.tracing.types.wrapper import MLflowSpanWrapper

_logger = logging.getLogger(__name__)


class MLflowSpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to MLflow.

    MLflow backend (will) only support logging the complete trace, not incremental updates
    for spans, so this exporter is designed to aggregate the spans into traces in memory.
    Therefore, this only works within a single process application and not intended to work
    in a distributed environment. For the same reason, this exporter should only be used with
    SimpleSpanProcessor.

    If we want to support distributed tracing, we should first implement an incremental trace
    logging in MLflow backend, then we can get rid of the in-memory trace aggregation.
    """

    _MAX_CHARS_IN_TRACE_INFO_INPUTS_OUTPUTS = 300  # TBD

    def __init__(self, client: TraceClient):
        self._client = client
        self._trace_aggregator = InMemoryTraceDataAggregator.get_instance()

    def export(self, spans: Sequence[MLflowSpanWrapper]):
        """
        Export the spans to MLflow backend.

        Args:
            spans: A sequence of MLflowSpanWrapper objects to be exported. The base
                OpenTelemetry (OTel) exporter should take OTel spans but this exporter
                takes the wrapper object, so we can carry additional MLflow-specific
                information such as inputs and outputs.
        """
        mlflow_spans = []
        for span in spans:
            if not isinstance(span, MLflowSpanWrapper):
                _logger.warning(
                    "Span exporter expected MLflowSpanWrapper, but got "
                    f"{type(span)}. Skipping the span."
                )
            mlflow_span = span._to_mlflow_span()
            self._trace_aggregator.add_span(mlflow_span)
            mlflow_spans.append(mlflow_span)

        # Call this after processing all spans because the parent-child order might
        # not be preserved in the input spans
        for span in mlflow_spans:
            if span.parent_span_id is None:
                self._export_trace(span)

    def _export_trace(self, root_span: Span):
        trace_data = self._trace_aggregator.pop_trace(root_span.context.trace_id)
        if trace_data is None:
            _logger.warning(f"Trace data with ID {root_span.context.trace_id} not found.")
            return

        # Create a TraceInfo object from the root span information
        trace_info = TraceInfo(
            trace_id=root_span.context.trace_id,
            name=root_span.name,
            start_time=root_span.start_time,
            end_time=root_span.end_time,
            status=root_span.status,
            inputs=self._serialize_inputs_outputs(root_span.inputs),
            outputs=self._serialize_inputs_outputs(root_span.outputs),
            # TODO: These fields should be set if necessary
            metadata={},
            tags={},
            source=None,
        )
        trace = Trace(trace_info, trace_data)
        self._client.log_trace(trace)

    def _serialize_inputs_outputs(self, input_or_output: Optional[Dict[str, Any]]) -> str:
        """
        Serialize inputs or outputs field of the span to a string, and truncate if necessary.
        """
        if not input_or_output:
            return ""

        try:
            serialized = json.dumps(input_or_output)
        except TypeError:
            # If not JSON-serializable, use string representation
            serialized = str(input_or_output)

        return serialized[: self._MAX_CHARS_IN_TRACE_INFO_INPUTS_OUTPUTS]


class InMemoryTraceDataAggregator:
    """
    Simple in-memory store for trace_id -> TraceData (i.e. spans).
    """

    _instance_lock = threading.Lock()
    _instance = ContextVar("InMemoryTraceDataAggregator", default=None)

    @classmethod
    def get_instance(cls):
        if cls._instance.get() is None:
            with cls._instance_lock:
                if cls._instance.get() is None:
                    cls._instance.set(InMemoryTraceDataAggregator())
        return cls._instance.get()

    def __init__(self):
        self._traces: Dict[str:TraceData] = {}
        self._lock = threading.Lock()  # Lock for _traces

    def add_span(self, span: Span):
        if not isinstance(span, Span):
            _logger.warning(f"Invalid span object {type(span)} is passed. Skipping.")
            return

        trace_id = span.context.trace_id
        if trace_id not in self._traces:
            with self._lock:
                if trace_id not in self._traces:
                    # NB: the first span might not be a root span, so we can only
                    # set trace_id here. Other information will be propagated from
                    # the root span when it ends.
                    self._traces[trace_id] = TraceData([])

        trace_data = self._traces[trace_id]
        trace_data.spans.append(span)

    def pop_trace(self, trace_id) -> Optional[TraceData]:
        with self._lock:
            return self._traces.pop(trace_id, None)

    def flush(self):
        """Clear all the aggregated trace data. This should only be used for testing."""
        with self._lock:
            self._traces.clear()
