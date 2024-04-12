import json
import logging
from collections import Counter
from typing import Any, Optional, Sequence

from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities import TraceData
from mlflow.tracing.clients import TraceClient
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.types.constant import (
    MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS,
    TRUNCATION_SUFFIX,
    TraceMetadataKey,
    TraceTagKey,
)
from mlflow.tracing.types.wrapper import MlflowSpanWrapper

_logger = logging.getLogger(__name__)


class MlflowSpanExporter(SpanExporter):
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

    def __init__(self, client: TraceClient):
        self._client = client
        self._trace_manager = InMemoryTraceManager.get_instance()

    def export(self, spans: Sequence[MlflowSpanWrapper]):
        """
        Export the spans to MLflow backend.

        Args:
            spans: A sequence of MlflowSpanWrapper objects to be exported. The base
                OpenTelemetry (OTel) exporter should take OTel spans but this exporter
                takes the wrapper object, so we can carry additional MLflow-specific
                information such as inputs and outputs.
        """
        for span in spans:
            if not isinstance(span, MlflowSpanWrapper):
                _logger.warning(
                    "Span exporter expected MlflowSpanWrapper, but got "
                    f"{type(span)}. Skipping the span."
                )
                continue

            self._trace_manager.add_or_update_span(span)

        # Exporting the trace when the root span is found. Note that we need to loop over
        # the input list again, because the root span might not be the last span in the list.
        # We must ensure the all child spans are added to the trace before exporting it.
        for span in spans:
            if span.parent_span_id is None:
                self._export_trace(span)

    def _export_trace(self, root_span: MlflowSpanWrapper):
        request_id = root_span.request_id
        trace = self._trace_manager.pop_trace(request_id)
        if trace is None:
            _logger.warning(f"Trace data with ID {request_id} not found.")
            return

        # Update a TraceInfo object with the root span information
        trace.info.timestamp_ms = root_span.start_time // 1_000  # microsecond to millisecond
        trace.info.execution_time_ms = (root_span.end_time - root_span.start_time) // 1_000
        trace.info.status = root_span.status.status_code
        trace.info.request_metadata.update(
            {
                TraceMetadataKey.INPUTS: self._serialize_inputs_outputs(root_span.inputs),
                TraceMetadataKey.OUTPUTS: self._serialize_inputs_outputs(root_span.outputs),
            }
        )
        # Mutable info like trace name should be recorded in tags
        trace.info.tags.update(
            {
                TraceTagKey.TRACE_NAME: root_span.name,
            }
        )

        # Rename spans to have unique names
        MlflowSpanExporter._deduplicate_span_names_in_place(trace.data)

        # TODO: Make this async
        self._client.log_trace(trace)

    def _serialize_inputs_outputs(self, input_or_output: Optional[Any]) -> str:
        """
        Serialize inputs or outputs field of the span to a string, and truncate if necessary.
        """
        if not input_or_output:
            return ""

        try:
            serialized = json.dumps(input_or_output, default=str)
        except TypeError:
            # If not JSON/string serializable, raise a warning and return an empty string
            _logger.warning(
                "Failed to serialize inputs/outputs for a trace, an empty string will be recorded."
            )
            return ""

        if len(serialized) > MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS:
            trunc_length = MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS - len(TRUNCATION_SUFFIX)
            serialized = serialized[:trunc_length] + TRUNCATION_SUFFIX
        return serialized

    @staticmethod
    def _deduplicate_span_names_in_place(trace_data: TraceData):
        """
        Deduplicate span names in the trace data by appending an index number to the span name.

        This is only applied when there are multiple spans with the same name. The span names
        are modified in place to avoid unnecessary copying.

        E.g.
            ["red", "red"] -> ["red_1", "red_2"]
            ["red", "red", "blue"] -> ["red_1", "red_2", "blue"]

        Args:
            trace_data: The trace data object to deduplicate span names.
        """
        span_name_counter = Counter(span.name for span in trace_data.spans)
        # Apply renaming only for duplicated spans
        span_name_counter = {name: 1 for name, count in span_name_counter.items() if count > 1}
        # Add index to the duplicated span names
        for span in trace_data.spans:
            if count := span_name_counter.get(span.name):
                span_name_counter[span.name] += 1
                span.name = f"{span.name}_{count}"
