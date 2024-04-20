import json
import logging
from collections import Counter
from typing import Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities import TraceData
from mlflow.entities.trace_status import TraceStatus
from mlflow.tracing.clients import TraceClient
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.types.constant import (
    MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS,
    TRUNCATION_SUFFIX,
    SpanAttributeKey,
    TraceMetadataKey,
    TraceTagKey,
)

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

    def export(self, spans: Sequence[ReadableSpan]):
        """
        Export the spans to MLflow backend.

        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects to be exported.
        """
        # Exporting the trace when the root span is found. Note that we need to loop over
        # the input list again, because the root span might not be the last span in the list.
        # We must ensure the all child spans are added to the trace before exporting it.
        for span in spans:
            if span._parent is None:
                self._export_trace(span)

    def _export_trace(self, root_span: ReadableSpan):
        request_id = json.loads(root_span.attributes.get(SpanAttributeKey.REQUEST_ID))
        trace = self._trace_manager.pop_trace(request_id)
        if trace is None:
            _logger.debug(f"Trace data with request ID {request_id} not found.")
            return

        # Update a TraceInfo object with the root span information
        trace.info.timestamp_ms = root_span.start_time // 1_000_000  # nanosecond to millisecond
        trace.info.execution_time_ms = (root_span.end_time - root_span.start_time) // 1_000_000
        trace.info.status = TraceStatus.from_otel_status(root_span.status)
        trace.info.request_metadata.update(
            {
                TraceMetadataKey.INPUTS: self._truncate_metadata(
                    root_span.attributes.get(SpanAttributeKey.INPUTS)
                ),
                TraceMetadataKey.OUTPUTS: self._truncate_metadata(
                    root_span.attributes.get(SpanAttributeKey.OUTPUTS)
                ),
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

    def _truncate_metadata(self, value: Optional[str]) -> str:
        """
        Get truncated value of the attribute if it exceeds the maximum length.
        """
        if not value:
            return ""

        if len(value) > MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS:
            trunc_length = MAX_CHARS_IN_TRACE_INFO_METADATA_AND_TAGS - len(TRUNCATION_SUFFIX)
            value = value[:trunc_length] + TRUNCATION_SUFFIX
        return value

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
                span._span._name = f"{span.name}_{count}"
