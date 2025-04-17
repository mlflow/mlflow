import json
import logging
from typing import Optional, Dict

from google.protobuf.duration_pb2 import Duration
from google.protobuf.timestamp_pb2 import Timestamp
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

from mlflow.entities import Trace, TraceInfoV3
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY, SpanAttributeKey
from mlflow.tracing.trace_manager import InMemoryTraceManager, _Trace
from mlflow.tracing.utils import (
    deduplicate_span_names_in_place,
    get_otel_attribute,
    maybe_get_dependencies_schemas,
)
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.fluent import _get_experiment_id

_logger = logging.getLogger(__name__)


class DatabricksSpanProcessor(SimpleSpanProcessor):
    """
    Defines custom hooks to be executed when a span is started or ended (before exporting).

    This process implements simple responsibilities to generate MLflow-style trace
    object from OpenTelemetry spans and store them in memory.
    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        client: Optional[MlflowClient] = None,
        experiment_id: Optional[str] = None,
    ):
        self.span_exporter = span_exporter
        self._client = client or MlflowClient()
        self._trace_manager = InMemoryTraceManager.get_instance()
        self._experiment_id = experiment_id

    def on_start(self, span: OTelSpan, parent_context: Optional[Context] = None):
        """
        Handle the start of a span. This method is called when an OpenTelemetry span is started.

        Args:
            span: An OpenTelemetry Span object that is started.
            parent_context: The context of the span. Note that this is only passed when the context
                object is explicitly specified to OpenTelemetry start_span call. If the parent
                span is obtained from the global context, it won't be passed here so we should not
                rely on it.
        """

        request_id = self._create_or_get_request_id(span)
        span.set_attribute(SpanAttributeKey.REQUEST_ID, json.dumps(request_id))

        tags = {}
        if dependencies_schema := maybe_get_dependencies_schemas():
            tags.update(dependencies_schema)

        if span._parent is None:
            trace_info = TraceInfo(
                request_id=request_id,
                experiment_id=self._experiment_id or _get_experiment_id(),
                timestamp_ms=span.start_time // 1_000_000,  # nanosecond to millisecond
                execution_time_ms=None,
                status=TraceStatus.IN_PROGRESS,
                request_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
                tags=tags,
            )
            self._trace_manager.register_trace(span.context.trace_id, trace_info)

    def _create_or_get_request_id(self, span: OTelSpan) -> str:
        if span._parent is None:
            return str(span.context.trace_id)  # Use otel-generated trace_id as request_id
        else:
            return self._trace_manager.get_request_id_from_trace_id(span.context.trace_id)

    def on_end(self, span: OTelReadableSpan) -> None:
        """
        Handle the end of a span. This method is called when an OpenTelemetry span is ended.

        Args:
            span: An OpenTelemetry ReadableSpan object that is ended.
        """
        # Processing the trace only when it is a root span.
        if span._parent is None:
            request_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)
            with self._trace_manager.get_trace(request_id) as trace:
                if trace is None:
                    _logger.debug(f"Trace data with request ID {request_id} not found.")
                    return

                # Update in-memory trace info
                trace.info.execution_time_ms = (span.end_time - span.start_time) // 1_000_000
                trace.info.status = TraceStatus.from_otel_status(span.status)
                deduplicate_span_names_in_place(list(trace.span_dict.values()))
                
                # Create and send V3 trace to MLflow backend
                self._send_trace_to_mlflow(trace, request_id)
    
    def _send_trace_to_mlflow(self, trace: _Trace, request_id: str) -> None:
        """
        Convert the internal trace format to MLflow's V3 format and send it to the MLflow backend.
        
        Args:
            trace: The trace object from the trace manager
            request_id: The unique identifier for the trace
        """
        # Create a timestamp from trace timestamp
        timestamp = Timestamp()
        timestamp.FromMilliseconds(trace.info.timestamp_ms)
        
        # Create duration from execution time
        duration = Duration()
        duration.FromMilliseconds(trace.info.execution_time_ms or 0)
        
        # Map status from TraceStatus to TraceInfoV3.State
        state_mapping = {
            TraceStatus.OK: TraceInfoV3.State.OK,
            TraceStatus.ERROR: TraceInfoV3.State.ERROR,
            TraceStatus.IN_PROGRESS: TraceInfoV3.State.IN_PROGRESS,
        }
        state = state_mapping.get(trace.info.status, TraceInfoV3.State.STATE_UNSPECIFIED)
        
        # Build metadata dictionary from trace info
        metadata: Dict[str, str] = {}
        for meta in trace.info.request_metadata:
            metadata[meta.key] = meta.value
        
        # Build tags dictionary from trace info
        tags: Dict[str, str] = {}
        for tag in trace.info.tags:
            tags[tag.key] = tag.value
        
        # Create TraceInfoV3 object
        trace_info_v3 = TraceInfoV3(
            trace_id=request_id,
            client_request_id=request_id,
            request_time=timestamp,
            execution_duration=duration,
            state=state,
            trace_metadata=metadata,
            tags=tags
        )
        
        # Create Trace object and send to MLflow backend
        mlflow_trace = Trace(trace_info=trace_info_v3)
        try:
            self._client._start_trace_v3(mlflow_trace)
        except Exception as e:
            _logger.warning(f"Failed to send trace to MLflow backend: {e}")
