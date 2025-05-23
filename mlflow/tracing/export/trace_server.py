import asyncio
import json
import logging
import time
import uuid
from typing import Sequence, Dict, Any, Optional

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.trace import Trace
from mlflow.environment_variables import (
    MLFLOW_ENABLE_ASYNC_TRACE_LOGGING,
)
from mlflow.tracing.export.trace_server_archival_pb2 import Span as ProtoSpan
from mlflow.tracing.export.async_export_queue import AsyncTraceExportQueue, Task
from mlflow.tracing.fluent import _set_last_active_trace_id
from mlflow.tracing.trace_manager import InMemoryTraceManager

from ingest_api_sdk import IngestApiSdk, TableProperties

_logger = logging.getLogger(__name__)


class TraceServerSpanExporter(SpanExporter):
    """
    An exporter implementation that sends OpenTelemetry spans to the Databricks Trace Server
    using the IngestApi.
    """

    def __init__(
        self,
        spans_table_name: str,
        ingest_url: str,
        workspace_url: str,
        pat: str,
    ):
        """
        Initialize a new TraceServerSpanExporter.
        
        Args:
            spans_table_name: The name of the table to ingest spans into.
            ingest_url: The URL of the ingest API.
            workspace_url: The URL of the Databricks workspace.
            pat: The personal access token for authentication.
        """
        self._spans_table_name = spans_table_name
        self._spans_table_properties = TableProperties(spans_table_name, ProtoSpan.DESCRIPTOR)
        
        # Initialize IngestApiSdk
        self._sdk_handle = IngestApiSdk(
            ingest_url,
            workspace_url,
            pat
        )
        
        # Set up async handling if enabled
        self._is_async = MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.get()
        if self._is_async:
            _logger.info("MLflow is configured to log traces asynchronously.")
            self._async_queue = AsyncTraceExportQueue()
        
        # Initialize an event loop for the exporter
        self._loop = asyncio.new_event_loop()

        # Initialize stream (will be created lazily when needed)
        self._stream = None

    def export(self, spans: Sequence[ReadableSpan]):
        """
        Export the spans to the Databricks Trace Server.

        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects passed from
                a span processor. Only root spans for each trace should be exported.
        """
        for span in spans:
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
        """
        Handles exporting a trace to the Databricks Trace Server using the IngestApi.
        
        Args:
            trace: MLflow Trace object containing spans data.
        """
        try:
            if not trace:
                _logger.warning("No trace provided, unable to export")
                return
                
            # Convert MLflow trace to OTel proto spans
            proto_spans = self._convert_trace_to_proto_spans(trace)
            
            # Run the async ingest function in the event loop
            self._loop.run_until_complete(self._ingest_spans(proto_spans))
                
        except Exception as e:
            import traceback
            _logger.warning(f"Failed to send trace to Databricks Trace Server: {e}")
            _logger.warning(f"Stack trace: {traceback.format_exc()}")

    async def _get_stream(self):
        """
        Get the stream for ingesting spans, creating it if it doesn't exist.
        
        Returns:
            A stream object for ingesting records.
        """
        if self._stream is None:
            self._stream = await self._sdk_handle.create_stream(self._spans_table_properties)
        return self._stream

    async def _ingest_spans(self, proto_spans: list[ProtoSpan]):
        """
        Ingest all spans from a trace into the trace server in a batch.
        
        Args:
            proto_spans: A list of ProtoSpan objects to be ingested.
        """
        if not proto_spans:
            return
            
        try:
            # Get or create stream to table
            stream = await self._get_stream()
            
            # Ingest all spans for the trace
            for proto_span in proto_spans:
                await stream.ingest_record(proto_span)
            
            # Wait until we receive the ack for all records
            await stream.flush()
            
        except Exception as e:
            _logger.warning(f"Failed to ingest spans: {e}")
            # If we encounter an error with the stream, reset it so we'll create a new one next time
            self._stream = None

    def _convert_trace_to_proto_spans(self, trace: Trace) -> list[ProtoSpan]:
        """
        Convert an MLflow trace to a list of OpenTelemetry Proto Spans.
        
        Args:
            trace: MLflow Trace object containing spans data.
            
        Returns:
            List of ProtoSpan objects.
        """
        proto_spans = []
        
        for span in trace.data.spans:
            current_time_ns = int(time.time() * 1e9)
            
            # Create a new ProtoSpan
            proto_span = ProtoSpan()
            
            # Set trace and span IDs
            proto_span.trace_id = trace.info.request_id
            proto_span.span_id = span.span_id or str(uuid.uuid4())
            proto_span.trace_state = ""
            proto_span.parent_span_id = span.parent_id or ""
            proto_span.flags = 0
            
            # Set span properties
            proto_span.name = span.name
            
            # Map span kind
            kind_mapping = {
                "internal": "INTERNAL",
                "server": "SERVER",
                "client": "CLIENT",
                "producer": "PRODUCER",
                "consumer": "CONSUMER"
            }
            proto_span.kind = kind_mapping.get(getattr(span, "kind", "internal").lower(), "INTERNAL")
            
            # Set timestamps
            start_time_ns = getattr(span, "start_time_ns", current_time_ns)
            end_time_ns = getattr(span, "end_time_ns", None)
            if end_time_ns is None:
                end_time_ns = start_time_ns + 1000000000  # fallback: 1s after start

            proto_span.start_time_unix_nano = start_time_ns
            proto_span.end_time_unix_nano = end_time_ns
            
            # Convert attributes to JSON string
            proto_span.attributes = json.dumps(getattr(span, "attributes", {}))
            proto_span.dropped_attributes_count = 0
            
            # Convert events to JSON string
            events = []
            for event in getattr(span, "events", []):
                events.append({
                    "time_unix_nano": getattr(event, "timestamp_ns", current_time_ns),
                    "name": getattr(event, "name", "event"),
                    "attributes": getattr(event, "attributes", {}),
                    "dropped_attributes_count": 0
                })
            proto_span.events = json.dumps(events)
            proto_span.dropped_events_count = 0
            
            # Set empty links
            proto_span.links = json.dumps([])
            proto_span.dropped_links_count = 0
            
            # Set status
            status_code = "UNSET"
            if hasattr(span, "status_code"):
                status_code = span.status_code
            elif hasattr(span, "status"):
                status = span.status
                if isinstance(status, str):
                    status_code = status
                elif hasattr(status, "status_code"):
                    status_code = status.status_code
            
            proto_span.status_code = status_code
            proto_span.status_message = getattr(span, "status_message", "")
            
            proto_spans.append(proto_span)
            
        return proto_spans
        
    def shutdown(self):
        """
        Shutdown the exporter, closing the stream and event loop.
        """
        try:
            if self._stream is not None:
                # Close the stream in the event loop
                self._loop.run_until_complete(self._stream.close())
                self._stream = None
                
            if not self._loop.is_closed():
                self._loop.close()
        except Exception as e:
            _logger.warning(f"Error shutting down exporter: {e}") 