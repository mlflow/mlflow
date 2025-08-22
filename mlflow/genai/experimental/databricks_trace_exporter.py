import atexit
import json
import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Sequence

from cachetools import TTLCache

from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.trace import Trace
from mlflow.genai.experimental.databricks_trace_exporter_utils import (
    DatabricksTraceServerClient,
    create_archival_zerobus_sdk,
    import_zerobus_sdk_classes,
)
from mlflow.genai.experimental.databricks_trace_otel_pb2 import Span as DeltaProtoSpan
from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter
from mlflow.utils.annotations import experimental

try:
    from opentelemetry.sdk.trace import ReadableSpan

    from mlflow.tracing.export.inference_table import InferenceTableSpanExporter
except ImportError:
    InferenceTableSpanExporter = None
    ReadableSpan = None

_logger = logging.getLogger(__name__)

TRACE_STORAGE_CONFIG_CACHE_TTL_SECONDS = 300  # Cache experiment configs for 5 minutes

if TYPE_CHECKING:
    try:
        TableProperties, _ = import_zerobus_sdk_classes()
    except ImportError:
        # When zerobus_sdk is not available, create a placeholder for type annotations
        from typing import Any

        TableProperties = Any


class DatabricksDeltaArchivalMixin:
    """
    Mixin class that provides Databricks Delta archiving functionality for span exporters.
    This mixin can be used with any span exporter to add Delta table archiving capabilities.
    """

    # Class-level cache for experiment configs
    _config_cache = TTLCache(maxsize=100, ttl=TRACE_STORAGE_CONFIG_CACHE_TTL_SECONDS)
    _config_cache_lock = threading.Lock()

    def archive_trace(self, trace: Trace):
        """
        Try to export a trace to Databricks Unity Catalog if a storage location is set
        for the experiment.
        This method handles all enablement checking, configuration resolution, and authentication.

        Args:
            trace: MLflow Trace object containing spans data.
        """
        try:
            # Extract experiment ID from trace location
            experiment_id = None
            if (
                trace.info.trace_location
                and trace.info.trace_location.type == "MLFLOW_EXPERIMENT"
                and trace.info.trace_location.mlflow_experiment
            ):
                experiment_id = trace.info.trace_location.mlflow_experiment.experiment_id

            if not experiment_id:
                _logger.debug("No experiment ID found in trace, skipping delta archival")
                return

            # Check if archival is configured for this experiment (with caching)
            config = self._get_trace_storage_config(experiment_id)
            if config is None:
                _logger.debug(
                    f"No storage location configured for experiment {experiment_id}, skipping."
                )
                return

            # Store configuration for this export
            self._spans_table_name = config.spans_table_name

            TableProperties, _ = import_zerobus_sdk_classes()
            self._spans_table_properties = TableProperties(
                config.spans_table_name, DeltaProtoSpan.DESCRIPTOR
            )

            # Run the trace logging
            self._archive_trace(trace, experiment_id, config.spans_table_name)

        except Exception as e:
            _logger.warning(f"Failed to export trace to Databricks Delta: {e}")

    def _get_trace_storage_config(self, experiment_id: str):
        """
        Get configuration for experiment, use a cache with TTL.

        Args:
            experiment_id: The MLflow experiment ID to get config for

        Returns:
            DatabricksTraceDeltaStorageConfig if archival is enabled, None otherwise
        """
        with self._config_cache_lock:
            if experiment_id not in self._config_cache:
                config = DatabricksTraceServerClient().get_trace_destination(experiment_id)
                self._config_cache[experiment_id] = config
                _logger.debug(f"Cached config for experiment {experiment_id}: {config is not None}")
            return self._config_cache[experiment_id]

    def _archive_trace(self, trace: Trace, experiment_id: str, spans_table_name: str):
        """
        Handles exporting a trace to Databricks Delta using the Zerobus.

        Args:
            trace: MLflow Trace object containing spans data.
            experiment_id: ID of the experiment for logging.
            spans_table_name: Name of the spans table for logging.
        """
        try:
            # Convert MLflow trace to OTel proto spans
            proto_spans = self._convert_trace_to_proto_spans(trace)

            if not proto_spans or len(proto_spans) == 0:
                _logger.debug("No proto spans to export")
                return

            # Get stream factory singleton for this table
            factory = ZerobusStreamFactory.get_instance(self._spans_table_properties)

            # Get stream from factory
            stream = factory.get_or_create_stream()

            # Ingest all spans for the trace
            _logger.debug(
                f"Ingesting {len(proto_spans)} spans for trace {trace.info.request_id} to table "
                f"{spans_table_name}"
            )
            for proto_span in proto_spans:
                stream.ingest_record(proto_span)

            # Always flush() to ensure data durability.  In sync logging mode, this is ok.
            # For async mode, the flush will be handled by the async queue
            # which runs in the background.
            stream.flush()

        except Exception as e:
            _logger.warning(f"Failed to send trace to Databricks Delta: {e}")

    def _convert_trace_to_proto_spans(self, trace: Trace) -> "list[DeltaProtoSpan]":
        """
        Convert an MLflow trace to a list of Delta Proto Spans.
        Direct conversion from MLflow Span to Delta format.

        Args:
            trace: MLflow Trace object containing spans data.

        TODO: move this to Span.to_proto() once the legacy trace server span are fully deprecated

        Returns:
            List of DeltaProtoSpan objects.
        """
        delta_proto_spans = []

        for span in trace.data.spans:
            # Skip spans that have no span ID since it will break parent-child relationships
            if span.span_id is None:
                _logger.debug(f"Span {span.name} has no span ID, skipping")
                continue

            delta_proto = DeltaProtoSpan()

            # Use raw OpenTelemetry trace ID instead of the one from the trace
            # (without "tr-" prefix) for full OTel compliance
            delta_proto.trace_id = span._trace_id
            delta_proto.span_id = span.span_id
            delta_proto.parent_span_id = span.parent_id or ""
            delta_proto.trace_state = ""
            delta_proto.flags = 0
            delta_proto.name = span.name

            # MLflow traces do not have `kind`, so we default to INTERNAL
            delta_proto.kind = "INTERNAL"

            # Set timestamps (convert to nanoseconds if needed)
            current_time_ns = int(time.time() * 1e9)
            delta_proto.start_time_unix_nano = getattr(span, "start_time_ns", current_time_ns)
            end_time_ns = getattr(span, "end_time_ns", None)
            if end_time_ns is None:
                end_time_ns = (
                    delta_proto.start_time_unix_nano + 1000000000
                )  # fallback: 1s after start
            delta_proto.end_time_unix_nano = end_time_ns

            # the raw otel span attributes are already json serialized
            attributes = dict(span._span.attributes)

            for key, value in attributes.items():
                delta_proto.attributes[str(key)] = value

            delta_proto.dropped_attributes_count = 0

            # Convert events directly
            events = getattr(span, "events", []) or []
            for event in events:
                attributes = getattr(event, "attributes", {}) or {}
                if event_timestamp := getattr(event, "timestamp", None):
                    timestamp_ns = int(event_timestamp * 1e3)
                else:
                    timestamp_ns = current_time_ns
                event_dict = {
                    "time_unix_nano": timestamp_ns,
                    "name": getattr(event, "name", "event"),
                    # serialize attribute values to json string
                    # custom logic to prevent double quotes for string and datetime values
                    "attributes": {
                        k: v
                        if isinstance(v, str)
                        else v.isoformat()
                        if isinstance(v, datetime)
                        else json.dumps(v)
                        for k, v in attributes.items()
                    },
                    "dropped_attributes_count": 0,
                }
                delta_proto.events.append(DeltaProtoSpan.Event(**event_dict))
            delta_proto.dropped_events_count = 0

            # Links are rarely used in MLflow, set to empty
            delta_proto.dropped_links_count = 0

            # Convert status directly
            status_dict = {
                "message": getattr(span.status, "description", "") or "",
                "code": getattr(span.status, "status_code", "UNSET"),
            }
            delta_proto.status.CopyFrom(DeltaProtoSpan.Status(**status_dict))

            delta_proto_spans.append(delta_proto)

        return delta_proto_spans


# TODO: update experimental version number before merging
@experimental(version="3.2.0")
class MlflowV3DeltaSpanExporter(MlflowV3SpanExporter, DatabricksDeltaArchivalMixin):
    """
    An exporter implementation that extends the standard MLflow V3 span export functionality
    to additionally archive traces to Databricks Delta tables when databricks-agents is available.

    This exporter provides the same core functionality as MlflowV3SpanExporter but with
    additional Databricks Delta archiving capabilities for long-term trace storage and analysis.
    """

    def __init__(self, tracking_uri: str | None = None):
        super().__init__(tracking_uri)

        # Delta archiving functionality is now provided by the mixin

    def _log_trace(self, trace: Trace, prompts: Sequence[PromptVersion]):
        """
        Handles exporting a trace via the MlflowV3SpanExporter with additional archiving to
        Databricks Delta tables.
        """
        # Call parent implementation for existing MlflowV3SpanExporter functionality
        super()._log_trace(trace, prompts)

        # Archive trace to Delta table using mixin functionality
        try:
            self.archive_trace(trace)
        except Exception as e:
            _logger.warning(f"Failed to archive trace to Databricks Delta: {e}")


# TODO: update experimental version number before merging
@experimental(version="3.2.0")
class InferenceTableDeltaSpanExporter(InferenceTableSpanExporter, DatabricksDeltaArchivalMixin):
    """
    An exporter implementation that extends the standard InferenceTableSpanExporter functionality
    to additionally archive traces to Databricks Delta tables when databricks-agents is available.

    This exporter provides the same core functionality as InferenceTableSpanExporter but with
    additional Databricks Delta archiving capabilities for long-term trace storage and analysis.
    """

    def __init__(self):
        super().__init__()

    def _export_trace(self, trace: Trace, manager_trace):
        """
        Export a single trace to the Inference Table buffer and archive to Databricks Delta.

        Args:
            trace: The MLflow Trace object to export.
            manager_trace: The manager trace object containing prompts and other metadata.
        """
        # Call parent implementation for inference table export
        super()._export_trace(trace, manager_trace)

        # Archive trace to Delta table using mixin functionality
        try:
            self.archive_trace(trace)
        except Exception as e:
            _logger.warning(f"Failed to archive trace to Databricks Delta: {e}")


class ZerobusStreamFactory:
    """
    Factory for creating and managing Zerobus streams with caching and automatic recovery.
    Simplified approach that flushes on each trace ingestion.
    """

    # Class-level singleton registry: table_name -> factory instance
    _instances: dict[str, "ZerobusStreamFactory"] = {}
    _instances_lock = threading.Lock()
    _atexit_registered = False

    @classmethod
    def get_instance(cls, table_properties: "TableProperties") -> "ZerobusStreamFactory":
        """
        Get or create a singleton factory instance for the given table.

        Args:
            table_properties: TableProperties for the target table

        Returns:
            ZerobusStreamFactory instance for the table
        """
        table_name = table_properties.table_name
        if table_name not in cls._instances:
            with cls._instances_lock:
                # Double-checked locking pattern
                if table_name not in cls._instances:
                    cls._instances[table_name] = cls(table_properties)

                    # Register atexit handler to ensure that all streams are properly closed
                    if not cls._atexit_registered:
                        atexit.register(cls.reset)
                        cls._atexit_registered = True
                        _logger.debug("Registered atexit handler for ZerobusStreamFactory cleanup")
        return cls._instances[table_name]

    def __init__(self, table_properties: "TableProperties"):
        """
        Initialize factory with table properties.
        Encapsulates SDK creation and stream lifecycle management.

        Args:
            table_properties: TableProperties for the target table
        """
        self.table_properties = table_properties

        # Initialize thread-local storage for streams
        self._thread_local = threading.local()

    def get_or_create_stream(self):
        """
        Factory method: Get or create a cached stream for current thread.

        Returns:
            A ZerobusStream instance ready for use
        """
        current_time = time.time()
        stream_cache = getattr(self._thread_local, "stream_cache", None)

        # Check if we have a cached stream
        if stream_cache and "stream" in stream_cache:
            stream = stream_cache["stream"]

            _, StreamState = import_zerobus_sdk_classes()

            # Invalidate the bad stream from the cache if stream is in a valid state
            if stream.get_state() not in [StreamState.OPENED, StreamState.FLUSHING]:
                _logger.debug(f"Stream in invalid state {stream.get_state()}, creating new stream")
                self._thread_local.stream_cache = None

        # Return the valid stream if it wasn't invalidated
        if hasattr(self._thread_local, "stream_cache") and self._thread_local.stream_cache:
            return stream
        else:
            _logger.debug("Creating new thread-local stream for Databricks Delta export")
            zerobus_sdk = create_archival_zerobus_sdk()
            new_stream = zerobus_sdk.create_stream(self.table_properties)

            # Store with metadata
            self._thread_local.stream_cache = {"stream": new_stream, "created_at": current_time}

            return new_stream

    def shutdown(self):
        """
        Factory cleanup - close all created streams.
        """
        try:
            # Close cached streams if they exist
            if hasattr(self._thread_local, "stream_cache"):
                stream_cache = self._thread_local.stream_cache
                if stream_cache and "stream" in stream_cache:
                    try:
                        stream = stream_cache["stream"]
                        stream.close()
                    except Exception as e:
                        _logger.debug(f"Error during stream cleanup: {e}")
                    finally:
                        self._thread_local.stream_cache = None

        except Exception as e:
            _logger.warning(f"Error shutting down ZerobusStreamFactory: {e}")

    @classmethod
    def reset(cls):
        """
        Reset all factory instances. Used for testing and cleanup.
        """
        with cls._instances_lock:
            for factory in cls._instances.values():
                factory.shutdown()
            cls._instances.clear()
            _logger.debug("Reset all ZerobusStreamFactory instances")
