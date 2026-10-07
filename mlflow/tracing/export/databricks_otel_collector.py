"""Databricks OTel collector span exporter for Unity Catalog trace destinations.

Spans are serialized to OTLP protobuf and sent to the Databricks Zerobus
collector when a Unity Catalog destination has service-principal credentials.
Trace-level metadata continues through the MLflow V3 exporter path.

The collector client is initialized without network I/O. Credential, workspace,
and endpoint discovery happen on the first span export; see
``databricks_otel_client.ZerobusOtelClient`` for the low-level wire contract.
"""

import logging
import threading
from typing import Sequence

import requests
import urllib3.exceptions
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceResponse
from opentelemetry.sdk.trace import ReadableSpan

from mlflow.entities.span import Span
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import UnityCatalog
from mlflow.environment_variables import (
    MLFLOW_ENABLE_ASYNC_TRACE_LOGGING,
    MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT,
)
from mlflow.tracing.export.databricks_otel_client import (
    ZerobusOtelClient,
    ZerobusOtelTokenError,
    ZerobusOtelTokenRefreshError,
)
from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter
from mlflow.tracing.export.span_batcher import SpanBatcher
from mlflow.tracing.export.uc_table import DatabricksUCSpanWriter
from mlflow.tracing.export.utils import flush_exporter
from mlflow.tracing.utils.otlp import build_otlp_export_request

_logger = logging.getLogger(__name__)

# Max body bytes included in a span-export failure warning log.
_BODY_SNIPPET_BYTES = 512

# Collector statuses for which the current batch can be replayed over REST while
# future batches still try the collector.
_REPLAYABLE_STATUS_CODES = (408, 413, 429)


def _is_connection_not_established(exc: BaseException) -> bool:
    """Whether *exc* proves a request was never delivered to the collector."""
    if isinstance(exc, requests.ConnectTimeout):
        return True
    if not isinstance(exc, requests.ConnectionError):
        return False

    stack = [exc]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, urllib3.exceptions.NewConnectionError):
            return True
        if isinstance(current, urllib3.exceptions.MaxRetryError) and isinstance(
            current.reason, BaseException
        ):
            stack.append(current.reason)
        stack.extend(arg for arg in current.args if isinstance(arg, BaseException))
        stack.extend(
            chained for chained in (current.__cause__, current.__context__) if chained is not None
        )
    return False


class DatabricksOtelCollectorSpanExporter(MlflowV3SpanExporter):
    """Export UC table spans through Databricks' OTLP collector when available.

    On ambiguous delivery (5xx, read timeout, or a reset after sending), the
    current batch is replayed through REST and later batches use REST. This may
    duplicate spans if the collector ingested the batch before failing.
    """

    def __init__(
        self,
        tracking_uri: str | None,
        token_source,
        table_name: str,
        host: str | None = None,
        workspace_id: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        super().__init__(tracking_uri=tracking_uri)
        self._tracking_uri = tracking_uri
        self._table_name = table_name
        self._span_writer = DatabricksUCSpanWriter()
        self._zerobus_client = ZerobusOtelClient(
            tracking_uri=tracking_uri,
            token_source=token_source,
            table_name=table_name,
            host=host,
            workspace_id=workspace_id,
            client_id=client_id,
            client_secret=client_secret,
            endpoint=endpoint,
        )

        # Set when the collector path is unusable or a batch's delivery is
        # uncertain. Later batches go straight to the UC REST span writer.
        self._collector_rejected = False
        self._has_warned_ambiguous_drop = False
        self._has_warned_ambiguous_fallback = False
        self._ambiguous_fallback_warning_lock = threading.Lock()

        if hasattr(self, "_async_queue"):
            self._span_batcher = SpanBatcher(
                async_task_queue=self._async_queue,
                log_spans_func=self._export_batch,
            )

    def _log_spans(self, location: str, spans: list[Span]) -> None:
        self._span_writer.log_spans(self._client, location, spans)

    def _should_enable_async_logging(self) -> bool:
        return MLFLOW_ENABLE_ASYNC_TRACE_LOGGING.get()

    def _should_log_spans_to_artifacts(self, trace_info: TraceInfo) -> bool:
        return False

    def _as_mlflow_spans(self, spans: Sequence[ReadableSpan | Span]) -> list[Span]:
        return [span if isinstance(span, Span) else Span(span) for span in spans]

    def _send_spans_via_rest(
        self,
        spans: Sequence[ReadableSpan | Span],
        *,
        from_collector_batch: bool = False,
    ) -> None:
        """Send *spans* to the UC REST writer for this exporter's table."""
        spans = self._as_mlflow_spans(spans)
        if self._should_log_async() and not from_collector_batch:
            for span in spans:
                self._span_batcher.add_span(location=self._table_name, span=span)
        else:
            self._log_spans(self._table_name, spans)

    def _fall_back_to_rest(
        self,
        spans: Sequence[ReadableSpan | Span],
        *,
        from_collector_batch: bool = False,
    ) -> None:
        """Replay a rejected batch and permanently use REST afterwards."""
        if not self._collector_rejected:
            self._collector_rejected = True
            _logger.warning(
                "The Databricks OTel collector rejected a span export. MLflow is falling "
                "back to the tracing server path for this and all subsequent span batches."
            )
        self._send_spans_via_rest(spans, from_collector_batch=from_collector_batch)

    def _replay_batch_via_rest(
        self,
        spans: Sequence[ReadableSpan | Span],
        reason: str,
        *args,
        from_collector_batch: bool = False,
    ) -> None:
        """Replay a transient or batch-specific rejection without sticky fallback."""
        _logger.debug(reason, *args)
        self._send_spans_via_rest(spans, from_collector_batch=from_collector_batch)

    def _log_ambiguous_drop(self, reason: str, *args) -> None:
        """Drop an uncertain batch and route later batches through REST."""
        self._collector_rejected = True
        message = reason + (
            " Delivery is ambiguous, so the spans were dropped instead of replayed over "
            "the MLflow tracing server path, which could duplicate them. Future span "
            "batches will use the tracing server path."
        )
        if self._has_warned_ambiguous_drop:
            _logger.debug(message, *args)
        else:
            _logger.warning(message, *args)
            self._has_warned_ambiguous_drop = True

    def _fall_back_to_rest_after_ambiguous_delivery(
        self,
        spans: Sequence[ReadableSpan | Span],
        reason: str,
        *args,
        from_collector_batch: bool = False,
    ) -> None:
        """Replay an uncertain batch and permanently use REST afterwards."""
        self._collector_rejected = True
        message = reason + (
            " The collector may have already ingested this batch, so replaying it through "
            "the MLflow tracing server path may create duplicate spans. New span "
            "batches will use the tracing server path; concurrent exports already "
            "in progress may still contact the collector."
        )
        with self._ambiguous_fallback_warning_lock:
            already_warned = self._has_warned_ambiguous_fallback
            self._has_warned_ambiguous_fallback = True
        if already_warned:
            _logger.debug(message, *args)
        else:
            _logger.warning(message, *args)
        self._send_spans_via_rest(spans, from_collector_batch=from_collector_batch)

    def _handle_post_exception(
        self,
        exc: BaseException,
        spans: Sequence[ReadableSpan | Span],
        *,
        from_collector_batch: bool = False,
    ) -> None:
        if _is_connection_not_established(exc):
            self._fall_back_to_rest(spans, from_collector_batch=from_collector_batch)
        else:
            self._fall_back_to_rest_after_ambiguous_delivery(
                spans,
                "The Databricks OTel collector span export request failed: %s.",
                exc,
                from_collector_batch=from_collector_batch,
            )

    def _handle_collector_response(
        self,
        response: requests.Response,
        spans: Sequence[ReadableSpan | Span],
        *,
        from_collector_batch: bool,
    ) -> None:
        """Handle a successful OTLP response, including partial success."""
        if not response.content:
            return

        response_message = ExportTraceServiceResponse()
        try:
            response_message.ParseFromString(response.content)
        except Exception as exc:
            self._log_ambiguous_drop(
                "The Databricks OTel collector returned an invalid HTTP 200 response: %s.",
                exc,
            )
            return

        rejected_spans = response_message.partial_success.rejected_spans
        if rejected_spans == 0:
            return
        if rejected_spans < 0 or rejected_spans > len(spans):
            self._log_ambiguous_drop(
                "The Databricks OTel collector reported %d rejected spans for a batch of %d spans.",
                rejected_spans,
                len(spans),
            )
            return
        if rejected_spans == len(spans):
            self._collector_rejected = True
            _logger.warning(
                "The Databricks OTel collector rejected all %d spans in a successful HTTP 200 "
                "partial response (%s). Replaying this batch via the MLflow tracing server "
                "path and using that path for future batches.",
                len(spans),
                response_message.partial_success.error_message,
            )
            self._send_spans_via_rest(spans, from_collector_batch=from_collector_batch)
            return

        self._collector_rejected = True
        _logger.warning(
            "The Databricks OTel collector partially accepted a span batch: %d of %d spans "
            "were rejected (%s). Future batches will use the MLflow tracing server path.",
            rejected_spans,
            len(spans),
            response_message.partial_success.error_message,
        )

    def _export_spans_to_collector(
        self,
        spans: Sequence[ReadableSpan | Span],
        *,
        from_collector_batch: bool = False,
    ) -> None:
        """Serialize and send *spans* to Zerobus, classifying failures for REST replay."""
        if not self._zerobus_client.ensure_ready():
            self._collector_rejected = True
            if not self._zerobus_client.config_warned:
                _log_collector_unavailable(
                    "The Databricks OTel collector endpoint could not be resolved for "
                    "workspace_id=%r, host=%r.",
                    self._zerobus_client.workspace_id,
                    self._zerobus_client.host,
                )
            self._send_spans_via_rest(spans, from_collector_batch=from_collector_batch)
            return

        mlflow_spans = self._as_mlflow_spans(spans)
        payload = build_otlp_export_request(mlflow_spans).SerializeToString()
        try:
            response = self._zerobus_client.post(payload)
        except ZerobusOtelTokenError as exc:
            self._replay_batch_via_rest(
                spans,
                "Minting the Databricks OTel collector token failed: %s. Replaying this "
                "batch via the MLflow tracing server path.",
                exc.__cause__ or exc,
                from_collector_batch=from_collector_batch,
            )
            return
        except ZerobusOtelTokenRefreshError:
            self._fall_back_to_rest(spans, from_collector_batch=from_collector_batch)
            return
        except Exception as exc:
            self._handle_post_exception(exc, spans, from_collector_batch=from_collector_batch)
            return

        # A 401 is handled by ZerobusOtelClient, which performs one forced
        # refresh and retry. A second 401 reaches the normal definitive 4xx path.
        if response.ok:
            if response.status_code == 200:
                self._handle_collector_response(
                    response,
                    spans,
                    from_collector_batch=from_collector_batch,
                )
            return

        if 400 <= response.status_code < 500:
            if response.status_code in _REPLAYABLE_STATUS_CODES:
                self._replay_batch_via_rest(
                    spans,
                    "The Databricks OTel collector returned HTTP %d for a span export "
                    "(transient or batch-specific rejection). Replaying this batch via the "
                    "MLflow tracing server path.",
                    response.status_code,
                    from_collector_batch=from_collector_batch,
                )
            else:
                self._fall_back_to_rest(spans, from_collector_batch=from_collector_batch)
            return

        if 500 <= response.status_code < 600:
            self._fall_back_to_rest_after_ambiguous_delivery(
                spans,
                "The Databricks OTel collector span export failed with HTTP %d: %r.",
                response.status_code,
                response.content[:_BODY_SNIPPET_BYTES],
                from_collector_batch=from_collector_batch,
            )
            return

        self._log_ambiguous_drop(
            "The Databricks OTel collector span export failed with HTTP %d: %r.",
            response.status_code,
            response.content[:_BODY_SNIPPET_BYTES],
        )

    def _export_batch(self, location: str, spans: list[Span]) -> None:
        if self._collector_rejected:
            self._log_spans(location, spans)
        else:
            self._export_spans_to_collector(spans, from_collector_batch=True)

    def _export_spans_incrementally(self, spans: Sequence[ReadableSpan]) -> None:
        """Send spans to Zerobus while preserving the V3 metadata path."""
        if not spans:
            return
        try:
            if self._collector_rejected:
                self._send_spans_via_rest(spans)
            elif self._should_log_async():
                for span in self._as_mlflow_spans(spans):
                    self._span_batcher.add_span(location=self._table_name, span=span)
            else:
                self._export_spans_to_collector(spans)
        except Exception as exc:
            _logger.warning(
                "Databricks OTel collector span export failed: %s. Spans were NOT exported "
                "to the collector; trace metadata export via the MLflow backend is unaffected.",
                exc,
            )

    def flush(self, terminate: bool = False) -> None:
        """Drain the span batcher and async queue used by this exporter."""
        flush_exporter(self, terminate=terminate)

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            try:
                # Drain queued collector batches before closing their connection pool.
                self.flush(terminate=True)
            finally:
                self._zerobus_client.close()


def _log_collector_unavailable(reason: str, *args) -> None:
    """Log why the default collector path is not used."""
    message = reason + " Falling back to the MLflow tracing server span export path."
    if MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.is_set():
        _logger.warning(message, *args)
    else:
        _logger.debug(message, *args)


def get_databricks_otel_collector_span_exporter(
    destination,
    tracking_uri: str | None,
) -> DatabricksOtelCollectorSpanExporter | None:
    """Return a lazy collector exporter for an eligible Unity Catalog destination."""
    if not MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.get():
        return None
    if not isinstance(destination, UnityCatalog):
        return None

    table_name = _get_table_name_from_destination(destination)
    if not table_name:
        _log_collector_unavailable(
            "Could not determine the UC table name from the trace destination %r.",
            destination,
        )
        return None

    _logger.debug(
        "Databricks OTel collector span exporter configured with deferred credential and "
        "endpoint resolution: table=%r",
        table_name,
    )
    try:
        return DatabricksOtelCollectorSpanExporter(
            tracking_uri=tracking_uri,
            token_source=None,
            table_name=table_name,
        )
    except Exception as exc:
        _log_collector_unavailable(
            "The Databricks OTel collector span exporter could not be constructed: %s.",
            exc,
        )
        return None


def _get_table_name_from_destination(destination) -> str | None:
    """Extract a fully-qualified OTLP spans table name from a destination."""
    return getattr(destination, "full_otel_spans_table_name", None)
