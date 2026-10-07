"""Route Unity Catalog spans through the Databricks OTel collector when possible.

The router composes the collector transport with the Unity Catalog exporter.  The
UC exporter owns trace metadata and the shared asynchronous export queue; the
router owns span batching and decides when a batch must be replayed through the
tracking server.
"""

import logging
import threading
from typing import Sequence

import requests
import urllib3.exceptions
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceResponse,
)
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from mlflow.entities.span import Span
from mlflow.entities.trace_location import UnityCatalog
from mlflow.environment_variables import MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT
from mlflow.tracing.export.databricks_otel_client import (
    ZerobusOtelTokenError,
    ZerobusOtelTokenRefreshError,
)
from mlflow.tracing.export.databricks_otel_collector import (
    DatabricksOtelCollectorSpanExporter,
    DatabricksOtelSerializationError,
)
from mlflow.tracing.export.span_batcher import SpanBatcher
from mlflow.tracing.export.uc_table import DatabricksUCTableSpanExporter

_logger = logging.getLogger(__name__)

_BODY_SNIPPET_BYTES = 512
_REPLAYABLE_STATUS_CODES = (408, 413, 429)


def _log_collector_unavailable(reason: str, *args) -> None:
    """Log collector configuration failures at the appropriate default level."""
    message = reason + " Falling back to the MLflow tracing server span export path."
    if MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.is_set():
        _logger.warning(message, *args)
    else:
        _logger.debug(message, *args)


def _is_connection_not_established(exc: BaseException) -> bool:
    """Return whether *exc* proves that the collector request was never sent."""
    if isinstance(exc, requests.ConnectTimeout):
        return True
    if not isinstance(exc, requests.ConnectionError):
        return False

    # requests and urllib3 use several layers of exception wrapping.  Walk the
    # args and exception chains so DNS and connection-establishment failures are
    # treated as safe to replay, while a reset after sending remains ambiguous.
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


class DatabricksOtelSpanRouter(SpanExporter):
    """Export spans to the collector with a pinned UC REST fallback.

    Metadata is exported exactly once through ``uc_exporter``.  Span batches are
    sent through the collector until a sticky failure pins the router to REST.
    ``collector`` and ``uc_exporter`` are injectable to keep routing behavior
    independently testable; normal callers should use the factory below.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        table_name: str | None = None,
        collector=None,
        uc_exporter=None,
    ) -> None:
        self._tracking_uri = tracking_uri
        self._table_name = table_name

        if self._table_name is None:
            raise ValueError("A fully-qualified Unity Catalog spans table is required")

        self._collector = collector or DatabricksOtelCollectorSpanExporter(
            tracking_uri=tracking_uri,
            token_source=None,
            table_name=self._table_name,
        )
        if uc_exporter is None:
            self._uc_exporter = DatabricksUCTableSpanExporter(
                tracking_uri=tracking_uri,
                metadata_only=True,
            )
        else:
            self._uc_exporter = uc_exporter

        self._collector_rejected = False
        self._state_lock = threading.RLock()
        self._has_warned_fallback = False
        self._has_warned_ambiguous_fallback = False
        self._has_warned_ambiguous_drop = False
        self._async_components_terminated = False
        self._shutdown = False

        self._span_batcher = None
        if (async_queue := self._get_async_queue()) is not None:
            self._span_batcher = SpanBatcher(
                async_task_queue=async_queue,
                log_spans_func=self._export_batch,
            )

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def collector(self):
        return self._collector

    @property
    def uc_exporter(self):
        return self._uc_exporter

    @property
    def collector_rejected(self) -> bool:
        return self._collector_rejected

    def _get_async_queue(self):
        return self._uc_exporter.async_queue

    def _should_log_async(self) -> bool:
        return self._uc_exporter.should_log_async()

    @staticmethod
    def _as_mlflow_spans(spans: Sequence[ReadableSpan | Span]) -> list[Span]:
        return [span if isinstance(span, Span) else Span(span) for span in spans]

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export metadata once and route the completed spans."""
        try:
            if spans:
                mlflow_spans = self._as_mlflow_spans(spans)
                if self._should_log_async() and self._span_batcher is not None:
                    for span in mlflow_spans:
                        self._span_batcher.add_span(location=self._table_name, span=span)
                else:
                    self._export_batch(self._table_name, mlflow_spans)
        except Exception as exc:
            _logger.warning(
                "Databricks OTel span routing failed: %s. Trace metadata export is unaffected.",
                exc,
            )
        finally:
            # MlflowV3SpanExporter exports metadata after incremental spans and
            # does so even for an empty batch, which flushes any deferred root.
            try:
                self._uc_exporter.export(spans)
            except Exception as exc:
                _logger.warning("Failed to export trace metadata through the UC exporter: %s", exc)
        return SpanExportResult.SUCCESS

    def _write_spans_to_table(self, spans: Sequence[ReadableSpan | Span]) -> None:
        mlflow_spans = self._as_mlflow_spans(spans)
        self._uc_exporter.write_spans_to_table(self._table_name, mlflow_spans)

    def _export_batch(self, location: str, spans: list[Span]) -> None:
        # SpanBatcher carries a location argument for generic table batching, but
        # this router is deliberately pinned to the factory-selected table.
        del location
        if self._collector_rejected:
            self._write_spans_to_table(spans)
        else:
            self._send_batch_to_collector(spans)

    def _warn_fallback_once(self, message: str, *args) -> None:
        with self._state_lock:
            already_warned = self._has_warned_fallback
            self._has_warned_fallback = True
        if already_warned:
            _logger.debug(message, *args)
        else:
            _logger.warning(message, *args)

    def _warn_ambiguous_fallback_once(self, message: str, *args) -> None:
        with self._state_lock:
            already_warned = self._has_warned_ambiguous_fallback
            self._has_warned_ambiguous_fallback = True
        if already_warned:
            _logger.debug(message, *args)
        else:
            _logger.warning(message, *args)

    def _set_collector_rejected(self) -> None:
        with self._state_lock:
            self._collector_rejected = True

    def _fallback_to_rest(self, spans: Sequence[ReadableSpan | Span], reason: str, *args) -> None:
        self._set_collector_rejected()
        self._warn_fallback_once(reason, *args)
        self._write_spans_to_table(spans)

    def _replay_batch_via_rest(
        self, spans: Sequence[ReadableSpan | Span], reason: str, *args
    ) -> None:
        _logger.debug(reason, *args)
        self._write_spans_to_table(spans)

    def _ambiguous_fallback_to_rest(
        self, spans: Sequence[ReadableSpan | Span], reason: str, *args
    ) -> None:
        self._set_collector_rejected()
        self._warn_ambiguous_fallback_once(reason, *args)
        self._write_spans_to_table(spans)

    def _ambiguous_drop(self, reason: str, *args) -> None:
        self._set_collector_rejected()
        with self._state_lock:
            already_warned = self._has_warned_ambiguous_drop
            self._has_warned_ambiguous_drop = True
        if already_warned:
            _logger.debug(reason, *args)
        else:
            _logger.warning(reason, *args)

    def _handle_collector_exception(self, exc: BaseException, spans: Sequence[Span]) -> None:
        if isinstance(exc, ZerobusOtelTokenError):
            self._replay_batch_via_rest(
                spans,
                "Minting the Databricks OTel collector token failed: %s. Replaying this "
                "batch through the MLflow tracing server path.",
                exc.__cause__ or exc,
            )
        elif isinstance(exc, ZerobusOtelTokenRefreshError):
            self._fallback_to_rest(
                spans,
                "Refreshing the Databricks OTel collector token failed. Falling back to the "
                "MLflow tracing server path for this and subsequent batches.",
            )
        elif _is_connection_not_established(exc):
            self._fallback_to_rest(
                spans,
                "The Databricks OTel collector connection was not established. Falling back "
                "to the MLflow tracing server path for this and subsequent span batches.",
            )
        else:
            self._ambiguous_fallback_to_rest(
                spans,
                "The Databricks OTel collector span export request failed: %s. The collector "
                "may have already ingested this batch; replaying it through the MLflow "
                "tracing server path can create duplicate spans. New span batches will use "
                "that path.",
                exc,
            )

    def _handle_collector_response(
        self, response: requests.Response, spans: Sequence[Span]
    ) -> None:
        if not response.content:
            return

        response_message = ExportTraceServiceResponse()
        try:
            response_message.ParseFromString(response.content)
        except Exception as exc:
            self._ambiguous_drop(
                "The Databricks OTel collector returned an invalid HTTP 200 response: %s. "
                "Delivery is ambiguous; dropping this batch and using the MLflow tracing "
                "server path for future batches.",
                exc,
            )
            return

        rejected_spans = response_message.partial_success.rejected_spans
        if rejected_spans == 0:
            return
        if rejected_spans < 0 or rejected_spans > len(spans):
            self._ambiguous_drop(
                "The Databricks OTel collector reported %d rejected spans for a batch of %d "
                "spans. Delivery is ambiguous; dropping this batch and using the MLflow "
                "tracing server path for future batches.",
                rejected_spans,
                len(spans),
            )
            return
        if rejected_spans == len(spans):
            self._set_collector_rejected()
            self._warn_fallback_once(
                "The Databricks OTel collector rejected all %d spans in a successful HTTP 200 "
                "partial response (%s). Replaying this batch through the MLflow tracing "
                "server path and using that path for future batches.",
                len(spans),
                response_message.partial_success.error_message,
            )
            self._write_spans_to_table(spans)
            return

        # The response does not identify rejected spans.  Replaying this batch
        # could duplicate the subset accepted by the collector, so drop it while
        # pinning future batches to REST.
        self._set_collector_rejected()
        self._warn_fallback_once(
            "The Databricks OTel collector partially accepted a span batch: %d of %d spans "
            "were rejected (%s). Future batches will use the MLflow tracing server path.",
            rejected_spans,
            len(spans),
            response_message.partial_success.error_message,
        )

    def _send_batch_to_collector(self, spans: Sequence[Span]) -> None:
        try:
            response = self._collector.send_batch(list(spans))
        except DatabricksOtelSerializationError:
            # Encoding failed locally before an HTTP request was attempted.  The
            # old exporter dropped this batch and left collector routing intact.
            raise
        except Exception as exc:
            self._handle_collector_exception(exc, spans)
            return

        # ``None`` means the collector could not become ready.  It is safe to
        # replay the current batch, but all future batches use REST.
        if response is None:
            host = getattr(self._collector, "host", None)
            workspace_id = getattr(self._collector, "workspace_id", None)
            config_warned = getattr(self._collector, "config_warned", False)
            self._set_collector_rejected()
            if not config_warned:
                _log_collector_unavailable(
                    "The Databricks OTel collector endpoint could not be resolved for "
                    "workspace_id=%r, host=%r.",
                    workspace_id,
                    host,
                )
            self._write_spans_to_table(spans)
            return

        status_code = response.status_code
        if response.ok:
            if status_code == 200:
                self._handle_collector_response(response, spans)
            return

        if 400 <= status_code < 500:
            if status_code in _REPLAYABLE_STATUS_CODES:
                self._replay_batch_via_rest(
                    spans,
                    "The Databricks OTel collector returned HTTP %d for a span export "
                    "(transient or batch-specific rejection). Replaying this batch through "
                    "the MLflow tracing server path.",
                    status_code,
                )
            else:
                self._fallback_to_rest(
                    spans,
                    "The Databricks OTel collector rejected a span export with HTTP %d. "
                    "Falling back to the MLflow tracing server path for this and subsequent "
                    "span batches.",
                    status_code,
                )
            return

        if 500 <= status_code < 600:
            self._ambiguous_fallback_to_rest(
                spans,
                "The Databricks OTel collector span export failed with HTTP %d: %r. The "
                "collector may have already ingested this batch; replaying it through the "
                "MLflow tracing server path can create duplicate spans. New span batches "
                "will use that path.",
                status_code,
                response.content[:_BODY_SNIPPET_BYTES],
            )
            return

        self._ambiguous_drop(
            "The Databricks OTel collector span export failed with HTTP %d: %r. Delivery is "
            "ambiguous, so this batch was dropped; future batches will use the MLflow tracing "
            "server path.",
            status_code,
            response.content[:_BODY_SNIPPET_BYTES],
        )

    def flush_async_components(self, terminate: bool = False) -> None:
        """Drain the metadata exporter, router batcher, and shared async queue."""
        if terminate and self._async_components_terminated:
            return

        try:
            # Flush deferred root metadata before the shared queue is drained.
            if terminate:
                self._uc_exporter.shutdown()
        finally:
            try:
                if self._span_batcher is not None:
                    if terminate:
                        self._span_batcher.shutdown()
                    else:
                        self._span_batcher.flush()
            finally:
                try:
                    if (async_queue := self._get_async_queue()) is not None:
                        async_queue.flush(terminate=terminate)
                finally:
                    if terminate:
                        self._async_components_terminated = True

    def flush(self, terminate: bool = False) -> None:
        self.flush_async_components(terminate=terminate)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        try:
            self.flush_async_components(terminate=True)
        finally:
            close = getattr(self._collector, "close", None)
            if callable(close):
                close()
            self._shutdown = True


def get_databricks_otel_span_router(
    destination,
    tracking_uri: str | None,
) -> DatabricksOtelSpanRouter | None:
    """Return the default collector router for Unity Catalog table destinations."""
    if not MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.get():
        return None
    if not isinstance(destination, UnityCatalog):
        return None

    table_name = destination.full_otel_spans_table_name
    if not table_name:
        _log_collector_unavailable(
            "Could not determine the UC table name from the trace destination %r; "
            "using the standard UC table exporter.",
            destination,
        )
        return None

    try:
        return DatabricksOtelSpanRouter(
            tracking_uri=tracking_uri,
            table_name=table_name,
        )
    except Exception as exc:
        _log_collector_unavailable(
            "The Databricks OTel span router could not be constructed: %s.", exc
        )
        return None


__all__ = [
    "DatabricksOtelSpanRouter",
    "DatabricksOtelSerializationError",
    "_is_connection_not_established",
    "get_databricks_otel_span_router",
]
