"""Export Unity Catalog spans through the Databricks OTel collector when possible.

The exporter owns destination eligibility, batching, metadata export, and
delivery failure classification, with the DatabricksUCTableSpanExporter as its
fallback. The client resolves credentials and the endpoint on the first span
batch, then serializes and sends it to the collector.
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
    DatabricksOTelClient,
    DatabricksOtelConfigurationError,
    DatabricksOtelSerializationError,
    DatabricksOtelTokenError,
    DatabricksOtelTokenRefreshError,
    DatabricksOtelUnavailableError,
)
from mlflow.tracing.export.span_batcher import SpanBatcher
from mlflow.tracing.export.uc_table import DatabricksUCTableSpanExporter

_logger = logging.getLogger(__name__)

_BODY_SNIPPET_BYTES = 512
_REPLAYABLE_STATUS_CODES = (408, 413, 429)


def _log_unavailable(reason: str, *args) -> None:
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


class DatabricksOtelExporter(SpanExporter):
    """Export spans to the collector with a pinned legacy-exporter fallback.

    Metadata is exported exactly once through ``fallback_exporter``. Span
    batches are sent through the OTel client until a sticky failure pins the
    exporter to the legacy path. Callers provide both dependencies; the factory below
    builds the standard pair.
    """

    def __init__(
        self,
        *,
        table_name: str,
        otel_client: DatabricksOTelClient,
        fallback_exporter: DatabricksUCTableSpanExporter,
    ) -> None:
        self._table_name = table_name

        if self._table_name is None:
            raise ValueError("A fully-qualified Unity Catalog spans table is required")

        self._otel_client = otel_client
        self._fallback_exporter = fallback_exporter

        self._use_legacy_exporter = False
        self._state_lock = threading.RLock()
        self._warned_events: set[str] = set()
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
    def otel_client(self):
        return self._otel_client

    @property
    def fallback_exporter(self):
        return self._fallback_exporter

    @property
    def using_legacy_exporter(self) -> bool:
        return self._use_legacy_exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Route spans through OTel while exporting trace metadata through the legacy exporter.

        - Wrap completed spans and optionally batch them for OTel delivery.
        - Replay or drop failed batches based on delivery certainty, and use the
          legacy exporter for future batches when necessary.
        - Attempt metadata export once through the legacy exporter, even for
          empty batches or when span routing fails. Log errors without propagating them.
        """
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
                self._fallback_exporter.export(spans)
            except Exception as exc:
                _logger.warning("Failed to export trace metadata through the UC exporter: %s", exc)
        return SpanExportResult.SUCCESS

    def flush(self, terminate: bool = False) -> None:
        self._flush_async_components(terminate=terminate)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        try:
            self._flush_async_components(terminate=True)
        finally:
            close = getattr(self._otel_client, "close", None)
            if callable(close):
                close()
            self._shutdown = True

    def _get_async_queue(self):
        return self._fallback_exporter.async_queue

    def _should_log_async(self) -> bool:
        return self._fallback_exporter.should_log_async()

    @staticmethod
    def _as_mlflow_spans(spans: Sequence[ReadableSpan | Span]) -> list[Span]:
        return [span if isinstance(span, Span) else Span(span) for span in spans]

    def _write_spans_to_table(self, spans: Sequence[ReadableSpan | Span]) -> None:
        mlflow_spans = self._as_mlflow_spans(spans)
        self._fallback_exporter.write_spans_to_table(self._table_name, mlflow_spans)

    def _export_batch(self, location: str, spans: list[Span]) -> None:
        # SpanBatcher carries a location argument for generic table batching, but
        # this exporter is deliberately pinned to the factory-selected table.
        del location
        if self._use_legacy_exporter:
            self._write_spans_to_table(spans)
        else:
            self._send_batch(spans)

    def _warn_once(self, key: str, message: str, *args) -> None:
        with self._state_lock:
            already_warned = key in self._warned_events
            self._warned_events.add(key)
        if already_warned:
            _logger.debug(message, *args)
        else:
            _logger.warning(message, *args)

    def _pin_to_legacy_exporter(self) -> None:
        with self._state_lock:
            self._use_legacy_exporter = True

    def _fallback_to_legacy_exporter(
        self, spans: Sequence[ReadableSpan | Span], reason: str, *args
    ) -> None:
        self._pin_to_legacy_exporter()
        self._warn_once("fallback", reason, *args)
        self._write_spans_to_table(spans)

    def _replay_batch_via_legacy_exporter(
        self, spans: Sequence[ReadableSpan | Span], reason: str, *args
    ) -> None:
        _logger.debug(reason, *args)
        self._write_spans_to_table(spans)

    def _ambiguous_fallback_to_legacy_exporter(
        self, spans: Sequence[ReadableSpan | Span], reason: str, *args
    ) -> None:
        self._pin_to_legacy_exporter()
        self._warn_once(
            "ambiguous_fallback",
            reason + " The collector may have already ingested this batch, so replaying it through "
            "the MLflow tracing server path may create duplicate spans. New span batches "
            "will use the MLflow tracing server path; concurrent exports already in progress "
            "may still contact the collector.",
            *args,
        )
        self._write_spans_to_table(spans)

    def _ambiguous_drop(self, reason: str, *args) -> None:
        self._pin_to_legacy_exporter()
        self._warn_once(
            "ambiguous_drop",
            reason + " Delivery is ambiguous, so the spans were dropped instead of replayed over "
            "the MLflow tracing server path, which could duplicate them. Future span "
            "batches will use the MLflow tracing server path.",
            *args,
        )

    def _handle_export_exception(self, exc: BaseException, spans: Sequence[Span]) -> None:
        if isinstance(exc, DatabricksOtelTokenError):
            self._replay_batch_via_legacy_exporter(
                spans,
                "Minting the Databricks OTel collector token failed: %s. Replaying this "
                "batch through the MLflow tracing server path.",
                exc.__cause__ or exc,
            )
        elif isinstance(exc, DatabricksOtelTokenRefreshError):
            self._fallback_to_legacy_exporter(
                spans,
                "Refreshing the Databricks OTel collector token failed. Falling back to the "
                "MLflow tracing server path for this and subsequent batches.",
            )
        elif isinstance(exc, DatabricksOtelUnavailableError):
            self._pin_to_legacy_exporter()
            if not isinstance(exc, DatabricksOtelConfigurationError):
                _log_unavailable("%s", exc)
            self._write_spans_to_table(spans)
        elif _is_connection_not_established(exc):
            self._fallback_to_legacy_exporter(
                spans,
                "The Databricks OTel collector connection was not established. Falling back "
                "to the MLflow tracing server path for this and subsequent span batches.",
            )
        else:
            self._ambiguous_fallback_to_legacy_exporter(
                spans,
                "The Databricks OTel collector span export request failed: %s.",
                exc,
            )

    def _handle_export_response(self, response: requests.Response, spans: Sequence[Span]) -> None:
        if not response.content:
            return

        response_message = ExportTraceServiceResponse()
        try:
            response_message.ParseFromString(response.content)
        except Exception as exc:
            self._ambiguous_drop(
                "The Databricks OTel collector returned an invalid HTTP 200 response: %s.",
                exc,
            )
            return

        rejected_spans = response_message.partial_success.rejected_spans
        if rejected_spans == 0:
            return
        if rejected_spans < 0 or rejected_spans > len(spans):
            self._ambiguous_drop(
                "The Databricks OTel collector reported %d rejected spans for a batch of %d spans.",
                rejected_spans,
                len(spans),
            )
            return
        if rejected_spans == len(spans):
            self._pin_to_legacy_exporter()
            self._warn_once(
                "fallback",
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
        # pinning future batches to the legacy exporter.
        self._pin_to_legacy_exporter()
        self._warn_once(
            "fallback",
            "The Databricks OTel collector partially accepted a span batch: %d of %d spans "
            "were rejected (%s). Future batches will use the MLflow tracing server path.",
            rejected_spans,
            len(spans),
            response_message.partial_success.error_message,
        )

    def _send_batch(self, spans: Sequence[Span]) -> None:
        try:
            response = self._otel_client.export_spans(list(spans))
        except DatabricksOtelSerializationError:
            # Encoding failed locally before an HTTP request was attempted.  The
            # old exporter dropped this batch and left collector routing intact.
            raise
        except Exception as exc:
            self._handle_export_exception(exc, spans)
            return

        status_code = response.status_code
        if response.ok:
            if status_code == 200:
                self._handle_export_response(response, spans)
            return

        if 400 <= status_code < 500:
            if status_code in _REPLAYABLE_STATUS_CODES:
                self._replay_batch_via_legacy_exporter(
                    spans,
                    "The Databricks OTel collector returned HTTP %d for a span export "
                    "(transient or batch-specific rejection). Replaying this batch through "
                    "the MLflow tracing server path.",
                    status_code,
                )
            else:
                self._fallback_to_legacy_exporter(
                    spans,
                    "The Databricks OTel collector rejected a span export with HTTP %d. "
                    "Falling back to the MLflow tracing server path for this and subsequent "
                    "span batches.",
                    status_code,
                )
            return

        if 500 <= status_code < 600:
            self._ambiguous_fallback_to_legacy_exporter(
                spans,
                "The Databricks OTel collector span export failed with HTTP %d: %r.",
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

    def _flush_async_components(self, terminate: bool = False) -> None:
        """Drain the fallback exporter, span batcher, and shared async queue."""
        if terminate and self._async_components_terminated:
            return

        try:
            # Flush deferred root metadata before the shared queue is drained.
            if terminate:
                self._fallback_exporter.shutdown()
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


def get_databricks_otel_exporter(
    destination,
    tracking_uri: str | None,
) -> DatabricksOtelExporter | None:
    """Return the default OTel exporter for Unity Catalog table destinations."""
    if not MLFLOW_ENABLE_DATABRICKS_OTEL_COLLECTOR_EXPORT.get():
        return None
    if not isinstance(destination, UnityCatalog):
        return None

    table_name = destination.full_otel_spans_table_name
    if not table_name:
        _log_unavailable(
            "Could not determine the UC table name from the trace destination %r; "
            "using the standard UC table exporter.",
            destination,
        )
        return None

    try:
        return DatabricksOtelExporter(
            table_name=table_name,
            otel_client=DatabricksOTelClient(
                tracking_uri=tracking_uri,
                token_source=None,
                table_name=table_name,
            ),
            fallback_exporter=DatabricksUCTableSpanExporter(
                tracking_uri=tracking_uri,
                metadata_only=True,
            ),
        )
    except Exception as exc:
        _log_unavailable("The Databricks OTel exporter could not be constructed: %s.", exc)
        return None


__all__ = [
    "DatabricksOtelExporter",
    "DatabricksOtelSerializationError",
    "_is_connection_not_established",
    "get_databricks_otel_exporter",
]
