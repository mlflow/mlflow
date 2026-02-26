"""
The ``mlflow.otel`` module provides generic OTEL-to-MLflow span forwarding.

When enabled, every span produced by any OpenTelemetry-instrumented library
(e.g. Langfuse, OpenInference / Arize Phoenix) is automatically forwarded
to the MLflow backend via the OTLP endpoint.

.. code-block:: python

    import mlflow.otel

    mlflow.otel.autolog()  # enable (synchronous export)
    mlflow.otel.autolog(batch=True)  # enable (batched export)
    mlflow.otel.autolog(disable=True)  # disable
"""

import logging

from opentelemetry import trace as otel_trace_api
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

import mlflow
from mlflow.tracing.utils.otlp import MLFLOW_EXPERIMENT_ID_HEADER, OTLP_TRACES_PATH
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.autologging_utils.safety import _AUTOLOGGING_CLEANUP_CALLBACKS

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "otel"

# Keep a reference so we can disable/re-enable it.
_active_processor: "_ToggleableSpanProcessor | None" = None


class _ToggleableSpanProcessor(SpanProcessor):
    """A span processor that can be enabled/disabled at runtime.

    Wraps a standard OTEL ``SpanProcessor`` (Simple or Batch).  Since there
    is no public API to *remove* a processor from a TracerProvider, we gate
    on_start/on_end behind a flag so that disabling autolog truly stops
    span processing.
    """

    def __init__(self, inner: SpanProcessor):
        self._inner = inner
        self._enabled = True

    def on_start(self, span: OTelSpan, parent_context: Context | None = None):
        if not self._enabled:
            return
        self._inner.on_start(span, parent_context)

    def on_end(self, span: OTelReadableSpan) -> None:
        if not self._enabled:
            return
        self._inner.on_end(span)

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False


def setup_otel_processor(flavor_name: str, batch: bool = False) -> None:
    """Register an MLflow span processor on the global OTEL TracerProvider.

    Spans are exported to the MLflow backend via the OTLP endpoint. The
    server handles trace creation and attribute translation automatically.

    Args:
        flavor_name: Integration name used for cleanup callback registration
            (e.g. ``"langfuse"``, ``"otel"``).
        batch: If ``True``, use ``BatchSpanProcessor`` for buffered export.
            If ``False`` (default), use ``SimpleSpanProcessor`` for synchronous export.
    """
    global _active_processor

    if _active_processor is not None:
        _active_processor.enable()
        _logger.debug("Re-enabled existing MLflow span processor.")
        return

    # Ensure a real (non-proxy) TracerProvider exists globally.
    # Langfuse performs the same check in _init_tracer_provider(): it
    # replaces a ProxyTracerProvider but reuses an existing SdkTracerProvider.
    # This means either initialization order works — whichever runs first
    # creates the SdkTracerProvider, and the other adds its processor to it.
    provider = otel_trace_api.get_tracer_provider()
    if isinstance(provider, otel_trace_api.ProxyTracerProvider):
        provider = SdkTracerProvider()
        otel_trace_api.set_tracer_provider(provider)

    if not isinstance(provider, SdkTracerProvider):
        _logger.warning(
            "Global TracerProvider is %s, not an SDK TracerProvider. "
            "Cannot register MLflow span processor.",
            type(provider).__name__,
        )
        return

    tracking_uri = mlflow.get_tracking_uri().rstrip("/")
    endpoint = f"{tracking_uri}{OTLP_TRACES_PATH}"
    experiment_id = _get_experiment_id()

    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={MLFLOW_EXPERIMENT_ID_HEADER: experiment_id},
    )

    inner = BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
    processor = _ToggleableSpanProcessor(inner)
    provider.add_span_processor(processor)

    _active_processor = processor

    # Register teardown so that ``revert_patches(flavor_name)`` — called by
    # the ``@autologging_integration`` decorator on ``autolog(disable=True)``
    # — disables the processor.  The decorator short-circuits before our
    # function body runs, so we cannot rely on the body itself.
    _AUTOLOGGING_CLEANUP_CALLBACKS.setdefault(flavor_name, []).append(teardown_otel_processor)

    _logger.debug(
        "Registered MLflow span processor on global TracerProvider "
        "(endpoint=%s, experiment_id=%s, batch=%s).",
        endpoint,
        experiment_id,
        batch,
    )


def teardown_otel_processor() -> None:
    """Disable the MLflow span processor (best-effort)."""
    if _active_processor is None:
        return

    _active_processor.disable()
    _logger.debug("Disabled MLflow span processor.")


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
    batch: bool = False,
):
    """
    Enables (or disables) generic OTEL-to-MLflow span forwarding.

    Args:
        log_traces: If ``True``, traces are logged to MLflow.
            If ``False``, no MLflow traces are collected.
            Default ``True``.
        disable: If ``True``, disables the OTEL autologging
            integration.  Default ``False``.
        silent: If ``True``, suppress all event logs and warnings from
            MLflow during OTEL autologging. Default ``False``.
        batch: If ``True``, use ``BatchSpanProcessor`` for buffered,
            asynchronous export.  If ``False`` (default), use
            ``SimpleSpanProcessor`` for synchronous, immediate export.
    """
    setup_otel_processor(flavor_name=FLAVOR_NAME, batch=batch)
