"""
The ``mlflow.otel`` module provides generic OTEL-to-MLflow span forwarding.

When enabled, every span produced by any OpenTelemetry-instrumented library
(e.g. Langfuse, OpenInference / Arize Phoenix) is automatically forwarded
to the MLflow backend.

.. code-block:: python

    import mlflow.otel

    mlflow.otel.autolog()  # enable
    mlflow.otel.autolog(disable=True)  # disable
"""

import logging

from opentelemetry import trace as otel_trace_api
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.export import SpanExporter

import mlflow
from mlflow.tracing.export.mlflow_v3 import MlflowV3SpanExporter
from mlflow.tracing.processor.mlflow_v3 import MlflowV3SpanProcessor
from mlflow.utils.autologging_utils import autologging_integration
from mlflow.utils.autologging_utils.safety import _AUTOLOGGING_CLEANUP_CALLBACKS

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "otel"

# Keep a reference so we can disable/re-enable it.
_active_processor: "_ToggleableSpanProcessor | None" = None


class _ToggleableSpanProcessor(MlflowV3SpanProcessor):
    """A span processor that can be enabled/disabled at runtime.

    Since there is no public API to *remove* a processor from a
    TracerProvider, we instead gate on_start/on_end behind a flag
    so that disabling autolog truly stops span processing.
    """

    def __init__(self, span_exporter: SpanExporter, export_metrics: bool):
        super().__init__(span_exporter=span_exporter, export_metrics=export_metrics)
        self._enabled = True

    def on_start(self, span: OTelSpan, parent_context: Context | None = None):
        if not self._enabled:
            return
        super().on_start(span, parent_context)

    def on_end(self, span: OTelReadableSpan) -> None:
        if not self._enabled:
            return
        super().on_end(span)

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False


def setup_otel_processor(flavor_name: str) -> None:
    """Register an MLflow span processor on the global OTEL TracerProvider.

    Args:
        flavor_name: Integration name used for cleanup callback registration
            (e.g. ``"langfuse"``, ``"otel"``).
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

    tracking_uri = mlflow.get_tracking_uri()
    exporter = MlflowV3SpanExporter(tracking_uri=tracking_uri)
    processor = _ToggleableSpanProcessor(span_exporter=exporter, export_metrics=False)
    provider.add_span_processor(processor)

    _active_processor = processor

    # Register teardown so that ``revert_patches(flavor_name)`` — called by
    # the ``@autologging_integration`` decorator on ``autolog(disable=True)``
    # — disables the processor.  The decorator short-circuits before our
    # function body runs, so we cannot rely on the body itself.
    _AUTOLOGGING_CLEANUP_CALLBACKS.setdefault(flavor_name, []).append(teardown_otel_processor)

    _logger.debug(
        "Registered MLflow span processor on global TracerProvider (tracking_uri=%s).",
        tracking_uri,
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
    """
    setup_otel_processor(flavor_name=FLAVOR_NAME)
