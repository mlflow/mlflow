"""
Autologging logic for Agno V2 (>= 2.0.0) using OpenTelemetry instrumentation.
"""

import importlib.metadata as _meta
import logging

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Tracer, TracerProvider
from packaging.version import Version

from mlflow.exceptions import MlflowException
from mlflow.tracing.provider import _get_tracer, get_current_context

_logger = logging.getLogger(__name__)
_agno_instrumentor = None


# AGNO SDK doesn't provide version parameter from 1.7.1 onwards. Hence we capture the
# latest version manually

try:
    import agno

    if not hasattr(agno, "__version__"):
        try:
            agno.__version__ = _meta.version("agno")
        except _meta.PackageNotFoundError:
            agno.__version__ = "1.7.7"
except ImportError:
    pass


def _is_agno_v2() -> bool:
    """Check if Agno V2 (>= 2.0.0) is installed."""
    try:
        return Version(_meta.version("agno")).major >= 2
    except _meta.PackageNotFoundError:
        return False


def _bridge_parent_context(context):
    """Resolve the parent context for an Agno span."""
    if context is not None:
        return context

    # get_current_context() returns MLflow's context in isolated mode
    # (MLFLOW_USE_DEFAULT_TRACER_PROVIDER=true), or None in unified mode. None => nothing to bridge,
    # so return None and let OTel resolve the parent from the native context.
    mlflow_context = get_current_context()
    if mlflow_context is None:
        return None

    # Already inside the Agno subtree (a parent Agno span is current in the native context): return
    # None so this span nests under that span natively, not under the outer MLflow span.
    span = trace.get_current_span()
    span_context = span.get_span_context() if span is not None else None
    if span_context is not None and span_context.is_valid:
        return None

    return mlflow_context


class _MlflowContextBridgingTracer(Tracer):
    """Delegating ``Tracer`` that parents Agno's OpenInference spans under active MLflow spans."""

    def __init__(self, tracer: Tracer):
        self._tracer = tracer

    def start_span(self, name: str, context: Context | None = None, **kwargs):
        return self._tracer.start_span(name, context=_bridge_parent_context(context), **kwargs)

    def start_as_current_span(self, name: str, context: Context | None = None, **kwargs):
        return self._tracer.start_as_current_span(
            name, context=_bridge_parent_context(context), **kwargs
        )

    def __getattr__(self, name):
        return getattr(self._tracer, name)


class _MlflowTracerProvider(TracerProvider):
    """``TracerProvider`` given to ``AgnoInstrumentor().instrument()``"""

    def get_tracer(
        self,
        instrumenting_module_name: str,
        *args,
        **kwargs,
    ) -> Tracer:
        return _MlflowContextBridgingTracer(_get_tracer(instrumenting_module_name))


def _setup_otel_instrumentation() -> None:
    """Set up OpenTelemetry instrumentation for Agno V2."""
    global _agno_instrumentor

    if _agno_instrumentor is not None:
        _logger.debug("OpenTelemetry instrumentation already set up for Agno V2")
        return

    try:
        from openinference.instrumentation.agno import AgnoInstrumentor

        _agno_instrumentor = AgnoInstrumentor()
        _agno_instrumentor.instrument(tracer_provider=_MlflowTracerProvider())
        _logger.debug("OpenTelemetry instrumentation enabled for Agno V2")

    except ImportError as exc:
        raise MlflowException(
            "Failed to set up OpenTelemetry instrumentation for Agno V2. "
            "Please install the following required packages: "
            "'pip install opentelemetry-exporter-otlp openinference-instrumentation-agno'. "
        ) from exc
    except Exception as exc:
        _logger.warning("Failed to set up OpenTelemetry instrumentation for Agno V2: %s", exc)


def _uninstrument_otel() -> None:
    """Uninstrument OpenTelemetry for Agno V2."""
    global _agno_instrumentor

    try:
        if _agno_instrumentor is not None:
            _agno_instrumentor.uninstrument()
            _agno_instrumentor = None
            _logger.debug("OpenTelemetry instrumentation disabled for Agno V2")
        else:
            _logger.warning("Instrumentor instance not found, cannot uninstrument")
    except Exception as exc:
        _logger.warning("Failed to uninstrument Agno V2: %s", exc)
