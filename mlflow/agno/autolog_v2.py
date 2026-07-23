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
    # Honor an explicitly-passed parent only when it actually carries a valid span. OpenInference
    # sometimes hands us a context wrapping INVALID_SPAN (e.g. a top-level Agno Team, which it
    # forces to a root span because its own get_current_span() check cannot see MLflow's active
    # span in isolated mode). Treat such a context as "no parent" so we can still bridge below.
    if context is not None and trace.get_current_span(context).get_span_context().is_valid:
        return context

    # A valid span is already active in the native OTel context, meaning we are inside the Agno
    # subtree. Return None so OTel nests this span under it natively.
    if trace.get_current_span().get_span_context().is_valid:
        return None

    # bridge to MLflow context management between isolated and non-isolated mode.
    return get_current_context()


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
