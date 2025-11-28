"""
Autologging logic for Agno V2 (>= 2.0.0) using OpenTelemetry instrumentation.
"""

import importlib.metadata as _meta
import logging

from packaging.version import Version

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracing.utils.otlp import MLFLOW_EXPERIMENT_ID_HEADER

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


def _setup_otel_instrumentation() -> None:
    """Set up OpenTelemetry instrumentation for Agno V2."""
    global _agno_instrumentor

    if _agno_instrumentor is not None:
        _logger.debug("OpenTelemetry instrumentation already set up for Agno V2")
        return

    try:
        from openinference.instrumentation.agno import AgnoInstrumentor
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from mlflow.tracking.fluent import _get_experiment_id

        tracking_uri = mlflow.get_tracking_uri()

        tracking_uri = tracking_uri.rstrip("/")
        endpoint = f"{tracking_uri}/v1/traces"

        experiment_id = _get_experiment_id()

        exporter = OTLPSpanExporter(
            endpoint=endpoint, headers={MLFLOW_EXPERIMENT_ID_HEADER: experiment_id}
        )

        tracer_provider = trace.get_tracer_provider()
        if not isinstance(tracer_provider, TracerProvider):
            tracer_provider = TracerProvider()
            trace.set_tracer_provider(tracer_provider)

        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

        _agno_instrumentor = AgnoInstrumentor()
        _agno_instrumentor.instrument()
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
