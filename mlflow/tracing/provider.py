from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.util._once import Once

from mlflow.tracing.clients import TraceClient, get_trace_client
from mlflow.tracing.export.mlflow import MlflowSpanExporter

# Once() object ensures a function is executed only once in a process.
# Note that it doesn't work as expected in a distributed environment.
_TRACER_PROVIDER_INITIALIZED = Once()


def get_tracer(module_name: str):
    """
    Get a tracer instance for the given module name.
    """
    # Initiate tracer provider only once in the application lifecycle
    _TRACER_PROVIDER_INITIALIZED.do_once(_setup_tracer_provider)

    tracer_provider = trace.get_tracer_provider()
    return tracer_provider.get_tracer(module_name)


def _setup_tracer_provider(client: Optional[TraceClient] = None):
    """
    Instantiate a tracer provider and set it as the global tracer provider.
    """
    client = client or get_trace_client()

    # TODO: Make factory method for exporters once we support more sink destinations
    exporter = MlflowSpanExporter(client)

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)
