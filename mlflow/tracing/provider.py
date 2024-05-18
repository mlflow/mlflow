import json
import logging
from typing import Optional, Tuple

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util._once import Once

from mlflow.tracing.constant import SpanAttributeKey
from mlflow.utils.databricks_utils import is_in_databricks_model_serving_environment

# Once() object ensures a function is executed only once in a process.
# Note that it doesn't work as expected in a distributed environment.
_TRACER_PROVIDER_INITIALIZED = Once()

_logger = logging.getLogger(__name__)


def start_span_in_context(name: str) -> trace.Span:
    """
    Start a new OpenTelemetry span in the current context.

    Note that this function doesn't set the started span as the active span in the context. To do
    that, the upstream also need to call `use_span()` function in the OpenTelemetry trace APIs.

    Args:
        name: The name of the span.

    Returns:
        The newly created OpenTelemetry span.

    :meta private:
    """
    return _get_tracer(__name__).start_span(name)


def start_detached_span(
    name: str, parent: Optional[trace.Span] = None, experiment_id: Optional[str] = None
) -> Optional[Tuple[str, trace.Span]]:
    """
    Start a new OpenTelemetry span that is not part of the current trace context, but with the
    explicit parent span ID if provided.

    Args:
        name: The name of the span.
        parent: The parent OpenTelemetry span. If not provided, the span will be created as a root
                span.

    Returns:
        The newly created OpenTelemetry span.

    :meta private:
    """
    tracer = _get_tracer(__name__)
    context = trace.set_span_in_context(parent) if parent else None
    attributes = (
        {SpanAttributeKey.EXPERIMENT_ID: json.dumps(experiment_id)} if experiment_id else None
    )
    return tracer.start_span(name, context=context, attributes=attributes)


def _get_tracer(module_name: str):
    """
    Get a tracer instance for the given module name.
    """
    # Initiate tracer provider only once in the application lifecycle
    _TRACER_PROVIDER_INITIALIZED.do_once(_setup_tracer_provider)

    tracer_provider = trace.get_tracer_provider()
    return tracer_provider.get_tracer(module_name)


def _setup_tracer_provider():
    """
    Instantiate a tracer provider and set it as the global tracer provider.
    """
    if is_in_databricks_model_serving_environment():
        from mlflow.tracing.export.inference_table import InferenceTableSpanExporter
        from mlflow.tracing.processor.inference_table import InferenceTableSpanProcessor

        exporter = InferenceTableSpanExporter()
        processor = InferenceTableSpanProcessor(exporter)
    else:
        from mlflow.tracing.export.mlflow import MlflowSpanExporter
        from mlflow.tracing.processor.mlflow import MlflowSpanProcessor

        exporter = MlflowSpanExporter()
        processor = MlflowSpanProcessor(exporter)

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)


def disable():
    """
    Disable tracing by setting the global tracer provider to NoOpTracerProvider.

    :meta private:
    """
    if isinstance(trace.get_tracer_provider(), trace.NoOpTracerProvider):
        _logger.info("Tracing is already disabled")
        return

    with trace._TRACER_PROVIDER_SET_ONCE._lock:
        trace._TRACER_PROVIDER_SET_ONCE._done = False

    trace.set_tracer_provider(trace.NoOpTracerProvider())


def enable():
    """
    Enable tracing by setting the global tracer provider to the actual tracer provider.

    :meta private:
    """
    from mlflow.tracing.provider import _setup_tracer_provider

    if not isinstance(trace.get_tracer_provider(), trace.NoOpTracerProvider):
        _logger.info("Tracing is already enabled")
        return

    with trace._TRACER_PROVIDER_SET_ONCE._lock:
        trace._TRACER_PROVIDER_SET_ONCE._done = False

    _setup_tracer_provider()
