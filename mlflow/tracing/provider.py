"""
This module provides a set of functions to manage the global tracer provider for MLflow tracing.

Every tracing operation in MLflow *MUST* be managed through this module, instead of directly
using the OpenTelemetry APIs. This is because MLflow needs to control the initialization of the
tracer provider and ensure that it won't interfere with the other external libraries that might
use OpenTelemetry e.g. PromptFlow, Snowpark.
"""

import functools
import json
import logging
from typing import Optional, Tuple

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from mlflow.exceptions import MlflowTracingException
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.utils.exception import raise_as_trace_exception
from mlflow.tracing.utils.once import Once
from mlflow.tracing.utils.otlp import get_otlp_exporter, should_use_otlp_exporter
from mlflow.utils.databricks_utils import (
    is_in_databricks_model_serving_environment,
    is_mlflow_tracing_enabled_in_model_serving,
)

# Global tracer provider instance. We manage the tracer provider by ourselves instead of using
# the global tracer provider provided by OpenTelemetry.
_MLFLOW_TRACER_PROVIDER = None

# Once() object ensures a function is executed only once in a process.
# Note that it doesn't work as expected in a distributed environment.
_MLFLOW_TRACER_PROVIDER_INITIALIZED = Once()

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
    """
    return _get_tracer(__name__).start_span(name)


def start_detached_span(
    name: str,
    parent: Optional[trace.Span] = None,
    experiment_id: Optional[str] = None,
    start_time_ns: Optional[int] = None,
) -> Optional[Tuple[str, trace.Span]]:
    """
    Start a new OpenTelemetry span that is not part of the current trace context, but with the
    explicit parent span ID if provided.

    Args:
        name: The name of the span.
        parent: The parent OpenTelemetry span. If not provided, the span will be created as a root
                span.
        experiment_id: The ID of the experiment. This is used to associate the span with a specific
            experiment in MLflow.
        start_time_ns: The start time of the span in nanoseconds.
            If not provided, the current timestamp is used.

    Returns:
        The newly created OpenTelemetry span.
    """
    tracer = _get_tracer(__name__)
    context = trace.set_span_in_context(parent) if parent else None
    attributes = {}

    # Set start time and experiment to attribute so we can pass it to the span processor
    if start_time_ns:
        attributes[SpanAttributeKey.START_TIME_NS] = json.dumps(start_time_ns)
    if experiment_id:
        attributes[SpanAttributeKey.EXPERIMENT_ID] = json.dumps(experiment_id)
    return tracer.start_span(name, context=context, attributes=attributes, start_time=start_time_ns)


def _get_tracer(module_name: str):
    """
    Get a tracer instance for the given module name.

    If the tracer provider is not initialized, this function will initialize the tracer provider.
    Other simultaneous calls to this function will block until the initialization is done.
    """
    # Initiate tracer provider only once in the application lifecycle
    _MLFLOW_TRACER_PROVIDER_INITIALIZED.do_once(_setup_tracer_provider)
    return _MLFLOW_TRACER_PROVIDER.get_tracer(module_name)


def _get_trace_exporter():
    """
    Get the exporter instance that is used by the current tracer provider.
    """
    if _MLFLOW_TRACER_PROVIDER:
        processors = _MLFLOW_TRACER_PROVIDER._active_span_processor._span_processors
        # There should be only one processor used for MLflow tracing
        processor = processors[0]
        return processor.span_exporter


def _setup_tracer_provider(disabled=False):
    """
    Instantiate a tracer provider and set it as the global tracer provider.

    Note that this function ALWAYS updates the global tracer provider, regardless of the current
    state. It is the caller's responsibility to ensure that the tracer provider is initialized
    only once, and update the _MLFLOW_TRACER_PROVIDER_INITIALIZED flag accordingly.
    """
    global _MLFLOW_TRACER_PROVIDER

    if disabled:
        _MLFLOW_TRACER_PROVIDER = trace.NoOpTracerProvider()
        return

    if should_use_otlp_exporter():
        # Export to OpenTelemetry Collector when configured
        from mlflow.tracing.processor.otel import OtelSpanProcessor

        exporter = get_otlp_exporter()
        processor = OtelSpanProcessor(exporter)

    elif is_in_databricks_model_serving_environment():
        # Export to Inference Table when running in Databricks Model Serving
        if not is_mlflow_tracing_enabled_in_model_serving():
            _MLFLOW_TRACER_PROVIDER = trace.NoOpTracerProvider()
            return

        from mlflow.tracing.export.inference_table import InferenceTableSpanExporter
        from mlflow.tracing.processor.inference_table import InferenceTableSpanProcessor

        exporter = InferenceTableSpanExporter()
        processor = InferenceTableSpanProcessor(exporter)

    else:
        # Default to MLflow Tracking Server
        from mlflow.tracing.export.mlflow import MlflowSpanExporter
        from mlflow.tracing.processor.mlflow import MlflowSpanProcessor

        exporter = MlflowSpanExporter()
        processor = MlflowSpanProcessor(exporter)

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(processor)
    _MLFLOW_TRACER_PROVIDER = tracer_provider


@raise_as_trace_exception
def disable():
    """
    Disable tracing.

    .. note::

        This function sets up `OpenTelemetry` to use
        `NoOpTracerProvider <https://github.com/open-telemetry/opentelemetry-python/blob/4febd337b019ea013ccaab74893bd9883eb59000/opentelemetry-api/src/opentelemetry/trace/__init__.py#L222>`_
        and effectively disables all tracing operations.

    Example:

    .. code-block:: python
        :test:

        import mlflow


        @mlflow.trace
        def f():
            return 0


        # Tracing is enabled by default
        f()
        assert len(mlflow.search_traces()) == 1

        # Disable tracing
        mlflow.tracing.disable()
        f()
        assert len(mlflow.search_traces()) == 1

    """
    if not _is_enabled():
        return

    _setup_tracer_provider(disabled=True)
    _MLFLOW_TRACER_PROVIDER_INITIALIZED.done = True


@raise_as_trace_exception
def enable():
    """
    Enable tracing.

    Example:

    .. code-block:: python
        :test:

        import mlflow


        @mlflow.trace
        def f():
            return 0


        # Tracing is enabled by default
        f()
        assert len(mlflow.search_traces()) == 1

        # Disable tracing
        mlflow.tracing.disable()
        f()
        assert len(mlflow.search_traces()) == 1

        # Re-enable tracing
        mlflow.tracing.enable()
        f()
        assert len(mlflow.search_traces()) == 2

    """
    if _is_enabled() and _MLFLOW_TRACER_PROVIDER_INITIALIZED.done:
        _logger.info("Tracing is already enabled")
        return

    _setup_tracer_provider()
    _MLFLOW_TRACER_PROVIDER_INITIALIZED.done = True


def trace_disabled(f):
    """
    A decorator that temporarily disables tracing for the duration of the decorated function.

    .. code-block:: python

        @trace_disabled
        def f():
            with mlflow.start_span("my_span") as span:
                span.set_attribute("my_key", "my_value")

            return


        # This function will not generate any trace
        f()

    :meta private:
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        is_func_called = False
        result = None
        try:
            if _is_enabled():
                disable()
                try:
                    is_func_called, result = True, f(*args, **kwargs)
                finally:
                    enable()
            else:
                is_func_called, result = True, f(*args, **kwargs)
        # We should only catch the exception from disable() and enable()
        # and let other exceptions propagate.
        except MlflowTracingException as e:
            _logger.warning(
                f"An error occurred while disabling or re-enabling tracing: {e} "
                "The original function will still be executed, but the tracing "
                "state may not be as expected. For full traceback, set "
                "logging level to debug.",
                exc_info=_logger.isEnabledFor(logging.DEBUG),
            )
            # If the exception is raised before the original function
            # is called, we should call the original function
            if not is_func_called:
                result = f(*args, **kwargs)

        return result

    return wrapper


def reset_tracer_setup():
    """
    Reset the flags that indicates whether the MLflow tracer provider has been initialized.
    This ensures that the tracer provider is re-initialized when next tracing
    operation is performed.
    """
    # Set NoOp tracer provider to reset the global tracer to the initial state.
    _setup_tracer_provider(disabled=True)
    # Flip _MLFLOW_TRACE_PROVIDER_INITIALIZED flag to False so that
    # the next tracing operation will re-initialize the provider.
    _MLFLOW_TRACER_PROVIDER_INITIALIZED.done = False


@raise_as_trace_exception
def _is_enabled() -> bool:
    """
    Check if tracing is enabled based on whether the global tracer
    is instantiated or not.

    Trace is considered as "enabled" if the followings
    1. The default state (before any tracing operation)
    2. The tracer is not either ProxyTracer or NoOpTracer
    """
    if not _MLFLOW_TRACER_PROVIDER_INITIALIZED.done:
        return True

    tracer = _get_tracer(__name__)
    # Occasionally ProxyTracer instance wraps the actual tracer
    if isinstance(tracer, trace.ProxyTracer):
        tracer = tracer._tracer
    return not isinstance(tracer, trace.NoOpTracer)
