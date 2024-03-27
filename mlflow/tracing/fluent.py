from __future__ import annotations

import contextlib
import logging
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import trace as trace_api

from mlflow.tracing.provider import get_tracer
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.types.wrapper import MLflowSpanWrapper, NoOpMLflowSpanWrapper
from mlflow.tracing.utils import capture_function_input_args

_logger = logging.getLogger(__name__)


def trace(
    _func: Optional[Callable] = None,
    name: Optional[str] = None,
    span_type: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Decorator that create a new span for the decorated function.

    The span will automatically captures the input and output of the function. When it is applied to
    a method, it doesn't capture the `self` argument.

    For example, the following code will yield a span with the name "my_function", capturing the
    input arguments `x` and `y`, and the output of the function.

    .. code-block:: python

        @mlflow.trace
        def my_function(x, y):
            return x + y

    Also this can be directly applied to a function call like this:

    .. code-block:: python

        def my_function(x, y):
            return x + y


        mlflow.trace(my_function)(1, 2)

    This works same as the previous example, but can be useful when you want to trace a function
    that is not defined by yourself.

    Args:
        _func: The function to be decorated. Must not be provided when using as a decorator.
        name: The name of the span. If not provided, the name of the function will be used.
        span_type: The type of the span. Can be either a string or a SpanType enum value.
        attributes: A dictionary of attributes to set on the span.
        tags: A string tag that can be attached to the span.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__

            with start_span(name=span_name, span_type=span_type, attributes=attributes) as span:
                if span:
                    span.set_attribute("function_name", func.__name__)
                    span.set_inputs(capture_function_input_args(func, args, kwargs))
                    result = func(*args, **kwargs)
                    span.set_outputs({"output": result})
                    return result
                else:
                    # If span creation fails, just call the function without tracing
                    return func(*args, **kwargs)

        return wrapper

    if _func is not None:
        return decorator(_func)

    return decorator


@contextlib.contextmanager
def start_span(
    name: str = "span", span_type: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None
):
    """
    Context manager to create a new span and start it as the current span in the context.

    Example:

    .. code-block:: python

        with mlflow.start_span("my_span") as span:
            span.set_inputs({"x": 1, "y": 2})
            z = x + y
            span.set_outputs({"z": z})
            span.set_attribute("key", "value")
            # do something

    Args:
        name: The name of the span.
        span_type: The type of the span. Can be either a string or a SpanType enum value.
        attributes: A dictionary of attributes to set on the span.
    """
    # TODO: refactor this logic
    try:
        tracer = get_tracer(__name__)
        span = tracer.start_span(name)
        span.set_attributes(attributes or {})
    except Exception:
        _logger.warning(f"Failed to start span with name {name}.")
        span = None

    try:
        if span is not None:
            trace_manager = InMemoryTraceManager.get_instance()
            # Setting end_on_exit = False to suppress the default span
            # export and instead invoke MLflowSpanWrapper.end()
            with trace_api.use_span(span, end_on_exit=False):
                mlflow_span = MLflowSpanWrapper(span, span_type=span_type)
                mlflow_span.set_attributes(attributes or {})
                trace_manager.add_or_update_span(mlflow_span)
                yield mlflow_span
        else:
            # Span creation should not raise an exception
            mlflow_span = NoOpMLflowSpanWrapper()
            yield mlflow_span
    finally:
        mlflow_span.end()


def get_traces(n: int = 1) -> List:
    """
    Get the last n traces.

    Args:
        n: The number of traces to return.

    Returns:
        A list of Trace objects.
    """
    from mlflow.tracing.clients import get_trace_client

    return get_trace_client().get_traces(n)
