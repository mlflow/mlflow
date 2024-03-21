import contextlib
import logging
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import trace as trace_api

from mlflow.tracing.provider import get_tracer
from mlflow.tracing.types.wrapper import MLflowSpanWrapper, NoOpMLflowSpanWrapper
from mlflow.tracing.utils import capture_function_input_args, get_caller_module

_logger = logging.getLogger(__name__)


def trace(
    _func: Optional[Callable] = None,
    name: Optional[str] = None,
    span_type: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
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
        tracer = get_tracer(get_caller_module())
        span = tracer.start_span(name)
        span.set_attributes(attributes or {})
    except Exception:
        _logger.warning(f"Failed to start span with name {name}.")
        span = None

    try:
        if span is not None:
            # Setting end_on_exit = False to suppress the default span
            # export and instead invoke MLflowSpanWrapper.end()
            with trace_api.use_span(span, end_on_exit=False):
                mlflow_span = MLflowSpanWrapper(span, span_type=span_type)
                mlflow_span.set_attributes(attributes or {})
                yield mlflow_span
        else:
            # Span creation should not raise an exception
            mlflow_span = NoOpMLflowSpanWrapper()
            yield mlflow_span
    finally:
        mlflow_span.end()


def start_detached_span(
    name: str = "span",
    span_type: Optional[str] = None,
    parent: Optional[MLflowSpanWrapper] = None,
):
    """
    Create a new span and start it without attaching it to the global trace context.

    This is an imperative API to manually create a new span under a specific trace id and parent
    span, unlike the higher-level APIs like `@mlflow.trace()` and `with mlflow.start_span()`,
    which automatically manage the span lifecycle and parent-child relationship.

    This API is particularly useful for the case where the automatic context management is not
    sufficient, such as callback-based instrumentation where span start and end are not in the same
    call stack, or multi-threaded applications where the context is not propagated automatically.

    To conclude the span for logging, call `span.end()`.

    Args:
        name: The name of the span.
        span_type: The type of the span. Can be either a string or a SpanType enum value.
        parent: The parent span. If None, the span to be created is a root span.

    Example:

    .. code-block:: python

            span = mlflow.start_detached_span("my_span")

            child_span = mlflow.start_detached_span("child_span", parent=span)
            child_span.set_attributes({"key": "value"})
            child_span.end()

            span.set_attribute("key", "value")
            span.end()
    """
    try:
        tracer = get_tracer(get_caller_module())
        context = trace_api.set_span_in_context(parent._span if parent else None)
        span = tracer.start_span(name=name, context=context)
        return MLflowSpanWrapper(span, span_type=span_type)
    except Exception as e:
        _logger.warning(f"Failed to start span with name {name}: {e}")
        return NoOpMLflowSpanWrapper()
