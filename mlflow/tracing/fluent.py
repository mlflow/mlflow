from __future__ import annotations

import contextlib
import functools
import logging
from typing import Any, Callable, Dict, List, Optional

from opentelemetry import trace as trace_api

from mlflow.entities import SpanType, Trace
from mlflow.tracing.provider import get_tracer
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.types.wrapper import MlflowSpanWrapper, NoOpMlflowSpanWrapper
from mlflow.tracing.utils import capture_function_input_args

_logger = logging.getLogger(__name__)


def trace(
    func: Optional[Callable] = None,
    name: Optional[str] = None,
    span_type: str = SpanType.UNKNOWN,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    A decorator that creates a new span for the decorated function.

    When you decorate a function with this :py:func:`@mlflow.trace() <trace>` decorator,
    a span will be created for the scope of the decorated function. The span will automatically
    capture the input and output of the function. When it is applied to a method, it doesn't
    capture the `self` argument. Any exception raised within the function will set the span
    status to ``ERROR`` and detailed information such as exception message and stacktrace
    will be recorded to the ``attributes`` field of the span.

    For example, the following code will yield a span with the name ``"my_function"``,
    capturing the input arguments ``x`` and ``y``, and the output of the function.

    .. code-block:: python

        @mlflow.trace
        def my_function(x, y):
            return x + y

    This is equivalent to doing the following using the :py:func:`mlflow.start_span` context
    manager, but requires less boilerplate code.

    .. code-block:: python

        def my_function(x, y):
            return x + y


        with mlflow.start_span("my_function") as span:
            span.set_inputs({"x": x, "y": y})
            result = my_function(x, y)
            span.set_outputs({"output": result})


    .. tip::

        The @mlflow.trace decorator is useful when you want to trace a function defined by
        yourself. However, you may also want to trace a function in external libraries. In
        such case, you can use this ``mlflow.trace()`` function to directly wrap the function,
        instead of using as the decorator. This will create the exact same span as the
        one created by the decorator i.e. captures information from the function call.

        .. code-block:: python

            from some.external.library import predict

            mlflow.trace(predict)(1, 2)

    Args:
        func: The function to be decorated. Must **not** be provided when using as a decorator.
        name: The name of the span. If not provided, the name of the function will be used.
        span_type: The type of the span. Can be either a string or a
            :py:class:`SpanType <mlflow.entities.SpanType>` enum value.
        attributes: A dictionary of attributes to set on the span.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            span_name = name or fn.__name__

            with start_span(name=span_name, span_type=span_type, attributes=attributes) as span:
                span.set_attribute("function_name", fn.__name__)
                span.set_inputs(capture_function_input_args(fn, args, kwargs))
                result = fn(*args, **kwargs)
                span.set_outputs(result)
                return result

        return wrapper

    return decorator(func) if func else decorator


@contextlib.contextmanager
def start_span(
    name: str = "span",
    span_type: Optional[str] = SpanType.UNKNOWN,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Context manager to create a new span and start it as the current span in the context.

    This context manager automatically manages the span lifecycle and parent-child relationships.
    The span will be ended when the context manager exits. Any exception raised within the
    context manager will set the span status to ``ERROR``, and detailed information such as
    exception message and stacktrace will be recorded to the ``attributes`` field of the span.
    New spans can be created within the context manager, then they will be assigned as child
    spans.

    .. code-block:: python

        with mlflow.start_span("my_span") as span:
            span.set_inputs({"x": 1, "y": 2})

            z = x + y

            span.set_outputs(z)
            span.set_attribute("key", "value")
            # do something

    When this context manager is used in the top-level scope, i.e. not within another span context,
    the span will be treated as a root span. The root span doesn't have a parent reference and
    **the entire trace will be logged when the root span is ended**.


    .. tip::

        If you want more explicit control over the trace lifecycle, you can use
        :py:func:`MLflow Client APIs <mlflow.client.MlflowClient.start_trace>`. It provides lower
        level to start and end traces manually, as well as setting the parent spans explicitly.
        However, it is generally recommended to use this context manager as long as it satisfies
        your requirements, because it requires less boilerplate code and is less error-prone.

    .. note::

        The context manager doesn't propagate the span context across threads. If you want to create
        a child span in a different thread, you should use
        :py:func:`MLflow Client APIs <mlflow.client.MlflowClient.start_trace>`
        and pass the parent span ID explicitly.

    .. note::

        All spans created under the root span (i.e. a single trace) are buffered in memory and
        not exported until the root span is ended. The buffer has a default size of 1000 traces
        and TTL of 1 hour. You can configure the buffer size and TTL using the environment variables
        ``MLFLOW_TRACE_BUFFER_MAX_SIZE`` and ``MLFLOW_TRACE_BUFFER_TTL_SECONDS`` respectively.

    Args:
        name: The name of the span.
        span_type: The type of the span. Can be either a string or
            a :py:class:`SpanType <mlflow.entities.SpanType>` enum value
        attributes: A dictionary of attributes to set on the span.

    Returns:
        Yields an :py:class:`mlflow.tracing.MlflowSpanWrapper` that represents the created span.
    """
    # TODO: refactor this logic
    try:
        tracer = get_tracer(__name__)
        span = tracer.start_span(name)
    except Exception:
        _logger.warning(f"Failed to start span with name {name}.")
        span = None

    try:
        if span is not None:
            trace_manager = InMemoryTraceManager.get_instance()
            # Setting end_on_exit = False to suppress the default span
            # export and instead invoke MlflowSpanWrapper.end()
            with trace_api.use_span(span, end_on_exit=False):
                mlflow_span = MlflowSpanWrapper(span, span_type=span_type)
                mlflow_span.set_attributes(attributes or {})
                trace_manager.add_or_update_span(mlflow_span)
                yield mlflow_span
        else:
            # Span creation should not raise an exception
            mlflow_span = NoOpMlflowSpanWrapper()
            yield mlflow_span
    finally:
        mlflow_span.end()


def get_traces(n: int = 1) -> List[Trace]:
    """
    Get the last n traces.

    Args:
        n: The number of traces to return.

    Returns:
        A list of :py:class:`mlflow.entities.Trace` objects.
    """
    from mlflow.tracing.clients import get_trace_client

    return get_trace_client().get_traces(n)


def get_current_active_span():
    """
    Get the current active span in the global context.

    .. attention::

        This only works when the span is created with fluent APIs like `@mlflow.trace` or
        `with mlflow.start_span`. If a span is created with MlflowClient APIs, it won't be
        attached to the global context so this function will not return it.

    Returns:
        The current active span if exists, otherwise None.
    """
    otel_span = trace_api.get_current_span()
    if otel_span is None:
        return None

    span_context = otel_span.get_span_context()
    trace_manager = InMemoryTraceManager.get_instance()
    return trace_manager.get_span_from_id(span_context.trace_id, span_context.span_id)
