import contextlib
import inspect
from typing import Union

from mlflow.tracing.provider import get_tracer
from mlflow.tracing.types.model import SpanType, StatusCode
from mlflow.tracing.types.wrapper import MLflowSpanWrapper
from mlflow.tracing.utils import capture_function_input_args


def trace(_func=None, name=None, span_type=SpanType.UNKNOWN, attributes=None):
    """
    Decorator that create a new span for the decorated function.

    The span will automatically captures the input and output of the function. When it is applied to
    a method, it doesn't capture the `self` argument.

    For example, the following code will yield a span with the name "my_function", capturing the
    input arguments `x` and `y`, and the output of the function.
    ```
    @mlflow.trace
    def my_function(x, y):
        return x + y
    ```

    Also this can be directly applied to a function call like this:
    ```
    def my_function(x, y):
        return x + y


    mlflow.trace(my_function)(1, 2)
    ```
    This works same as the previous example, but can be useful when you want to trace a function
    that is not defined by yourself.

    Args:
        _func: The function to be decorated. Must not be provided when using as a decorator.
        name: The name of the span. If not provided, the name of the function will be used.
        span_type: The type of the span. Can be either a string or a SpanType enum value.
        attributes: A dictionary of attributes to set on the span.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            with start_span(name=span_name, span_type=span_type, attributes=attributes) as span:
                span.set_attribute("function_name", func.__name__)
                span.set_inputs(capture_function_input_args(func, args, kwargs))
                result = func(*args, **kwargs)
                span.set_outputs({"output": result})
                return result

        return wrapper

    if _func is not None:
        return decorator(_func)

    return decorator


@contextlib.contextmanager
def start_span(
    name: str = "span", span_type: Union[str, SpanType] = SpanType.UNKNOWN, attributes=None
):
    """
    Context manager to create a new span and start it as the current span in the context.

    Example:
    ```
    with mlflow.start_span("my_span") as span:
        span.set_inputs({"x": 1, "y": 2})
        z = x + y
        span.set_outputs({"z": z})
        span.set_attribute("key", "value")
        # do something
    ```

    Args:
        name: The name of the span.
        span_type: The type of the span. Can be either a string or a SpanType enum value.
        attributes: A dictionary of attributes to set on the span.
    """
    caller_module = inspect.getmodule(inspect.currentframe().f_back)
    module_name = caller_module.__name__ if caller_module else "unknown_module"
    tracer = get_tracer(module_name)

    # Setting end_on_exit = False to suppress the default span
    # export and instead invoke MLflowSpanWrapper.end()
    try:
        with tracer.start_as_current_span(name, end_on_exit=False) as raw_span:
            mlflow_span = MLflowSpanWrapper(raw_span, span_type=span_type)
            mlflow_span.set_attributes(attributes or {})
            yield mlflow_span
    finally:
        # NB: In OpenTelemetry, status code remains UNSET if not explicitly set
        # by the user. However, there is not way to set the status when using
        # @mlflow.trace decorator. Therefore, we just automatically set the status
        # to OK if it is not ERROR.
        if mlflow_span.status.status_code != StatusCode.ERROR:
            mlflow_span.set_status(StatusCode.OK)

        mlflow_span.end()
