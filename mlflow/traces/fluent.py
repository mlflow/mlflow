import contextlib
import inspect
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from typing import Any, Dict

from mlflow.traces.client import DummyTraceClient, DummyTraceClientWithHTMLDisplay
from mlflow.traces.export import MLflowSpanExporter, TraceAggregator
from mlflow.traces.types import MLflowSpanWrapper, SpanType
from mlflow.utils.databricks_utils import is_in_databricks_runtime


_TRACER_PROVIDER = TracerProvider()

# TODO: Will move this to factory when we add more exporting options
if is_in_databricks_runtime():
    client = DummyTraceClientWithHTMLDisplay()
else:
    client = DummyTraceClient()
aggregator = TraceAggregator()
exporter = MLflowSpanExporter(client, aggregator)
_TRACER_PROVIDER.add_span_processor(SimpleSpanProcessor(exporter))


@contextlib.contextmanager
def start_span(name=None,
               span_type=SpanType.UNKNOWN,
               attributes=None):
    called_module = "MLflowTraceTest" # TODO: Implement using inspect module
    tracer = _TRACER_PROVIDER.get_tracer(called_module)

    # Setting end_on_exit = False to suppress the default span
    # export and instead invoke MLflowSpanWrapper.end()
    with tracer.start_as_current_span(name, end_on_exit=False) as raw_span:
        mlflow_span = MLflowSpanWrapper(raw_span, span_type=span_type)
        mlflow_span.set_attributes(attributes or {})

        try:
            yield mlflow_span
        finally:
            mlflow_span.end()


# Decorator that wraps a function with start_span() context.
# This has _func as a first parameter so it can also wraps a function directly.
def trace(_func=None,
          name=None,
          span_type=SpanType.UNKNOWN,
          attributes=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            with start_span(name=span_name, span_type=span_type, attributes=attributes) as span:
                span.set_attribute("function_name", func.__name__)
                span.set_inputs(_capture_input_args(func, args, kwargs))
                result = func(*args, **kwargs)
                span.set_outputs({"output": result})
                return result

        return wrapper

    if _func is not None:
        return decorator(_func)

    return decorator


def _capture_input_args(func, args, kwargs) -> Dict[str, Any]:
    # Avoid capturing `self`
    func_signature = inspect.signature(func)
    bound_arguments = func_signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    # Remove `self` from bound arguments if it exists
    if bound_arguments.arguments.get("self"):
        del bound_arguments.arguments["self"]

    return bound_arguments.arguments