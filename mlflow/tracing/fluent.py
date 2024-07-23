from __future__ import annotations

import contextlib
import functools
import importlib
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional

from cachetools import TTLCache
from opentelemetry import trace as trace_api

from mlflow import MlflowClient
from mlflow.entities import NoOpSpan, SpanType, Trace
from mlflow.entities.span import LiveSpan, create_mlflow_span
from mlflow.environment_variables import (
    MLFLOW_TRACE_BUFFER_MAX_SIZE,
    MLFLOW_TRACE_BUFFER_TTL_SECONDS,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.store.tracking import SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.tracing import provider
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.display import get_display_handler
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import (
    SPANS_COLUMN_NAME,
    capture_function_input_args,
    encode_span_id,
    get_otel_attribute,
)
from mlflow.tracing.utils.search import extract_span_inputs_outputs, traces_to_df
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.utils import get_results_from_paginated_fn
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import is_in_databricks_model_serving_environment

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas


# Traces are stored in memory after completion so they can be retrieved conveniently.
# For example, Databricks model serving fetches the trace data from the buffer after
# making the prediction request, and logging them into the Inference Table.
TRACE_BUFFER = TTLCache(
    maxsize=MLFLOW_TRACE_BUFFER_MAX_SIZE.get(),
    ttl=MLFLOW_TRACE_BUFFER_TTL_SECONDS.get(),
)


@experimental
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
        :test:

        import mlflow


        @mlflow.trace
        def my_function(x, y):
            return x + y

    This is equivalent to doing the following using the :py:func:`mlflow.start_span` context
    manager, but requires less boilerplate code.

    .. code-block:: python
        :test:

        import mlflow


        def my_function(x, y):
            return x + y


        with mlflow.start_span("my_function") as span:
            x = 1
            y = 2
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
            :test:

            import math

            import mlflow

            mlflow.trace(math.factorial)(5)

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
                span.set_attribute(SpanAttributeKey.FUNCTION_NAME, fn.__name__)
                try:
                    span.set_inputs(capture_function_input_args(fn, args, kwargs))
                except Exception:
                    _logger.warning(f"Failed to capture inputs for function {fn.__name__}.")
                result = fn(*args, **kwargs)
                span.set_outputs(result)
                return result

        return wrapper

    return decorator(func) if func else decorator


@experimental
@contextlib.contextmanager
def start_span(
    name: str = "span",
    span_type: Optional[str] = SpanType.UNKNOWN,
    attributes: Optional[Dict[str, Any]] = None,
) -> Generator[LiveSpan, None, None]:
    """
    Context manager to create a new span and start it as the current span in the context.

    This context manager automatically manages the span lifecycle and parent-child relationships.
    The span will be ended when the context manager exits. Any exception raised within the
    context manager will set the span status to ``ERROR``, and detailed information such as
    exception message and stacktrace will be recorded to the ``attributes`` field of the span.
    New spans can be created within the context manager, then they will be assigned as child
    spans.

    .. code-block:: python
        :test:

        import mlflow

        with mlflow.start_span("my_span") as span:
            x = 1
            y = 2
            span.set_inputs({"x": x, "y": y})

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
        Yields an :py:class:`mlflow.entities.Span` that represents the created span.
    """
    try:
        otel_span = provider.start_span_in_context(name)

        # Create a new MLflow span and register it to the in-memory trace manager
        request_id = get_otel_attribute(otel_span, SpanAttributeKey.REQUEST_ID)
        mlflow_span = create_mlflow_span(otel_span, request_id, span_type)
        mlflow_span.set_attributes(attributes or {})
        InMemoryTraceManager.get_instance().register_span(mlflow_span)

    except Exception as e:
        _logger.warning(
            f"Failed to start span: {e}. For full traceback, set logging level to debug.",
            exc_info=_logger.isEnabledFor(logging.DEBUG),
        )
        mlflow_span = NoOpSpan()
        yield mlflow_span
        return

    try:
        # Setting end_on_exit = False to suppress the default span
        # export and instead invoke MLflow span's end() method.
        with trace_api.use_span(mlflow_span._span, end_on_exit=False):
            yield mlflow_span
    finally:
        try:
            mlflow_span.end()
        except Exception as e:
            _logger.warning(
                f"Failed to end span {mlflow_span.span_id}: {e}. "
                "For full traceback, set logging level to debug.",
                exc_info=_logger.isEnabledFor(logging.DEBUG),
            )


@experimental
def get_trace(request_id: str) -> Optional[Trace]:
    """
    Get a trace by the given request ID if it exists.

    This function retrieves the trace from the in-memory buffer first, and if it doesn't exist,
    it fetches the trace from the tracking store. If the trace is not found in the tracking store,
    it returns None.

    Args:
        request_id: The request ID of the trace.


    .. code-block:: python
        :test:

        import mlflow


        with mlflow.start_span(name="span") as span:
            span.set_attribute("key", "value")

        trace = mlflow.get_trace(span.request_id)
        print(trace)


    Returns:
        A :py:class:`mlflow.entities.Trace` objects with the given request ID.
    """
    # Try to get the trace from the in-memory buffer first
    if trace := TRACE_BUFFER.get(request_id, None):
        return trace

    try:
        return MlflowClient().get_trace(request_id, display=False)
    except MlflowException as e:
        _logger.warning(
            f"Failed to get trace from the tracking store: {e}"
            "For full traceback, set logging level to debug.",
            exc_info=_logger.isEnabledFor(logging.DEBUG),
        )
        return None


@experimental
def search_traces(
    experiment_ids: Optional[List[str]] = None,
    filter_string: Optional[str] = None,
    max_results: Optional[int] = None,
    order_by: Optional[List[str]] = None,
    extract_fields: Optional[List[str]] = None,
) -> "pandas.DataFrame":
    """
    Return traces that match the given list of search expressions within the experiments.

    .. tip::

        This API returns a **Pandas DataFrame** that contains the traces as rows. To retrieve
        a list of the original :py:class:`Trace <mlflow.entities.Trace>` objects,
        you can use the :py:meth:`MlflowClient().search_traces
        <mlflow.client.MlflowClient.search_traces>` method instead.

    Args:
        experiment_ids: List of experiment ids to scope the search. If not provided, the search
            will be performed across the current active experiment.
        filter_string: A search filter string.
        max_results: Maximum number of traces desired. If None, all traces matching the search
            expressions will be returned.
        order_by: List of order_by clauses.
        extract_fields: Specify fields to extract from traces using the format
            ``"span_name.[inputs|outputs].field_name"`` or ``"span_name.[inputs|outputs]"``.
            For instance, ``"predict.outputs.result"`` retrieves the output ``"result"`` field from
            a span named ``"predict"``, while ``"predict.outputs"`` fetches the entire outputs
            dictionary, including keys ``"result"`` and ``"explanation"``.

            By default, no fields are extracted into the DataFrame columns. When multiple
            fields are specified, each is extracted as its own column. If an invalid field
            string is provided, the function silently returns without adding that field's column.
            The supported fields are limited to ``"inputs"`` and ``"outputs"`` of spans. If the
            span name or field name contains a dot it must be enclosed in backticks. For example:

            .. code-block:: python

                # span name contains a dot
                extract_fields = ["`span.name`.inputs.field"]

                # field name contains a dot
                extract_fields = ["span.inputs.`field.name`"]

                # span name and field name contain a dot
                extract_fields = ["`span.name`.inputs.`field.name`"]

    Returns:
        A Pandas DataFrame containing information about traces that satisfy the search expressions.

    .. code-block:: python
        :test:
        :caption: Search traces with extract_fields

        import mlflow

        with mlflow.start_span(name="span1") as span:
            span.set_inputs({"a": 1, "b": 2})
            span.set_outputs({"c": 3, "d": 4})

        mlflow.search_traces(
            extract_fields=["span1.inputs", "span1.outputs", "span1.outputs.c"]
        )


    .. code-block:: python
        :test:
        :caption: Search traces with extract_fields and non-dictionary span inputs and outputs

        import mlflow

        with mlflow.start_span(name="non_dict_span") as span:
            span.set_inputs(["a", "b"])
            span.set_outputs([1, 2, 3])

        mlflow.search_traces(
            extract_fields=["non_dict_span.inputs", "non_dict_span.outputs"],
        )
    """
    # Check if pandas is installed early to avoid unnecessary computation
    if importlib.util.find_spec("pandas") is None:
        raise MlflowException(
            message=(
                "The `pandas` library is not installed. Please install `pandas` to use"
                "`mlflow.search_traces` function."
            ),
        )

    if not experiment_ids:
        if experiment_id := _get_experiment_id():
            experiment_ids = [experiment_id]
        else:
            raise MlflowException(
                "No active experiment found. Set an experiment using `mlflow.set_experiment`, "
                "or specify the list of experiment IDs in the `experiment_ids` parameter."
            )

    def pagination_wrapper_func(number_to_get, next_page_token):
        return MlflowClient().search_traces(
            experiment_ids=experiment_ids,
            max_results=number_to_get,
            filter_string=filter_string,
            order_by=order_by,
            page_token=next_page_token,
        )

    results = get_results_from_paginated_fn(
        pagination_wrapper_func,
        max_results_per_page=SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        max_results=max_results,
    )

    get_display_handler().display_traces(results)

    traces_df = traces_to_df(results)
    if extract_fields:
        traces_df = extract_span_inputs_outputs(
            traces=traces_df,
            fields=extract_fields,
            col_name=SPANS_COLUMN_NAME,
        )

    return traces_df


@experimental
def get_current_active_span() -> Optional[LiveSpan]:
    """
    Get the current active span in the global context.

    .. attention::

        This only works when the span is created with fluent APIs like `@mlflow.trace` or
        `with mlflow.start_span`. If a span is created with MlflowClient APIs, it won't be
        attached to the global context so this function will not return it.


    .. code-block:: python
        :test:

        import mlflow


        @mlflow.trace
        def f():
            span = mlflow.get_current_active_span()
            span.set_attribute("key", "value")
            return 0


        f()

    Returns:
        The current active span if exists, otherwise None.
    """
    otel_span = trace_api.get_current_span()
    # NonRecordingSpan is returned if a tracer is not instantiated.
    if otel_span is None or isinstance(otel_span, trace_api.NonRecordingSpan):
        return None

    trace_manager = InMemoryTraceManager.get_instance()
    request_id = json.loads(otel_span.attributes.get(SpanAttributeKey.REQUEST_ID))
    return trace_manager.get_span_from_id(request_id, encode_span_id(otel_span.context.span_id))


@experimental
def get_last_active_trace() -> Optional[Trace]:
    """
    Get the last active trace in the same process if exists.

    .. warning::

        This function DOES NOT work in the model deployed in Databricks model serving.

    .. note::

        The last active trace is only stored in-memory for the time defined by the TTL
        (Time To Live) configuration. By default, the TTL is 1 hour and can be configured
        using the environment variable ``MLFLOW_TRACE_BUFFER_TTL_SECONDS``.

    .. note::

        This function returns an immutable copy of the original trace that is logged
        in the tracking store. Any changes made to the returned object will not be reflected
        in the original trace. To modify the already ended trace (while most of the data is
        immutable after the trace is ended, you can still edit some fields such as `tags`),
        please use the respective MlflowClient APIs with the request ID of the trace, as
        shown in the example below.

    .. code-block:: python
        :test:

        import mlflow


        @mlflow.trace
        def f():
            pass


        f()

        trace = mlflow.get_last_active_trace()


        # Use MlflowClient APIs to mutate the ended trace
        mlflow.MlflowClient().set_trace_tag(trace.info.request_id, "key", "value")

    Returns:
        The last active trace if exists, otherwise None.
    """
    if is_in_databricks_model_serving_environment():
        raise MlflowException(
            "The function `mlflow.get_last_active_trace` is not supported in "
            "Databricks model serving.",
            error_code=BAD_REQUEST,
        )

    if len(TRACE_BUFFER) > 0:
        last_active_request_id = list(TRACE_BUFFER.keys())[-1]
        return TRACE_BUFFER.get(last_active_request_id)
    else:
        return None
