from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Generator, Optional, Union

from cachetools import TTLCache
from opentelemetry import trace as trace_api

from mlflow import MlflowClient
from mlflow.entities import NoOpSpan, SpanType, Trace
from mlflow.entities.span import LiveSpan, create_mlflow_span
from mlflow.entities.trace_status import TraceStatus
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
from mlflow.tracing.provider import is_tracing_enabled
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


def trace(
    func: Optional[Callable] = None,
    name: Optional[str] = None,
    span_type: str = SpanType.UNKNOWN,
    attributes: Optional[dict[str, Any]] = None,
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

    class _WrappingContext:
        # define the wrapping logic as a coroutine to avoid code duplication
        # between sync and async cases
        @staticmethod
        def _wrapping_logic(fn, args, kwargs):
            span_name = name or fn.__name__

            with start_span(name=span_name, span_type=span_type, attributes=attributes) as span:
                span.set_attribute(SpanAttributeKey.FUNCTION_NAME, fn.__name__)
                try:
                    span.set_inputs(capture_function_input_args(fn, args, kwargs))
                except Exception:
                    _logger.warning(f"Failed to capture inputs for function {fn.__name__}.")
                result = yield  # sync/async function output to be sent here
                span.set_outputs(result)
                yield result

        def __init__(self, fn, args, kwargs):
            self.coro = self._wrapping_logic(fn, args, kwargs)

        def __enter__(self):
            next(self.coro)
            return self.coro

        def __exit__(self, exc_type, exc_value, traceback):
            # Since the function call occurs outside the coroutine,
            # if an exception occurs, we need to throw it back in, so that
            # we return control to the coro (in particular, so that the __exit__'s
            # of start_span and OTel's use_span can execute).
            if exc_type is not None:
                self.coro.throw(exc_type, exc_value, traceback)
            self.coro.close()

    def decorator(fn):
        if inspect.iscoroutinefunction(fn):

            async def wrapper(*args, **kwargs):
                with _WrappingContext(fn, args, kwargs) as wrapping_coro:
                    return wrapping_coro.send(await fn(*args, **kwargs))
        else:

            def wrapper(*args, **kwargs):
                with _WrappingContext(fn, args, kwargs) as wrapping_coro:
                    return wrapping_coro.send(fn(*args, **kwargs))

        return functools.wraps(fn)(wrapper)

    return decorator(func) if func else decorator


@contextlib.contextmanager
def start_span(
    name: str = "span",
    span_type: Optional[str] = SpanType.UNKNOWN,
    attributes: Optional[dict[str, Any]] = None,
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

    except Exception:
        _logger.debug(f"Failed to start span {name}.", exc_info=True)
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
        except Exception:
            _logger.debug(f"Failed to end span {mlflow_span.span_id}.", exc_info=True)


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


def search_traces(
    experiment_ids: Optional[list[str]] = None,
    filter_string: Optional[str] = None,
    max_results: Optional[int] = None,
    order_by: Optional[list[str]] = None,
    extract_fields: Optional[list[str]] = None,
    run_id: Optional[str] = None,
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
        run_id: A run id to scope the search. When a trace is created under an active run,
            it will be associated with the run and you can filter on the run id to retrieve the
            trace. See the example below for how to filter traces by run id.

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

    .. code-block:: python
        :test:
        :caption: Search traces by run ID

        import mlflow


        @mlflow.trace
        def traced_func(x):
            return x + 1


        with mlflow.start_run() as run:
            traced_func(1)

        mlflow.search_traces(run_id=run.info.run_id)

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
            run_id=run_id,
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


def update_current_trace(
    tags: Optional[dict[str, str]] = None,
):
    """
    Update the current active trace with the given tags.

    You can use this function either within a function decorated with `@mlflow.trace` or within the
    scope of the `with mlflow.start_span` context manager. If there is no active trace found, this
    function will raise an exception.

    Using within a function decorated with `@mlflow.trace`:

    .. code-block:: python

        @mlflow.trace
        def my_func(x):
            mlflow.update_current_trace(tags={"fruit": "apple"})
            return x + 1

    Using within the `with mlflow.start_span` context manager:

    .. code-block:: python

        with mlflow.start_span("span"):
            mlflow.update_current_trace(tags={"fruit": "apple"})

    """
    active_span = get_current_active_span()

    if not active_span:
        raise MlflowException(
            "No active trace found. Please create a span using `mlflow.start_span` or "
            "`@mlflow.trace` before calling this function.",
            error_code=BAD_REQUEST,
        )

    # Update tags for the trace stored in-memory rather than directly updating the
    # backend store. The in-memory trace will be exported when it is ended. By doing
    # this, we can avoid unnecessary server requests for each tag update.
    request_id = active_span.request_id
    with InMemoryTraceManager.get_instance().get_trace(request_id) as trace:
        trace.info.tags.update(tags or {})


@experimental
def add_trace(trace: Union[Trace, dict[str, Any]], target: Optional[LiveSpan] = None):
    """
    Add a completed trace object into another trace.

    This is particularly useful when you call a remote service instrumented by
    MLflow Tracing. By using this function, you can merge the trace from the remote
    service into the current active local trace, so that you can see the full
    trace including what happens inside the remote service call.

    The following example demonstrates how to use this function to merge a trace from a remote
    service to the current active trace in the function.

    .. code-block:: python

        @mlflow.trace(name="predict")
        def predict(input):
            # Call a remote service that returns a trace in the response
            resp = requests.get("https://your-service-endpoint", ...)

            # Extract the trace from the response
            trace_json = resp.json().get("trace")

            # Use the remote trace as a part of the current active trace.
            # It will be merged under the span "predict" and exported together when it is ended.
            mlflow.add_trace(trace_json)

    If you have a specific target span to merge the trace under, you can pass the target span

    .. code-block:: python

        def predict(input):
            # Create a local span
            span = MlflowClient().start_span(name="predict")

            resp = requests.get("https://your-service-endpoint", ...)
            trace_json = resp.json().get("trace")

            # Merge the remote trace under the span created above
            mlflow.add_trace(trace_json, target=span)

    Args:
        trace: A :py:class:`Trace <mlflow.entities.Trace>` object or a dictionary representation
            of the trace. The trace **must** be already completed i.e. no further updates should
            be made to it. Otherwise, this function will raise an exception.

            .. attention:

                The spans in the trace must be ordered in a way that the parent span comes
                before its children. If the spans are not ordered correctly, this function
                will raise an exception.

        target: The target span to merge the given trace.

            - If provided, the trace will be merged under the target span.
            - If not provided, the trace will be merged under the current active span.
            - If not provided and there is no active span, a new span named "Remote Trace <...>"
              will be created and the trace will be merged under it.
    """
    if not is_tracing_enabled():
        _logger.debug("Tracing is disabled. Skipping add_trace.")
        return

    if isinstance(trace, dict):
        try:
            trace = Trace.from_dict(trace)
        except Exception as e:
            raise MlflowException.invalid_parameter_value(
                "Failed to load a trace object from the given dictionary. Please ensure the "
                f"dictionary is in the correct MLflow Trace format. Error: {e}",
            )
    elif not isinstance(trace, Trace):
        raise MlflowException.invalid_parameter_value(
            f"Invalid trace object: {type(trace)}. Please provide a valid MLflow Trace object "
            "to use it as a remote trace. You can create a Trace object from its json format by "
            "using the Trace.from_dict() method."
        )

    if trace.info.status not in TraceStatus.end_statuses():
        raise MlflowException.invalid_parameter_value(
            "The trace must be ended before adding it to another trace. "
            f"Current status: {trace.info.status}.",
        )

    if target_span := target or get_current_active_span():
        _merge_trace(
            trace=trace,
            target_request_id=target_span.request_id,
            target_parent_span_id=target_span.span_id,
        )
    else:
        # If there is no target span, create a new root span named "Remote Trace <...>"
        # and put the remote trace under it. This design aims to keep the trace export
        # logic simpler and consistent, rather than directly exporting the remote trace.
        client = MlflowClient()
        remote_root_span = trace.data.spans[0]
        span = client.start_trace(
            name=f"Remote Trace <{remote_root_span.name}>",
            inputs=remote_root_span.inputs,
            attributes={
                # Exclude request ID attribute not to reuse same request ID
                k: v
                for k, v in remote_root_span.attributes.items()
                if k != SpanAttributeKey.REQUEST_ID
            },
            start_time_ns=remote_root_span.start_time_ns,
        )
        _merge_trace(
            trace=trace,
            target_request_id=span.request_id,
            target_parent_span_id=span.span_id,
        )
        client.end_trace(
            request_id=span.request_id,
            status=trace.info.status,
            outputs=remote_root_span.outputs,
            end_time_ns=remote_root_span.end_time_ns,
        )


def _merge_trace(
    trace: Trace,
    target_request_id: str,
    target_parent_span_id: str,
):
    """
    Merge the given trace object under an existing trace in the in-memory trace registry.

    Args:
        trace: The trace object to be merged.
        target_request_id: The request ID of the parent trace.
        target_parent_span_id: The parent span ID, under which the child trace should be merged.
    """
    trace_manager = InMemoryTraceManager.get_instance()

    # The merged trace should have the same trace ID as the parent trace.
    with trace_manager.get_trace(target_request_id) as parent_trace:
        if not parent_trace:
            _logger.warning(
                f"Parent trace with request ID {target_request_id} not found. Skipping merge."
            )
            return

        new_trace_id = parent_trace.span_dict[target_parent_span_id]._trace_id

    for span in trace.data.spans:
        parent_span_id = span.parent_id or target_parent_span_id

        # NB: We clone span one by one in the order it was saved in the original trace. This
        # works upon the assumption that the parent span always comes before its children.
        # This is guaranteed in current implementation, but if it changes in the future,
        # we have to traverse the tree to determine the order.
        if not trace_manager.get_span_from_id(target_request_id, parent_span_id):
            raise MlflowException.invalid_parameter_value(
                f"Span with ID {parent_span_id} not found. Please make sure the "
                "spans in the trace are ordered correctly i.e. the parent span comes before "
                "its children."
            )

        cloned_span = LiveSpan.from_immutable_span(
            span=span,
            parent_span_id=parent_span_id,
            request_id=target_request_id,
            trace_id=new_trace_id,
        )
        trace_manager.register_span(cloned_span)

    # Merge the tags and metadata from the child trace to the parent trace.
    with trace_manager.get_trace(target_request_id) as parent_trace:
        # Order of merging is important to ensure the parent trace's metadata is
        # not overwritten by the child trace's metadata if they have the same key.
        parent_trace.info.tags = {**trace.info.tags, **parent_trace.info.tags}
        parent_trace.info.request_metadata = {
            **trace.info.request_metadata,
            **parent_trace.info.request_metadata,
        }
