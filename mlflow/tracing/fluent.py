from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import json
import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal, Optional, Union

from cachetools import TTLCache
from opentelemetry import trace as trace_api

from mlflow.entities import NoOpSpan, SpanType, Trace
from mlflow.entities.span import NO_OP_SPAN_TRACE_ID, LiveSpan, create_mlflow_span
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.entities.trace_state import TraceState
from mlflow.entities.trace_status import TraceStatus
from mlflow.exceptions import MlflowException
from mlflow.store.tracking import SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.tracing import provider
from mlflow.tracing.client import TracingClient
from mlflow.tracing.constant import (
    STREAM_CHUNK_EVENT_NAME_FORMAT,
    STREAM_CHUNK_EVENT_VALUE_KEY,
    SpanAttributeKey,
)
from mlflow.tracing.provider import is_tracing_enabled, safe_set_span_in_context
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import (
    SPANS_COLUMN_NAME,
    TraceJSONEncoder,
    capture_function_input_args,
    encode_span_id,
    exclude_immutable_tags,
    get_otel_attribute,
    set_chat_attributes_special_case,
)
from mlflow.tracing.utils.search import extract_span_inputs_outputs, traces_to_df
from mlflow.tracing.utils.warning import request_id_backward_compatible
from mlflow.utils import get_results_from_paginated_fn
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas


_LAST_ACTIVE_TRACE_ID_GLOBAL = None
_LAST_ACTIVE_TRACE_ID_THREAD_LOCAL = ContextVar("last_active_trace_id", default=None)

# Cache mapping between evaluation request ID to MLflow backend request ID.
# This is necessary for evaluation harness to access generated traces during
# evaluation using the dataset row ID (evaluation request ID).
_EVAL_REQUEST_ID_TO_TRACE_ID = TTLCache(maxsize=10000, ttl=3600)


def trace(
    func: Optional[Callable] = None,
    name: Optional[str] = None,
    span_type: str = SpanType.UNKNOWN,
    attributes: Optional[dict[str, Any]] = None,
    output_reducer: Optional[Callable] = None,
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


    The @mlflow.trace decorator currently support the following types of functions:

    .. list-table:: Supported Function Types
        :widths: 20 30
        :header-rows: 1

        * - Function Type
          - Supported
        * - Sync
          - ✅
        * - Async
          - ✅ (>= 2.16.0)
        * - Generator
          - ✅ (>= 2.20.2)
        * - Async Generator
          - ✅ (>= 2.20.2)
        * - ClassMethod
          - ✅ (>= 3.0.0)
        * - StaticMethod
          - ✅ (>= 3.0.0)

    For more examples of using the @mlflow.trace decorator, including streaming/async
    handling, see the `MLflow Tracing documentation <https://www.mlflow.org/docs/latest/tracing/api/manual-instrumentation#decorator>`_.

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
        output_reducer: A function that reduces the outputs of the generator function into a
            single value to be set as the span output.
    """

    def decorator(fn):
        # Check if the function is a classmethod or staticmethod
        is_classmethod = isinstance(fn, classmethod)
        is_staticmethod = isinstance(fn, staticmethod)

        # Extract the original function if it's a descriptor
        original_fn = fn.__func__ if is_classmethod or is_staticmethod else fn

        # Apply the appropriate wrapper to the original function
        if inspect.isgeneratorfunction(original_fn) or inspect.isasyncgenfunction(original_fn):
            wrapped = _wrap_generator(
                original_fn,
                name,
                span_type,
                attributes,
                output_reducer,
            )
        else:
            if output_reducer is not None:
                raise MlflowException.invalid_parameter_value(
                    "The output_reducer argument is only supported for generator functions."
                )
            wrapped = _wrap_function(original_fn, name, span_type, attributes)

        # If the original was a descriptor, wrap the result back as the same type of descriptor
        if is_classmethod:
            return classmethod(wrapped)
        elif is_staticmethod:
            return staticmethod(wrapped)
        else:
            return wrapped

    return decorator(func) if func else decorator


def _wrap_function(
    fn: Callable,
    name: Optional[str] = None,
    span_type: str = SpanType.UNKNOWN,
    attributes: Optional[dict[str, Any]] = None,
) -> Callable:
    class _WrappingContext:
        # define the wrapping logic as a coroutine to avoid code duplication
        # between sync and async cases
        @staticmethod
        def _wrapping_logic(fn, args, kwargs):
            span_name = name or fn.__name__

            with start_span(name=span_name, span_type=span_type, attributes=attributes) as span:
                span.set_attribute(SpanAttributeKey.FUNCTION_NAME, fn.__name__)
                inputs = capture_function_input_args(fn, args, kwargs)
                span.set_inputs(inputs)
                result = yield  # sync/async function output to be sent here
                span.set_outputs(result)

                set_chat_attributes_special_case(span, inputs=inputs, outputs=result)

                try:
                    yield result
                except GeneratorExit:
                    # Swallow `GeneratorExit` raised when the generator is closed
                    pass

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

    if inspect.iscoroutinefunction(fn):

        async def wrapper(*args, **kwargs):
            with _WrappingContext(fn, args, kwargs) as wrapping_coro:
                return wrapping_coro.send(await fn(*args, **kwargs))
    else:

        def wrapper(*args, **kwargs):
            with _WrappingContext(fn, args, kwargs) as wrapping_coro:
                return wrapping_coro.send(fn(*args, **kwargs))

    return _wrap_function_safe(fn, wrapper)


def _wrap_generator(
    fn: Callable,
    name: Optional[str] = None,
    span_type: str = SpanType.UNKNOWN,
    attributes: Optional[dict[str, Any]] = None,
    output_reducer: Optional[Callable] = None,
) -> Callable:
    """
    Wrap a generator function to create a span.
    Generator functions need special handling because of its lazy evaluation nature.
    Let's say we have a generator function like this:
    ```
    @mlflow.trace
    def generate_stream():
        # B
        for i in range(10):
            # C
            yield i * 2
        # E


    stream = generate_stream()
    # A
    for chunk in stream:
        # D
        pass
    # F
    ```
    The execution order is A -> B -> C -> D -> C -> D -> ... -> E -> F.
    The span should only be "active" at B, C, and E, namely, when the code execution
    is inside the generator function. Otherwise it will create wrong span tree, or
    even worse, leak span context and pollute subsequent traces.
    """

    def _start_stream_span(fn, inputs):
        try:
            return start_span_no_context(
                name=name or fn.__name__,
                parent_span=get_current_active_span(),
                span_type=span_type,
                attributes=attributes,
                inputs=inputs,
            )
        except Exception as e:
            _logger.debug(f"Failed to start stream span: {e}")
            return NoOpSpan()

    def _end_stream_span(
        span: LiveSpan,
        inputs: Optional[dict[str, Any]] = None,
        outputs: Optional[list[Any]] = None,
        output_reducer: Optional[Callable] = None,
        error: Optional[Exception] = None,
    ):
        if error:
            span.add_event(SpanEvent.from_exception(error))
            span.end(status=SpanStatusCode.ERROR)
            return

        if output_reducer:
            try:
                outputs = output_reducer(outputs)
            except Exception as e:
                _logger.debug(f"Failed to reduce outputs from stream: {e}")

        set_chat_attributes_special_case(span, inputs=inputs, outputs=outputs)
        span.end(outputs=outputs)

    def _record_chunk_event(span: LiveSpan, chunk: Any, chunk_index: int):
        try:
            event = SpanEvent(
                name=STREAM_CHUNK_EVENT_NAME_FORMAT.format(index=chunk_index),
                # OpenTelemetry SpanEvent only support str-str key-value pairs for attributes
                attributes={STREAM_CHUNK_EVENT_VALUE_KEY: json.dumps(chunk, cls=TraceJSONEncoder)},
            )
            span.add_event(event)
        except Exception as e:
            _logger.debug(f"Failing to record chunk event for span {span.name}: {e}")

    if inspect.isgeneratorfunction(fn):

        def wrapper(*args, **kwargs):
            inputs = capture_function_input_args(fn, args, kwargs)
            span = _start_stream_span(fn, inputs)
            generator = fn(*args, **kwargs)

            i = 0
            outputs = []
            while True:
                try:
                    # NB: Set the span to active only when the generator is running
                    with safe_set_span_in_context(span):
                        value = next(generator)
                except StopIteration:
                    break
                except Exception as e:
                    _end_stream_span(span, error=e)
                    raise e
                else:
                    outputs.append(value)
                    _record_chunk_event(span, value, i)
                    yield value
                    i += 1
            _end_stream_span(span, inputs, outputs, output_reducer)
    else:

        async def wrapper(*args, **kwargs):
            inputs = capture_function_input_args(fn, args, kwargs)
            span = _start_stream_span(fn, inputs)
            generator = fn(*args, **kwargs)

            i = 0
            outputs = []
            while True:
                try:
                    with safe_set_span_in_context(span):
                        value = await generator.__anext__()
                except StopAsyncIteration:
                    break
                except Exception as e:
                    _end_stream_span(span, error=e)
                    raise e
                else:
                    outputs.append(value)
                    _record_chunk_event(span, value, i)
                    yield value
                    i += 1
            _end_stream_span(span, inputs, outputs, output_reducer)

    return _wrap_function_safe(fn, wrapper)


def _wrap_function_safe(fn: Callable, wrapper: Callable) -> Callable:
    wrapped = functools.wraps(fn)(wrapper)
    # Update the signature of the wrapper to match the signature of the original (safely)
    try:
        wrapped.__signature__ = inspect.signature(fn)
    except Exception:
        pass
    return wrapped


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
        the `mlflow.start_span_no_context()` API. It provides lower
        level to start spans and control the parent-child relationships explicitly.
        However, it is generally recommended to use this context manager as long as it satisfies
        your requirements, because it requires less boilerplate code and is less error-prone.

    .. note::

        The context manager doesn't propagate the span context across threads by default. see
        `Multi Threading <https://mlflow.org/docs/latest/tracing/api/manual-instrumentation#multi-threading>`_
        for how to propagate the span context across threads.

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
        attributes = dict(attributes) if attributes is not None else {}
        mlflow_span.set_attributes(attributes)
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


def start_span_no_context(
    name: str,
    span_type: str = SpanType.UNKNOWN,
    parent_span: Optional[LiveSpan] = None,
    inputs: Optional[Any] = None,
    attributes: Optional[dict[str, str]] = None,
    tags: Optional[dict[str, str]] = None,
    experiment_id: Optional[str] = None,
    start_time_ns: Optional[int] = None,
) -> LiveSpan:
    """
    Start a span without attaching it to the global tracing context.

    This is useful when you want to create a span without automatically linking
    with a parent span and instead manually manage the parent-child relationships.

    The span started with this function must be ended manually using the
    `end()` method of the span object.

    Args:
        name: The name of the span.
        span_type: The type of the span. Can be either a string or
            a :py:class:`SpanType <mlflow.entities.SpanType>` enum value
        parent_span: The parent span to link with. If None, the span will be treated as a root span.
        inputs: The input data for the span.
        attributes: A dictionary of attributes to set on the span.
        tags: A dictionary of tags to set on the trace.
        experiment_id: The experiment ID to associate with the trace. If not provided,
            the current active experiment will be used.
        start_time_ns: The start time of the span in nanoseconds. If not provided,
            the current time will be used.

    Returns:
        A :py:class:`mlflow.entities.Span` that represents the created span.

    Example:
        .. code-block:: python
            :test:

            import mlflow

            root_span = mlflow.start_span_no_context("my_trace")

            # Create a child span
            child_span = mlflow.start_span_no_context(
                "child_span",
                # Manually specify the parent span
                parent_span=root_span,
            )
            # Do something...
            child_span.end()

            root_span.end()

    """
    # If parent span is no-op span, the child should also be no-op too
    if parent_span and parent_span.trace_id == NO_OP_SPAN_TRACE_ID:
        return NoOpSpan()

    try:
        # Create new trace and a root span
        # Once OTel span is created, SpanProcessor.on_start is invoked
        # TraceInfo is created and logged into backend store inside on_start method
        otel_span = provider.start_detached_span(
            name,
            parent=parent_span._span if parent_span else None,
            start_time_ns=start_time_ns,
            experiment_id=experiment_id,
        )

        if parent_span:
            trace_id = parent_span.trace_id
        else:
            trace_id = get_otel_attribute(otel_span, SpanAttributeKey.REQUEST_ID)

        mlflow_span = create_mlflow_span(otel_span, trace_id, span_type)

        # # If the span is a no-op span i.e. tracing is disabled, do nothing
        if isinstance(mlflow_span, NoOpSpan):
            return mlflow_span

        if inputs is not None:
            mlflow_span.set_inputs(inputs)
        mlflow_span.set_attributes(attributes or {})

        trace_manager = InMemoryTraceManager.get_instance()
        if tags := exclude_immutable_tags(tags or {}):
            # Update trace tags for trace in in-memory trace manager
            with trace_manager.get_trace(trace_id) as trace:
                trace.info.tags.update(tags)

        # Register new span in the in-memory trace manager
        trace_manager.register_span(mlflow_span)

        return mlflow_span
    except Exception as e:
        _logger.warning(
            f"Failed to start span {name}: {e}. For full traceback, set logging level to debug.",
            exc_info=_logger.isEnabledFor(logging.DEBUG),
        )
    return NoOpSpan()


@request_id_backward_compatible
def get_trace(trace_id: str) -> Optional[Trace]:
    """
    Get a trace by the given request ID if it exists.

    This function retrieves the trace from the in-memory buffer first, and if it doesn't exist,
    it fetches the trace from the tracking store. If the trace is not found in the tracking store,
    it returns None.

    Args:
        trace_id: The ID of the trace.


    .. code-block:: python
        :test:

        import mlflow


        with mlflow.start_span(name="span") as span:
            span.set_attribute("key", "value")

        trace = mlflow.get_trace(span.trace_id)
        print(trace)


    Returns:
        A :py:class:`mlflow.entities.Trace` objects with the given request ID.
    """
    # Special handling for evaluation request ID.
    trace_id = _EVAL_REQUEST_ID_TO_TRACE_ID.get(trace_id) or trace_id

    try:
        return TracingClient().get_trace(trace_id)
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
    return_type: Optional[Literal["pandas", "list"]] = None,
    model_id: Optional[str] = None,
    sql_warehouse_id: Optional[str] = None,
    include_spans: bool = True,
) -> Union["pandas.DataFrame", list[Trace]]:
    """
    Return traces that match the given list of search expressions within the experiments.

    .. note::

        If expected number of search results is large, consider using the
        `MlflowClient.search_traces` API directly to paginate through the results. This
        function returns all results in memory and may not be suitable for large result sets.

    Args:
        experiment_ids: List of experiment ids to scope the search. If not provided, the search
            will be performed across the current active experiment.
        filter_string: A search filter string.
        max_results: Maximum number of traces desired. If None, all traces matching the search
            expressions will be returned.
        order_by: List of order_by clauses.
        extract_fields: Specify fields to extract from traces using the format
            ``"span_name.[inputs|outputs].field_name"`` or ``"span_name.[inputs|outputs]"``.

            .. note::

                This parameter is only supported when the return type is set to "pandas".

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

        return_type: The type of the return value. The following return types are supported. If
            the pandas library is installed, the default return type is "pandas". Otherwise, the
            default return type is "list".

            - `"pandas"`: Returns a Pandas DataFrame containing information about traces
                where each row represents a single trace and each column represents a field of the
                trace e.g. trace_id, spans, etc.
            - `"list"`: Returns a list of :py:class:`Trace <mlflow.entities.Trace>` objects.

        model_id: If specified, search traces associated with the given model ID.
        sql_warehouse_id: Only used in Databricks. The ID of the SQL warehouse to use for
            searching traces in inference tables.

        include_spans: If ``True``, include spans in the returned traces. Otherwise, only
            the trace metadata is returned, e.g., trace ID, start time, end time, etc,
            without any spans. Default to ``True``.

    Returns:
        Traces that satisfy the search expressions. Either as a list of
        :py:class:`Trace <mlflow.entities.Trace>` objects or as a Pandas DataFrame,
        depending on the value of the `return_type` parameter.

    .. code-block:: python
        :test:
        :caption: Search traces with extract_fields

        import mlflow

        with mlflow.start_span(name="span1") as span:
            span.set_inputs({"a": 1, "b": 2})
            span.set_outputs({"c": 3, "d": 4})

        mlflow.search_traces(
            extract_fields=["span1.inputs", "span1.outputs", "span1.outputs.c"],
            return_type="pandas",
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
        :caption: Search traces by run ID and return as a list of Trace objects

        import mlflow


        @mlflow.trace
        def traced_func(x):
            return x + 1


        with mlflow.start_run() as run:
            traced_func(1)

        mlflow.search_traces(run_id=run.info.run_id, return_type="list")

    """
    from mlflow.tracking.fluent import _get_experiment_id

    # Default to "pandas" only if the pandas library is installed
    if return_type is None:
        try:
            import pandas  # noqa: F401

            return_type = "pandas"
        except ImportError:
            return_type = "list"

    if return_type not in ["pandas", "list"]:
        raise MlflowException.invalid_parameter_value(
            f"Invalid return type: {return_type}. Return type must be either 'pandas' or 'list'."
        )
    elif return_type == "list" and extract_fields:
        raise MlflowException.invalid_parameter_value(
            "The `extract_fields` parameter is only supported when return type is set to 'pandas'."
        )
    elif return_type == "pandas":
        # Check if pandas is installed early to avoid unnecessary computation
        if importlib.util.find_spec("pandas") is None:
            raise MlflowException(
                message=(
                    "The `pandas` library is not installed. Please install `pandas` to use"
                    " the `return_type='pandas'` option, or set `return_type='list'`."
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
        return TracingClient().search_traces(
            experiment_ids=experiment_ids,
            run_id=run_id,
            max_results=number_to_get,
            filter_string=filter_string,
            order_by=order_by,
            page_token=next_page_token,
            model_id=model_id,
            sql_warehouse_id=sql_warehouse_id,
            include_spans=include_spans,
        )

    results = get_results_from_paginated_fn(
        pagination_wrapper_func,
        max_results_per_page=SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        max_results=max_results,
    )

    if return_type == "pandas":
        results = traces_to_df(results)
        if extract_fields:
            results = extract_span_inputs_outputs(
                traces=results,
                fields=extract_fields,
                col_name=SPANS_COLUMN_NAME,
            )

    return results


def get_current_active_span() -> Optional[LiveSpan]:
    """
    Get the current active span in the global context.

    .. attention::

        This only works when the span is created with fluent APIs like `@mlflow.trace` or
        `with mlflow.start_span`. If a span is created with the
        `mlflow.start_span_no_context` APIs, it won't be
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


def get_active_trace_id() -> Optional[str]:
    """
    Get the active trace ID in the current process.

    This function is thread-safe.

    Example:

    .. code-block:: python
        :test:

        import mlflow


        @mlflow.trace
        def f():
            trace_id = mlflow.get_active_trace_id()
            print(trace_id)


        f()

    Returns:
        The ID of the current active trace if exists, otherwise None.
    """
    active_span = get_current_active_span()
    if active_span:
        return active_span.trace_id
    return None


def get_last_active_trace_id(thread_local: bool = False) -> Optional[str]:
    """
    Get the **LAST** active trace in the same process if exists.

    .. warning::

        This function is not thread-safe by default, returns the last active trace in
        the same process. If you want to get the last active trace in the current thread,
        set the `thread_local` parameter to True.

    Args:

        thread_local: If True, returns the last active trace in the current thread. Otherwise,
            returns the last active trace in the same process. Default is False.

    Returns:
        The ID of the last active trace if exists, otherwise None.

    .. code-block:: python
        :test:

        import mlflow


        @mlflow.trace
        def f():
            pass


        f()

        trace_id = mlflow.get_last_active_trace_id()

        # Set a tag on the trace
        mlflow.set_trace_tag(trace_id, "key", "value")

        # Get the full trace object
        trace = mlflow.get_trace(trace_id)
    """
    return (
        _LAST_ACTIVE_TRACE_ID_THREAD_LOCAL.get() if thread_local else _LAST_ACTIVE_TRACE_ID_GLOBAL
    )


def _set_last_active_trace_id(trace_id: str):
    """Internal function to set the last active trace ID."""
    global _LAST_ACTIVE_TRACE_ID_GLOBAL
    _LAST_ACTIVE_TRACE_ID_GLOBAL = trace_id
    _LAST_ACTIVE_TRACE_ID_THREAD_LOCAL.set(trace_id)


def update_current_trace(
    tags: Optional[dict[str, str]] = None,
    metadata: Optional[dict[str, str]] = None,
    client_request_id: Optional[str] = None,
    request_preview: Optional[str] = None,
    response_preview: Optional[str] = None,
    state: Optional[Union[TraceState, str]] = None,
):
    """
    Update the current active trace with the given options.

    Args:
        tags: A dictionary of tags to update the trace with Tags are designed for mutable values,
            that can be updated after the trace is created via MLflow UI or API.
        metadata: A dictionary of metadata to update the trace with. Metadata cannot be updated
            once the trace is logged. It is suitable for recording immutable values like the
            git hash of the application version that produced the trace.
        client_request_id: Client supplied request ID to associate with the trace. This is
            useful for linking the trace back to a specific request in your application or
            external system. If None, the client request ID is not updated.
        request_preview: A preview of the request to be shown in the Trace list view in the UI.
            By default, MLflow will truncate the trace request naively by limiting the length.
            This parameter allows you to specify a custom preview string.
        response_preview: A preview of the response to be shown in the Trace list view in the UI.
            By default, MLflow will truncate the trace response naively by limiting the length.
            This parameter allows you to specify a custom preview string.
        state: The state to set on the trace. Can be a TraceState enum value or string.
            Only "OK" and "ERROR" are allowed. This overrides the overall trace state without
            affecting the status of the current span.

    Example:

        You can use this function either within a function decorated with ``@mlflow.trace`` or
        within the scope of the `with mlflow.start_span` context manager. If there is no active
        trace found, this function will raise an exception.

        Using within a function decorated with `@mlflow.trace`:

        .. code-block:: python

            @mlflow.trace
            def my_func(x):
                mlflow.update_current_trace(tags={"fruit": "apple"}, client_request_id="req-12345")
                return x + 1

        Using within the ``with mlflow.start_span`` context manager:

        .. code-block:: python

            with mlflow.start_span("span"):
                mlflow.update_current_trace(tags={"fruit": "apple"}, client_request_id="req-12345")

        Updating source information of the trace. These keys are reserved ones and MLflow populate
        them from environment information by default. You can override them if needed. Please refer
        to the MLflow Tracing documentation for the full list of reserved metadata keys.

        .. code-block:: python

            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": "session-4f855da00427",
                    "mlflow.trace.user": "user-id-cc156f29bcfb",
                    "mlflow.source.name": "inference.py",
                    "mlflow.source.git.commit": "1234567890",
                    "mlflow.source.git.repoURL": "https://github.com/mlflow/mlflow",
                },
            )

        Updating request preview:

        .. code-block:: python

            import mlflow
            import openai


            @mlflow.trace
            def predict(messages: list[dict]) -> str:
                # Customize the request preview to show the first and last messages
                custom_preview = f"{messages[0]['content'][:10]} ... {messages[-1]['content'][:10]}"
                mlflow.update_current_trace(request_preview=custom_preview)

                # Call the model
                response = openai.chat.completions.create(
                    model="o4-mini",
                    messages=messages,
                )

                return response.choices[0].message.content


            messages = [
                {"role": "user", "content": "Hi, how are you?"},
                {"role": "assistant", "content": "I'm good, thank you!"},
                {"role": "user", "content": "What's your name?"},
                # ... (long message history)
                {"role": "assistant", "content": "Bye!"},
            ]
            predict(messages)

            # The request preview rendered in the UI will be:
            #     "Hi, how are you? ... Bye!"

    """
    active_span = get_current_active_span()

    if not active_span:
        _logger.warning(
            "No active trace found. Please create a span using `mlflow.start_span` or "
            "`@mlflow.trace` before calling `mlflow.update_current_trace`.",
        )
        return

    def _warn_non_string_values(d: dict[str, Any], field_name: str):
        non_string_items = {k: v for k, v in d.items() if not isinstance(v, str)}
        if non_string_items:
            _logger.warning(
                f"Found non-string values in {field_name}. Non-string values in {field_name} will "
                f"automatically be stringified when the trace is logged. Non-string items: "
                f"{non_string_items}"
            )

    _warn_non_string_values(tags or {}, "tags")
    _warn_non_string_values(metadata or {}, "metadata")

    # Update tags and client request ID for the trace stored in-memory rather than directly
    # updating the backend store. The in-memory trace will be exported when it is ended.
    # By doing this, we can avoid unnecessary server requests for each tag update.
    if request_preview is not None and not isinstance(request_preview, str):
        raise MlflowException.invalid_parameter_value(
            "The `request_preview` parameter must be a string."
        )
    if response_preview is not None and not isinstance(response_preview, str):
        raise MlflowException.invalid_parameter_value(
            "The `response_preview` parameter must be a string."
        )

    # Update trace info for the trace stored in-memory rather than directly updating the
    # backend store. The in-memory trace will be exported when it is ended. By doing
    # this, we can avoid unnecessary server requests for each tag update.
    request_id = active_span.request_id
    with InMemoryTraceManager.get_instance().get_trace(request_id) as trace:
        if request_preview:
            trace.info.request_preview = request_preview
        if response_preview:
            trace.info.response_preview = response_preview
        if state is not None:

            def _invalid_state_error(value):
                return MlflowException.invalid_parameter_value(
                    f"State must be either 'OK' or 'ERROR', but got '{value}'."
                )

            if state not in (TraceState.OK, TraceState.ERROR):
                raise _invalid_state_error(state)

            trace.info.state = state

        trace.info.tags.update(tags or {})
        trace.info.trace_metadata.update(metadata or {})
        if client_request_id is not None:
            trace.info.client_request_id = str(client_request_id)


@request_id_backward_compatible
def set_trace_tag(trace_id: str, key: str, value: str):
    """
    Set a tag on the trace with the given trace ID.

    The trace can be an active one or the one that has already ended and recorded in the
    backend. Below is an example of setting a tag on an active trace. You can replace the
    ``trace_id`` parameter to set a tag on an already ended trace.

    .. code-block:: python
        :test:

        import mlflow

        with mlflow.start_span(name="span") as span:
            mlflow.set_trace_tag(span.trace_id, "key", "value")

    Args:
        trace_id: The ID of the trace to set the tag on.
        key: The string key of the tag. Must be at most 250 characters long, otherwise
            it will be truncated when stored.
        value: The string value of the tag. Must be at most 250 characters long, otherwise
            it will be truncated when stored.
    """
    TracingClient().set_trace_tag(trace_id, key, value)


@request_id_backward_compatible
def delete_trace_tag(trace_id: str, key: str) -> None:
    """
    Delete a tag on the trace with the given trace ID.

    The trace can be an active one or the one that has already ended and recorded in the
    backend. Below is an example of deleting a tag on an active trace. You can replace the
    ``trace_id`` parameter to delete a tag on an already ended trace.

    .. code-block:: python
        :test:

        import mlflow

        with mlflow.start_span("my_span") as span:
            mlflow.set_trace_tag(span.trace_id, "key", "value")
            mlflow.delete_trace_tag(span.trace_id, "key")

    Args:
        trace_id: The ID of the trace to delete the tag from.
        key: The string key of the tag. Must be at most 250 characters long, otherwise
            it will be truncated when stored.
    """
    TracingClient().delete_trace_tag(trace_id, key)


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
            with mlflow.start_span(name="predict") as span:
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
            target_trace_id=target_span.trace_id,
            target_parent_span_id=target_span.span_id,
        )
    else:
        # If there is no target span, create a new root span named "Remote Trace <...>"
        # and put the remote trace under it. This design aims to keep the trace export
        # logic simpler and consistent, rather than directly exporting the remote trace.
        remote_root_span = trace.data.spans[0]
        span = start_span_no_context(
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
            target_trace_id=span.trace_id,
            target_parent_span_id=span.span_id,
        )
        span.end(
            status=trace.info.status,
            outputs=remote_root_span.outputs,
            end_time_ns=remote_root_span.end_time_ns,
        )


@experimental
def log_trace(
    name: str = "Task",
    request: Optional[Any] = None,
    response: Optional[Any] = None,
    intermediate_outputs: Optional[dict[str, Any]] = None,
    attributes: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, str]] = None,
    start_time_ms: Optional[int] = None,
    execution_time_ms: Optional[int] = None,
) -> str:
    """
    Create a trace with a single root span.
    This API is useful when you want to log an arbitrary (request, response) pair
    without structured OpenTelemetry spans. The trace is linked to the active experiment.

    Args:
        name: The name of the trace (and the root span). Default to "Task".
        request: Input data for the entire trace. This is also set on the root span of the trace.
        response: Output data for the entire trace. This is also set on the root span of the trace.
        intermediate_outputs: A dictionary of intermediate outputs produced by the model or agent
            while handling the request. Keys are the names of the outputs,
            and values are the outputs themselves. Values must be JSON-serializable.
        attributes: A dictionary of attributes to set on the root span of the trace.
        tags: A dictionary of tags to set on the trace.
        start_time_ms: The start time of the trace in milliseconds since the UNIX epoch.
            When not specified, current time is used for start and end time of the trace.
        execution_time_ms: The execution time of the trace in milliseconds since the UNIX epoch.

    Returns:
        The ID of the logged trace.

    Example:

    .. code-block:: python
        :test:

        import time
        import mlflow

        trace_id = mlflow.log_trace(
            request="Does mlflow support tracing?",
            response="Yes",
            intermediate_outputs={
                "retrieved_documents": ["mlflow documentation"],
                "system_prompt": ["answer the question with yes or no"],
            },
            start_time_ms=int(time.time() * 1000),
            execution_time_ms=5129,
        )
        trace = mlflow.get_trace(trace_id)

        print(trace.data.intermediate_outputs)
    """
    if intermediate_outputs:
        if attributes:
            attributes.update(SpanAttributeKey.INTERMEDIATE_OUTPUTS, intermediate_outputs)
        else:
            attributes = {SpanAttributeKey.INTERMEDIATE_OUTPUTS: intermediate_outputs}

    span = start_span_no_context(
        name=name,
        inputs=request,
        attributes=attributes,
        tags=tags,
        start_time_ns=start_time_ms * 1000000 if start_time_ms else None,
    )
    span.end(
        outputs=response,
        end_time_ns=(start_time_ms + execution_time_ms) * 1000000
        if start_time_ms and execution_time_ms
        else None,
    )

    return span.trace_id


def _merge_trace(
    trace: Trace,
    target_trace_id: str,
    target_parent_span_id: str,
):
    """
    Merge the given trace object under an existing trace in the in-memory trace registry.

    Args:
        trace: The trace object to be merged.
        target_trace_id: The ID of the parent trace.
        target_parent_span_id: The parent span ID, under which the child trace should be merged.
    """
    trace_manager = InMemoryTraceManager.get_instance()

    # The merged trace should have the same trace ID as the parent trace.
    with trace_manager.get_trace(target_trace_id) as parent_trace:
        if not parent_trace:
            _logger.warning(f"Parent trace with ID {target_trace_id} not found. Skipping merge.")
            return

        new_trace_id = parent_trace.span_dict[target_parent_span_id]._trace_id

    for span in trace.data.spans:
        parent_span_id = span.parent_id or target_parent_span_id

        # NB: We clone span one by one in the order it was saved in the original trace. This
        # works upon the assumption that the parent span always comes before its children.
        # This is guaranteed in current implementation, but if it changes in the future,
        # we have to traverse the tree to determine the order.
        if not trace_manager.get_span_from_id(target_trace_id, parent_span_id):
            raise MlflowException.invalid_parameter_value(
                f"Span with ID {parent_span_id} not found. Please make sure the "
                "spans in the trace are ordered correctly i.e. the parent span comes before "
                "its children."
            )

        cloned_span = LiveSpan.from_immutable_span(
            span=span,
            parent_span_id=parent_span_id,
            trace_id=target_trace_id,
            otel_trace_id=new_trace_id,
        )
        trace_manager.register_span(cloned_span)

    # Merge the tags and metadata from the child trace to the parent trace.
    with trace_manager.get_trace(target_trace_id) as parent_trace:
        # Order of merging is important to ensure the parent trace's metadata is
        # not overwritten by the child trace's metadata if they have the same key.
        parent_trace.info.tags = {**trace.info.tags, **parent_trace.info.tags}
        parent_trace.info.trace_metadata = {
            **trace.info.request_metadata,
            **parent_trace.info.trace_metadata,
        }
