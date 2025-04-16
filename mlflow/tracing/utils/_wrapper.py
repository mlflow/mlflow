import functools
import inspect
import json
import logging
from typing import Any, Callable, Optional

from mlflow.entities import NoOpSpan, SpanType
from mlflow.entities.span import LiveSpan
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.constant import (
    STREAM_CHUNK_EVENT_NAME_FORMAT,
    STREAM_CHUNK_EVENT_VALUE_KEY,
    SpanAttributeKey,
)
from mlflow.tracing.fluent import get_current_active_span, start_span, start_span_no_context
from mlflow.tracing.provider import safe_set_span_in_context
from mlflow.tracing.utils import (
    TraceJSONEncoder,
    capture_function_input_args,
)

_logger = logging.getLogger(__name__)


def wrap_function(
    fn: Callable,
    name: Optional[str] = None,
    span_type: str = SpanType.UNKNOWN,
    attributes: Optional[dict[str, Any]] = None,
    model_id: Optional[str] = None,
) -> Callable:
    class _WrappingContext:
        # define the wrapping logic as a coroutine to avoid code duplication
        # between sync and async cases
        @staticmethod
        def _wrapping_logic(fn, args, kwargs):
            span_name = name or fn.__name__

            with start_span(
                name=span_name, span_type=span_type, attributes=attributes, model_id=model_id
            ) as span:
                span.set_attribute(SpanAttributeKey.FUNCTION_NAME, fn.__name__)
                span.set_inputs(capture_function_input_args(fn, args, kwargs))
                result = yield  # sync/async function output to be sent here
                span.set_outputs(result)
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

    return functools.wraps(fn)(wrapper)


def wrap_generator(
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

    def _start_stream_span(fn, args, kwargs):
        try:
            return start_span_no_context(
                name=name or fn.__name__,
                parent_span=get_current_active_span(),
                span_type=span_type,
                attributes=attributes,
                inputs=capture_function_input_args(fn, args, kwargs),
            )
        except Exception as e:
            _logger.debug(f"Failed to start stream span: {e}")
            return NoOpSpan()

    def _end_stream_span(
        span: LiveSpan,
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
            span = _start_stream_span(fn, args, kwargs)
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
            _end_stream_span(span, outputs, output_reducer)
    else:

        async def wrapper(*args, **kwargs):
            span = _start_stream_span(fn, args, kwargs)
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
            _end_stream_span(span, outputs, output_reducer)

    return functools.wraps(fn)(wrapper)
