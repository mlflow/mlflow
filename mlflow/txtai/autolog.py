"""
Autolog patch methods for ``mlflow.txtai``
"""

import inspect
import json
import warnings

import mlflow
from mlflow import MlflowClient, start_span
from mlflow.entities import SpanType
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.tracing.constant import STREAM_CHUNK_EVENT_NAME_FORMAT, STREAM_CHUNK_EVENT_VALUE_KEY
from mlflow.tracing.fluent import get_current_active_span
from mlflow.tracing.provider import safe_set_span_in_context
from mlflow.tracing.utils import (
    TraceJSONEncoder,
    end_client_span_or_trace,
    start_client_span_or_trace,
)
from mlflow.utils.autologging_utils.config import AutoLoggingConfig


def patch_generator(target, method, function):
    """
    Patches a generator method with trace logging.

    Args:
        target: target class
        method: target method
        function: target.method function
    """

    def fn(self, *args, **kwargs):
        return _patch_class_generator(function, self, *args, **kwargs)

    # Add original function as attribute
    setattr(fn, "__wrapped__", function)
    setattr(target, method, fn)


def patch_class_call(original, self, *args, **kwargs):
    """
    Patches a method with trace logging.

    Args:
        original: original method
        self: object instance
        args: arguments
        kwargs: keyword arguments

    Returns:
        self.original result
    """

    config = AutoLoggingConfig.init(flavor_name=mlflow.txtai.FLAVOR_NAME)

    if config.log_traces:
        with start_span(
            name=_get_span_name(original, self), span_type=_get_span_type(self)
        ) as span:
            # Set attributes
            for attribute, value in vars(self).items():
                span.set_attribute(attribute, value)

            # Set inputs
            inputs = _construct_full_inputs(original, self, *args, **kwargs)
            span.set_inputs(inputs)

            # Run method
            results = original(self, *args, **kwargs)

            # Set outputs
            outputs = results.__dict__ if hasattr(results, "__dict__") else results
            span.set_outputs(outputs)

            return results

    return None


def _patch_class_generator(original, self, *args, **kwargs):
    """
    Patches a generator method with trace logging.

    Args:
        original: generator method
        self: object instance
        args: arguments
        kwargs: keyword arguments

    Returns:
        self.original result
    """

    config = AutoLoggingConfig.init(flavor_name=mlflow.txtai.FLAVOR_NAME)

    if config.log_traces:
        span = _start_stream_span(original, self, *args, **kwargs)

        # Start generator
        generator = original(self, *args, **kwargs)

        index, outputs = 0, []
        while True:
            try:
                # Set the span to active only when the generator is running
                with safe_set_span_in_context(span):
                    value = next(generator)
            except StopIteration:
                break
            except Exception as e:
                _end_stream_span(span, error=e)
                raise e
            else:
                outputs.append(value)
                _stream_span_chunk(span, value, index)
                yield value
                index += 1

        _end_stream_span(span, outputs)


def _start_stream_span(original, self, *args, **kwargs):
    """
    Starts a span.

    Args:
        original: original method
        self: object instance
        args: arguments
        kwargs: keyword arguments

    Returns:
        span
    """

    # Start span
    return start_client_span_or_trace(
        client=MlflowClient(),
        name=_get_span_name(original, self),
        parent_span=get_current_active_span(),
        span_type=_get_span_type(self),
        attributes=vars(self),
        inputs=_construct_full_inputs(original, self, *args, **kwargs),
    )


def _stream_span_chunk(span, chunk, index):
    """
    Adds a span event with a chunk of data.

    Args:
        span: span instance
        chunk: data chunk
        index: data chunk index
    """

    event = SpanEvent(
        name=STREAM_CHUNK_EVENT_NAME_FORMAT.format(index=index),
        attributes={STREAM_CHUNK_EVENT_VALUE_KEY: json.dumps(chunk, cls=TraceJSONEncoder)},
    )
    span.add_event(event)


def _end_stream_span(span, outputs=None, error=None):
    """
    Ends a span.

    Args:
        span: span to end
        outputs: optional outputs to log
        error: error to log, if any
    """

    client = MlflowClient()
    if error:
        span.add_event(SpanEvent.from_exception(error))
        end_client_span_or_trace(client, span, status=SpanStatusCode.ERROR)
        return

    end_client_span_or_trace(client, span, outputs=outputs)


def _get_span_name(original, self):
    """
    Creates a span name for inputs.

    Args:
        original: original method
        self: object instance

    Returns:
        span name
    """

    name = self.__class__.__name__
    if not original.__name__.startswith("__"):
        name += f".{original.__name__}"

    return name


def _get_span_type(instance):
    """
    Maps txtai objects to MLflow span types.

    Args:
        instance: txtai object instance

    Returns:
        SpanType
    """

    # pylint: disable=C0415
    import txtai

    if isinstance(instance, txtai.Agent):
        return SpanType.AGENT

    if isinstance(instance, txtai.Embeddings):
        return SpanType.RETRIEVER

    if isinstance(instance, txtai.LLM):
        return SpanType.LLM

    if isinstance(instance, (txtai.pipeline.Pipeline, txtai.workflow.Task)):
        return SpanType.PARSER

    if isinstance(instance, (txtai.RAG, txtai.Workflow)):
        return SpanType.CHAIN

    if isinstance(instance, txtai.vectors.Vectors):
        return SpanType.EMBEDDING

    # Default to RETRIEVER
    return SpanType.RETRIEVER


def _construct_full_inputs(func, *args, **kwargs):
    """
    Constructs function inputs as a dictionary.

    Args:
        func: function
        args: arguments
        kwargs: keyword arguments

    Returns:
        dictionary of function inputs
    """

    # Get function arguments
    signature = inspect.signature(func)
    arguments = signature.bind_partial(*args, **kwargs).arguments

    if "self" in arguments:
        arguments.pop("self")

    # Avoid non serializable objects and circular references
    return {
        k: v.__dict__ if hasattr(v, "__dict__") else v
        for k, v in arguments.items()
        if v is not None and _is_serializable(v)
    }


def _is_serializable(value):
    """
    Checks if a value is serializable.

    Args:
        value: value to check

    Returns:
        True if the value is serializable, False otherwise
    """

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            json.dumps(value, cls=TraceJSONEncoder, ensure_ascii=False)
        return True
    except (TypeError, ValueError):
        return False
