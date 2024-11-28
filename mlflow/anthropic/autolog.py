import inspect

import mlflow
import mlflow.anthropic
from mlflow.entities import SpanType
from mlflow.utils.autologging_utils.config import AutoLoggingConfig


def _get_span_type(task_name: str) -> str:
    # Anthropic has a few APIs in beta, e.g., count_tokens.
    # Once they are stable, we can add them to the mapping.
    span_type_mapping = {
        "create": SpanType.CHAT_MODEL,
    }
    return span_type_mapping.get(task_name, SpanType.UNKNOWN)


def construct_full_inputs(func, *args, **kwargs):
    signature = inspect.signature(func)
    # this does not create copy. So values should not be mutated directly
    arguments = signature.bind_partial(*args, **kwargs).arguments

    if "self" in arguments:
        arguments.pop("self")

    return arguments


def patched_class_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.anthropic.FLAVOR_NAME)

    if config.log_traces:
        with mlflow.start_span(
            name=f"{self.__class__.__name__}.{original.__name__}",
            span_type=_get_span_type(original.__name__),
        ) as span:
            inputs = construct_full_inputs(original, self, *args, **kwargs)
            span.set_inputs(inputs)
            outputs = original(self, *args, **kwargs)
            span.set_outputs(outputs)
            return outputs
