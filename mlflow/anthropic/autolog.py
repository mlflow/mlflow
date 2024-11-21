import inspect

import mlflow
import mlflow.anthropic
from mlflow.entities import SpanType
from mlflow.utils.autologging_utils.config import AutoLoggingConfig


def _get_span_type(task_name: str) -> str:
    span_type_mapping = {
        "create": SpanType.CHAT_MODEL,
    }
    return span_type_mapping.get(task_name, SpanType.UNKNOWN)


def construct_full_inputs(func, *args, **kwargs):
    signature = inspect.signature(func)
    # this does not create copy. So values should not be mutated directly
    arguments = signature.bind_partial(*args, **kwargs).arguments

    if "self" in arguments:
        self = arguments.pop("self")

        if hasattr(self, "model_name"):
            arguments["model_name"] = self.model_name

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
            result = original(self, *args, **kwargs)
            # need to convert the response of generate_content for better visualization
            outputs = result.to_dict() if hasattr(result, "to_dict") else result
            span.set_outputs(outputs)

            return result
