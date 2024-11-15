import inspect
import mlflow
from mlflow.entities import SpanType
import mlflow.gemini
from mlflow.utils.autologging_utils.config import AutoLoggingConfig


def _get_span_type(task) -> str:
    import google.generativeai as genai

    span_type_mapping = {
        genai.GenerativeModel: SpanType.LLM,
        genai.ChatSession: SpanType.CHAT_MODEL,
    }
    return span_type_mapping.get(task, SpanType.UNKNOWN)

def construct_full_inputs(func, self, *args, **kwargs):
    signature = inspect.signature(func)
    arguments = signature.bind_partial(self, *args, **kwargs).arguments
    
    if "self" in arguments:
        arguments.pop("self")
    
    if hasattr(self, "model_name"):
        arguments["model_name"] = self.model_name

    return arguments


def patched_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.gemini.FLAVOR_NAME)

    if config.log_traces:
        with mlflow.start_span(
            name=self.__class__.__name__,
            span_type=_get_span_type(self.__class__),
            ) as span:
            inputs = construct_full_inputs(original, self, *args, **kwargs)
            span.set_inputs(inputs)
            result = original(self, *args, **kwargs)
            if hasattr(result, "to_dict"):
                outputs = result.to_dict()
            else:
                outputs = result
            span.set_outputs(outputs)
            return result