import inspect
import logging

import mlflow
import mlflow.gemini
from mlflow.entities import SpanType
from mlflow.gemini.chat import (
    convert_gemini_func_to_mlflow_chat_tool,
)
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.provider import detach_span_from_context, set_span_in_context
from mlflow.tracing.utils import construct_full_inputs, set_span_chat_tools
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

try:
    # This is for supporting the previous Google GenAI SDK
    # https://github.com/google-gemini/generative-ai-python
    from google import generativeai

    has_generativeai = True
except ImportError:
    has_generativeai = False

try:
    from google import genai

    has_genai = True
except ImportError:
    has_genai = False

_logger = logging.getLogger(__name__)


def patched_class_call(original, self, *args, **kwargs):
    """
    This method is used for patching class methods of gemini SDKs.
    This patch creates a span and set input and output of the original method to the span.
    """
    with TracingSession(original, self, args, kwargs) as manager:
        output = original(self, *args, **kwargs)
        manager.output = output
        return output


async def async_patched_class_call(original, self, *args, **kwargs):
    """
    This method is used for patching async class methods of gemini SDKs.
    This patch creates a span and set input and output of the original method to the span.
    """
    async with TracingSession(original, self, args, kwargs) as manager:
        output = await original(self, *args, **kwargs)
        manager.output = output
        return output


class TracingSession:
    """Context manager for handling MLflow spans in both sync and async contexts."""

    def __init__(self, original, instance, args, kwargs):
        self.original = original
        self.instance = instance
        self.inputs = construct_full_inputs(original, instance, *args, **kwargs)

        # These attributes are set outside the constructor.
        self.span = None
        self.token = None
        self.output = None

    def __enter__(self):
        return self._enter_impl()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._exit_impl(exc_type, exc_val, exc_tb)

    async def __aenter__(self):
        return self._enter_impl()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._exit_impl(exc_type, exc_val, exc_tb)

    def _enter_impl(self):
        config = AutoLoggingConfig.init(flavor_name=mlflow.gemini.FLAVOR_NAME)
        if not config.log_traces:
            return self

        self.span = mlflow.start_span_no_context(
            name=f"{self.instance.__class__.__name__}.{self.original.__name__}",
            span_type=_get_span_type(self.original.__name__),
            inputs=self.inputs,
            attributes={SpanAttributeKey.MESSAGE_FORMAT: "gemini"},
        )
        if has_generativeai and isinstance(self.instance, generativeai.GenerativeModel):
            _log_generativeai_tool_definition(self.instance, self.span)

        if _is_genai_model_or_chat(self.instance):
            _log_genai_tool_definition(self.instance, self.inputs, self.span)

        # Attach the span to the current context. This is necessary because single Gemini
        # SDK call might create multiple child spans.
        self.token = set_span_in_context(self.span)
        return self

    def _exit_impl(self, exc_type, exc_val, exc_tb) -> None:
        if not self.span:
            return

        # Detach span from the context at first. This must not be interrupted by any exception,
        # otherwise the span context will leak and pollute other traces created next.
        detach_span_from_context(self.token)

        if exc_val:
            self.span.record_exception(exc_val)

        # need to convert the response of generate_content for better visualization
        outputs = self.output.to_dict() if hasattr(self.output, "to_dict") else self.output
        self.span.end(outputs=outputs)


def _is_genai_model_or_chat(instance) -> bool:
    return has_genai and isinstance(
        instance,
        (
            genai.models.Models,
            genai.chats.Chat,
            genai.models.AsyncModels,
            genai.chats.AsyncChat,
        ),
    )


def patched_module_call(original, *args, **kwargs):
    """
    This method is used for patching standalone functions of the google.generativeai module.
    This patch creates a span and set input and output of the original function to the span.
    """
    config = AutoLoggingConfig.init(flavor_name=mlflow.gemini.FLAVOR_NAME)
    if not config.log_traces:
        return original(*args, **kwargs)

    with mlflow.start_span(
        name=f"{original.__name__}",
        span_type=_get_span_type(original.__name__),
    ) as span:
        inputs = _construct_full_inputs(original, *args, **kwargs)
        span.set_inputs(inputs)
        span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "gemini")
        result = original(*args, **kwargs)
        # need to convert the response of generate_content for better visualization
        outputs = result.to_dict() if hasattr(result, "to_dict") else result
        span.set_outputs(outputs)

    return result


def _get_keys(dic, keys):
    for key in keys:
        if key in dic:
            return dic[key]

    return None


def _log_generativeai_tool_definition(model, span):
    """
    This method extract tool definition from generativeai tool type.
    """
    # when tools are not passed
    if not getattr(model, "_tools", None):
        return

    try:
        set_span_chat_tools(
            span,
            [
                convert_gemini_func_to_mlflow_chat_tool(func)
                for func in model._tools.to_proto()[0].function_declarations
            ],
        )
    except Exception as e:
        _logger.warning(f"Failed to set tool definitions for {span}. Error: {e}")


def _log_genai_tool_definition(model, inputs, span):
    """
    This method extract tool definition from genai tool type.
    """
    config = inputs.get("config")
    tools = getattr(config, "tools", None)
    if not tools:
        return
    # Here, we use an internal function of gemini library to convert callable to Tool schema to
    # avoid having the same logic on mlflow side and there is no public attribute for Tool schema.
    # https://github.com/googleapis/python-genai/blob/01b15e32d3823a58d25534bb6eea93f30bf82219/google/genai/_transformers.py#L662
    tools = genai._transformers.t_tools(model._api_client, tools)

    try:
        set_span_chat_tools(
            span,
            [
                convert_gemini_func_to_mlflow_chat_tool(function_declaration)
                for tool in tools
                for function_declaration in tool.function_declarations
            ],
        )
    except Exception as e:
        _logger.warning(f"Failed to set tool definitions for {span}. Error: {e}")


def _get_span_type(task_name: str) -> str:
    span_type_mapping = {
        "generate_content": SpanType.LLM,
        "_generate_content": SpanType.LLM,
        "send_message": SpanType.CHAT_MODEL,
        "count_tokens": SpanType.LLM,
        "embed_content": SpanType.EMBEDDING,
    }
    return span_type_mapping.get(task_name, SpanType.UNKNOWN)


def _construct_full_inputs(func, *args, **kwargs):
    signature = inspect.signature(func)
    # this method does not create copy. So values should not be mutated directly
    arguments = signature.bind_partial(*args, **kwargs).arguments

    if "self" in arguments:
        arguments.pop("self")

    return arguments
