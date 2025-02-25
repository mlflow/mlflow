import inspect
import logging

import mlflow
import mlflow.gemini
from mlflow.entities import SpanType
from mlflow.gemini.chat import (
    convert_gemini_func_to_mlflow_chat_tool,
    parse_gemini_content_to_mlflow_chat_messages,
)
from mlflow.tracing.utils import set_span_chat_messages, set_span_chat_tools
from mlflow.types.chat import ChatMessage
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
    config = AutoLoggingConfig.init(flavor_name=mlflow.gemini.FLAVOR_NAME)

    if config.log_traces:
        with mlflow.start_span(
            name=f"{self.__class__.__name__}.{original.__name__}",
            span_type=_get_span_type(original.__name__),
        ) as span:
            inputs = _construct_full_inputs(original, self, *args, **kwargs)
            span.set_inputs(inputs)
            if has_generativeai and isinstance(self, generativeai.GenerativeModel):
                _log_generativeai_tool_definition(self, span)
            if has_genai and isinstance(self, (genai.models.Models, genai.chats.Chat)):
                _log_genai_tool_definition(self, inputs, span)

            result = original(self, *args, **kwargs)

            if (
                has_generativeai and isinstance(result, generativeai.types.GenerateContentResponse)
            ) or (has_genai and isinstance(result, genai.types.GenerateContentResponse)):
                try:
                    content = _get_keys(inputs, ["contents", "content", "message"])
                    messages = parse_gemini_content_to_mlflow_chat_messages(content)
                    messages += _parse_outputs(result)
                    if messages:
                        set_span_chat_messages(span=span, messages=messages)
                except Exception as e:
                    _logger.warning(
                        f"An exception occurred on logging chat attributes for {span}. Error: {e}"
                    )

            # need to convert the response of generate_content for better visualization
            outputs = result.to_dict() if hasattr(result, "to_dict") else result
            span.set_outputs(outputs)

            return result


def patched_module_call(original, *args, **kwargs):
    """
    This method is used for patching standalone functions of the google.generativeai module.
    This patch creates a span and set input and output of the original function to the span.
    """
    config = AutoLoggingConfig.init(flavor_name=mlflow.gemini.FLAVOR_NAME)

    if config.log_traces:
        with mlflow.start_span(
            name=f"{original.__name__}",
            span_type=_get_span_type(original.__name__),
        ) as span:
            inputs = _construct_full_inputs(original, *args, **kwargs)
            span.set_inputs(inputs)
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


def _parse_outputs(outputs) -> list[ChatMessage]:
    """
    This method extract chat messages from genai.types.generation_types.GenerateContentResponse
    """
    # content always exist on output
    # https://github.com/googleapis/googleapis/blob/9e966149c59f47f6305d66c98e2a9e7d9c26a2eb/google/ai/generativelanguage/v1beta/generative_service.proto#L490
    return sum(
        [
            parse_gemini_content_to_mlflow_chat_messages(candidate.content)
            for candidate in outputs.candidates
        ],
        [],
    )


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
