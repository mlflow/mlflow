import inspect
import logging

import mlflow
import mlflow.anthropic
from mlflow.anthropic.chat import convert_message_to_mlflow_chat, convert_tool_to_mlflow_chat_tool
from mlflow.entities import SpanType
from mlflow.tracing.utils import set_span_chat_messages, set_span_chat_tools
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


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

            if (tools := inputs.get("tools")) is not None:
                try:
                    tools = [convert_tool_to_mlflow_chat_tool(tool) for tool in tools]
                    set_span_chat_tools(span, tools)
                except Exception as e:
                    _logger.debug(f"Failed to set tools for {span}. Error: {e}")

            messages = [convert_message_to_mlflow_chat(msg) for msg in inputs.get("messages", [])]
            try:
                outputs = original(self, *args, **kwargs)
                span.set_outputs(outputs)
            finally:
                # Set message attribute once at the end to avoid multiple JSON serialization
                try:
                    messages.append(convert_message_to_mlflow_chat(outputs))
                    set_span_chat_messages(span, messages)
                except Exception as e:
                    _logger.debug(f"Failed to set chat messages for {span}. Error: {e}")

            return outputs
