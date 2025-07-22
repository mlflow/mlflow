import inspect
import logging

import mlflow
import mlflow.mistral
from mlflow.entities import SpanType
from mlflow.mistral.chat import convert_tool_to_mlflow_chat_tool
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.utils import set_span_chat_tools
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def _construct_full_inputs(func, *args, **kwargs):
    signature = inspect.signature(func)
    # this does not create copy. So values should not be mutated directly
    arguments = signature.bind_partial(*args, **kwargs).arguments

    if "self" in arguments:
        arguments.pop("self")

    return arguments


def patched_class_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.mistral.FLAVOR_NAME)

    if config.log_traces:
        with mlflow.start_span(
            name=f"{self.__class__.__name__}.{original.__name__}",
            span_type=SpanType.CHAT_MODEL,
        ) as span:
            inputs = _construct_full_inputs(original, self, *args, **kwargs)
            span.set_inputs(inputs)
            span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "mistral")

            if (tools := inputs.get("tools")) is not None:
                try:
                    tools = [convert_tool_to_mlflow_chat_tool(tool) for tool in tools if tool]
                    set_span_chat_tools(span, tools)
                except Exception as e:
                    _logger.debug(f"Failed to set tools for {span}. Error: {e}")

            outputs = original(self, *args, **kwargs)

            try:
                span.set_outputs(outputs)
                if usage := getattr(outputs, "usage", None):
                    span.set_attribute(
                        SpanAttributeKey.CHAT_USAGE,
                        {
                            TokenUsageKey.INPUT_TOKENS: usage.prompt_tokens,
                            TokenUsageKey.OUTPUT_TOKENS: usage.completion_tokens,
                            TokenUsageKey.TOTAL_TOKENS: usage.total_tokens,
                        },
                    )
            except Exception as e:
                _logger.debug(f"Failed to process outputs for {span}. Error: {e}")

            return outputs
