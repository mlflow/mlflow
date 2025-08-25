import logging
from typing import Any

from opentelemetry.trace import get_current_span
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.kernel_content import KernelContent
from semantic_kernel.contents.streaming_content_mixin import StreamingContentMixin
from semantic_kernel.functions import FunctionResult
from semantic_kernel.utils.telemetry.model_diagnostics import (
    gen_ai_attributes as model_gen_ai_attributes,
)
from semantic_kernel.utils.telemetry.model_diagnostics.decorators import (
    CHAT_COMPLETION_OPERATION,
    CHAT_STREAMING_COMPLETION_OPERATION,
    TEXT_COMPLETION_OPERATION,
    TEXT_STREAMING_COMPLETION_OPERATION,
)
from semantic_kernel.utils.telemetry.model_diagnostics.function_tracer import (
    OPERATION_NAME as FUNCTION_OPERATION_NAME,
)

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.utils import (
    construct_full_inputs,
    get_mlflow_span_for_otel_span,
)

_OPERATION_TO_SPAN_TYPE = {
    CHAT_COMPLETION_OPERATION: SpanType.CHAT_MODEL,
    CHAT_STREAMING_COMPLETION_OPERATION: SpanType.CHAT_MODEL,
    TEXT_COMPLETION_OPERATION: SpanType.LLM,
    TEXT_STREAMING_COMPLETION_OPERATION: SpanType.LLM,
    FUNCTION_OPERATION_NAME: SpanType.TOOL,
    # https://github.com/microsoft/semantic-kernel/blob/d5ee6aa1c176a4b860aba72edaa961570874661b/python/semantic_kernel/utils/telemetry/agent_diagnostics/decorators.py#L22
    "invoke_agent": SpanType.AGENT,
}

_logger = logging.getLogger(__name__)


def semantic_kernel_diagnostics_wrapper(original, *args, **kwargs) -> None:
    """
    Wrapper for Semantic Kernel's model diagnostics decorators.

    This wrapper is used to record the inputs and outputs to the span, because
    Semantic Kernel's Otel span do not record the inputs and outputs.
    """
    full_kwargs = construct_full_inputs(original, *args, **kwargs)
    current_span = full_kwargs.get("current_span") or get_current_span()
    mlflow_span = get_mlflow_span_for_otel_span(current_span)

    if not mlflow_span:
        _logger.debug("Span is not found or recording. Skipping error handling.")
        return original(*args, **kwargs)

    if prompt := full_kwargs.get("prompt"):
        # Wrapping _set_completion_input
        # https://github.com/microsoft/semantic-kernel/blob/d5ee6aa1c176a4b860aba72edaa961570874661b/python/semantic_kernel/utils/telemetry/model_diagnostics/decorators.py#L369
        mlflow_span.set_inputs(_parse_content(prompt))

    if completions := full_kwargs.get("completions"):
        # Wrapping _set_completion_response
        # https://github.com/microsoft/semantic-kernel/blob/d5ee6aa1c176a4b860aba72edaa961570874661b/
        mlflow_span.set_outputs({"messages": [_parse_content(c) for c in completions]})

    if error := full_kwargs.get("error"):
        # Wrapping _set_completion_error
        # https://github.com/microsoft/semantic-kernel/blob/d5ee6aa1c176a4b860aba72edaa961570874661b/python/semantic_kernel/utils/telemetry/model_diagnostics/decorators.py#L452
        mlflow_span.record_exception(error)

    return original(*args, **kwargs)


async def patched_kernel_entry_point(original, self, *args, **kwargs):
    with mlflow.start_span(
        name=f"{self.__class__.__name__}.{original.__name__}",
        span_type=SpanType.AGENT,
    ) as mlflow_span:
        inputs = construct_full_inputs(original, self, *args, **kwargs)
        mlflow_span.set_inputs(_parse_content(inputs))

        result = await original(self, *args, **kwargs)

        mlflow_span.set_outputs(_parse_content(result))

    return result


def _parse_content(value: Any) -> Any:
    """
    Parse the message content objects in Semantic Kernel into a more readable format.

    Those objects are Pydantic models, but includes many noisy fields that are not
    useful for debugging and hard to read. The base KernelContent class has a to_dict()
    method that converts them into more readable format (role, content), so we use that.
    """
    if isinstance(value, dict) and (chat_history := value.get("chat_history")):
        value = _parse_content(chat_history)
    elif isinstance(value, ChatHistory):
        # Record chat history as a list of messages for better readability
        value = {"messages": [_parse_content(m) for m in value.messages]}
    elif isinstance(value, (KernelContent, StreamingContentMixin)):
        value = value.to_dict()
    elif isinstance(value, FunctionResult):
        # Extract "value" field from the FunctionResult object
        value = _parse_content(value.value)
    elif isinstance(value, list):
        value = [_parse_content(item) for item in value]
    return value


def set_span_type(mlflow_span: LiveSpan) -> str:
    """Determine the span type based on the operation."""
    span_type = SpanType.UNKNOWN
    if operation := mlflow_span.get_attribute(model_gen_ai_attributes.OPERATION):
        span_type = _OPERATION_TO_SPAN_TYPE.get(operation, SpanType.UNKNOWN)

    mlflow_span.set_span_type(span_type)


def set_token_usage(mlflow_span: LiveSpan) -> None:
    """Set token usage attributes on the MLflow span."""
    input_tokens = mlflow_span.get_attribute(model_gen_ai_attributes.INPUT_TOKENS)
    output_tokens = mlflow_span.get_attribute(model_gen_ai_attributes.OUTPUT_TOKENS)

    usage_dict = {}
    if input_tokens is not None:
        usage_dict[TokenUsageKey.INPUT_TOKENS] = input_tokens
    if output_tokens is not None:
        usage_dict[TokenUsageKey.OUTPUT_TOKENS] = output_tokens

    if input_tokens is not None or output_tokens is not None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
        usage_dict[TokenUsageKey.TOTAL_TOKENS] = total_tokens

    if usage_dict:
        mlflow_span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
