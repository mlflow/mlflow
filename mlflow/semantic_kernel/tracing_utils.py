import logging
from typing import Any, Callable, Optional

from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.trace import get_current_span
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.kernel_content import KernelContent
from semantic_kernel.contents.streaming_content_mixin import StreamingContentMixin
from semantic_kernel.utils.telemetry.model_diagnostics import (
    gen_ai_attributes as model_gen_ai_attributes,
)
from semantic_kernel.utils.telemetry.model_diagnostics.decorators import (
    CHAT_COMPLETION_OPERATION,
    CHAT_STREAMING_COMPLETION_OPERATION,
    TEXT_COMPLETION_OPERATION,
    TEXT_STREAMING_COMPLETION_OPERATION,
)

from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.utils import construct_full_inputs

# NB: Use global variable instead of the instance variable of the processor, because sometimes
# multiple span processor instances can be created and we need to share the same map.
_OTEL_SPAN_ID_TO_MLFLOW_SPAN_AND_TOKEN = {}


_logger = logging.getLogger(__name__)


def semantic_kernel_diagnostics_wrapper(original, *args, **kwargs) -> None:
    """
    Wrapper for Semantic Kernel's model diagnostics decorators.

    This wrapper is used to record the inputs and outputs to the span, because
    Semantic Kernel's Otel span do not record the inputs and outputs.
    """
    full_kwargs = construct_full_inputs(original, *args, **kwargs)
    current_span = full_kwargs.get("current_span") or get_current_span()
    otel_span_id = current_span.get_span_context().span_id
    mlflow_span = _get_live_span_from_otel_span_id(otel_span_id)

    if not mlflow_span:
        _logger.debug("Span is not found or recording. Skipping error handling.")
        return original(*args, **kwargs)

    if prompt := full_kwargs.get("prompt"):
        # Wrapping _set_completion_input
        # https://github.com/microsoft/semantic-kernel/blob/d5ee6aa1c176a4b860aba72edaa961570874661b/python/semantic_kernel/utils/telemetry/model_diagnostics/decorators.py#L369
        mlflow_span.set_inputs(_parse_content(prompt))
        mlflow_span.set_span_type(SpanType.CHAT_MODEL)

    if completions := full_kwargs.get("completions"):
        # Wrapping _set_completion_response
        # https://github.com/microsoft/semantic-kernel/blob/d5ee6aa1c176a4b860aba72edaa961570874661b/python/semantic_kernel/utils/telemetry/model_diagnostics/decorators.py#L400
        full_responses = []
        for completion in completions:
            full_response: dict[str, Any] = completion.to_dict()

            if isinstance(completion, ChatMessageContent):
                full_response["finish_reason"] = completion.finish_reason.value
            if isinstance(completion, StreamingContentMixin):
                full_response["index"] = completion.choice_index

            full_responses.append(full_response)

        mlflow_span.set_outputs({"messages": full_responses})

    if error := full_kwargs.get("error"):
        # Wrapping _set_completion_error
        # https://github.com/microsoft/semantic-kernel/blob/d5ee6aa1c176a4b860aba72edaa961570874661b/python/semantic_kernel/utils/telemetry/model_diagnostics/decorators.py#L452
        mlflow_span.record_exception(error)

    return original(*args, **kwargs)


def create_trace_wrapper(
    span_type: Optional[SpanType] = None,
) -> Callable[[Any], Any]:
    """Create a trace wrapper with specific parser and serializer."""

    async def _trace_wrapper(original, self, *args, **kwargs):
        otel_span_id = get_current_span().get_span_context().span_id
        mlflow_span = _get_live_span_from_otel_span_id(otel_span_id)

        if not mlflow_span:
            return await original(self, *args, **kwargs)

        if span_type:
            mlflow_span.set_span_type(span_type)

        inputs = construct_full_inputs(original, self, *args, **kwargs)
        mlflow_span.set_inputs(_parse_content(inputs))

        try:
            result = await original(self, *args, **kwargs)
        except Exception as e:
            mlflow_span.record_exception(e)
            raise

        mlflow_span.set_outputs(_parse_content(result))
        return result

    return _trace_wrapper


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
    elif isinstance(value, KernelContent):
        value = value.to_dict()
    elif isinstance(value, list):
        value = [_parse_content(item) for item in value]
    return value


def _get_live_span_from_otel_span_id(otel_span_id: str) -> Optional[LiveSpan]:
    if span_and_token := _OTEL_SPAN_ID_TO_MLFLOW_SPAN_AND_TOKEN.get(otel_span_id):
        return span_and_token[0]
    else:
        _logger.debug(
            f"Live span not found for OTel span ID: {otel_span_id}. "
            "Cannot map OTel span ID to MLflow span ID, so we will skip registering "
            "additional attributes. "
        )
        return None


def _get_span_type(span: OTelSpan) -> str:
    """Determine the span type based on the operation."""
    span_type = None

    if hasattr(span, "attributes") and (
        operation := span.attributes.get(model_gen_ai_attributes.OPERATION)
    ):
        span_map = {
            CHAT_COMPLETION_OPERATION: SpanType.CHAT_MODEL,
            CHAT_STREAMING_COMPLETION_OPERATION: SpanType.CHAT_MODEL,
            TEXT_COMPLETION_OPERATION: SpanType.LLM,
            TEXT_STREAMING_COMPLETION_OPERATION: SpanType.LLM,
            # added from https://github.com/microsoft/semantic-kernel/blob/79d3dde556e4cdc482d83c9f5f0a459c5cc79a48/python/semantic_kernel/utils/telemetry/model_diagnostics/function_tracer.py#L24
            "execute_tool": SpanType.TOOL,
        }
        span_type = span_map.get(operation)

    return span_type or SpanType.UNKNOWN


def _set_token_usage(mlflow_span: LiveSpan, sk_attributes: dict[str, Any]) -> None:
    """Set token usage attributes on the MLflow span."""
    input_tokens = sk_attributes.get(model_gen_ai_attributes.INPUT_TOKENS)
    output_tokens = sk_attributes.get(model_gen_ai_attributes.OUTPUT_TOKENS)

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
