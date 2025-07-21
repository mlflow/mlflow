import json
import logging
from typing import Any, Callable, Optional

from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.trace import get_current_span
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.functions.function_result import FunctionResult
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

_logger = logging.getLogger(__name__)


def _parse_chat_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Parse chat completion method inputs."""
    try:
        chat_history = args[1] if len(args) > 1 else kwargs.get("chat_history")
        if chat_history and hasattr(chat_history, "messages"):
            return {
                "messages": [
                    {
                        "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                        "content": str(msg.content),
                    }
                    for msg in chat_history.messages
                    if hasattr(msg, "role") and hasattr(msg, "content")
                ]
            }
    except Exception:
        pass
    return {}


def _parse_text_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Parse text completion method inputs."""
    try:
        # Text methods always have prompt as first positional arg after self
        prompt = args[1] if len(args) > 1 else kwargs.get("prompt", "")
        return {"prompt": str(prompt)}
    except Exception:
        pass
    return {}


def _parse_embedding_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Parse embedding generation method inputs."""
    try:
        # Embedding methods always have texts as first positional arg after self
        texts = args[1] if len(args) > 1 else kwargs.get("texts", [])
        if isinstance(texts, list):
            return {"texts": [str(t) for t in texts]}
    except Exception:
        pass
    return {}


def _parse_kernel_invoke_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Parse Kernel.invoke method inputs."""
    try:
        return {k: kwargs[k] for k in ["function_name", "plugin_name"] if k in kwargs and kwargs[k]}
    except Exception:
        pass
    return {}


def _parse_kernel_invoke_prompt_inputs(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Parse Kernel.invoke_prompt method inputs."""
    try:
        # invoke_prompt has prompt as first positional arg after self
        prompt = args[1] if len(args) > 1 else kwargs.get("prompt", "")
        return {"prompt": str(prompt)}
    except Exception:
        pass
    return {}


def _serialize_chat_output(result: Any) -> str:
    """Serialize chat completion outputs."""
    try:
        if result is None:
            return json.dumps(None)
        if isinstance(result, ChatMessageContent):
            result = [result]
        if isinstance(result, list) and result and isinstance(result[0], ChatMessageContent):
            return json.dumps(
                {"messages": [{"role": msg.role.value, "content": msg.content} for msg in result]}
            )
        return json.dumps(None)
    except Exception as e:
        _logger.warning(f"Failed to serialize chat result: {e}")
        return json.dumps(None)


def _serialize_text_output(result: Any) -> str:
    """Serialize text completion outputs."""
    try:
        if result is None:
            return json.dumps(None)
        if hasattr(result, "to_dict"):
            return json.dumps(result.to_dict())
        if isinstance(result, list) and result and hasattr(result[0], "to_dict"):
            return json.dumps([item.to_dict() for item in result])
        return json.dumps(str(result))
    except Exception as e:
        _logger.warning(f"Failed to serialize text result: {e}")
        return json.dumps(str(result))


def _serialize_kernel_output(result: Any) -> str:
    """Serialize kernel function outputs."""
    try:
        if result is None:
            return json.dumps(None)
        if isinstance(result, FunctionResult):
            return _serialize_kernel_output(result.value)
        return json.dumps(result)
    except Exception as e:
        _logger.warning(f"Failed to serialize kernel result: {e}")
        return json.dumps(str(result))


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
        }
        span_type = span_map.get(operation)

    return span_type or SpanType.UNKNOWN


def _set_token_usage(mlflow_span: LiveSpan, sk_attributes: dict[str, Any]) -> None:
    """Set token usage attributes on the MLflow span."""
    if value := sk_attributes.get(model_gen_ai_attributes.INPUT_TOKENS):
        mlflow_span.set_attribute(TokenUsageKey.INPUT_TOKENS, value)
    if value := sk_attributes.get(model_gen_ai_attributes.OUTPUT_TOKENS):
        mlflow_span.set_attribute(TokenUsageKey.OUTPUT_TOKENS, value)

    if (input_tokens := sk_attributes.get(model_gen_ai_attributes.INPUT_TOKENS)) and (
        output_tokens := sk_attributes.get(model_gen_ai_attributes.OUTPUT_TOKENS)
    ):
        mlflow_span.set_attribute(TokenUsageKey.TOTAL_TOKENS, input_tokens + output_tokens)


def _set_span_inputs(
    span: Any,
    parser: Optional[Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]]],
    original: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Set input attributes on the span."""
    if not span or not span.is_recording():
        return

    span.set_attribute(SpanAttributeKey.FUNCTION_NAME, original.__qualname__)
    parsed_inputs = parser(args, kwargs) if parser else {}

    if not parsed_inputs:  # Fallback parsing
        parsed_inputs = {"function": original.__qualname__}
        if args[1:]:  # Skip self
            parsed_inputs["args"] = [getattr(a, "__class__.__name__", str(a)) for a in args[1:]]
        if kwargs:
            parsed_inputs["kwargs"] = {k: str(v) for k, v in kwargs.items()}

    span.set_attribute(SpanAttributeKey.INPUTS, json.dumps(parsed_inputs))


def _set_span_outputs(
    span: Any,
    serializer: Optional[Callable[[Any], str]],
    result: Any,
    error: Optional[Exception] = None,
) -> None:
    """Set output attributes on the span."""
    if not span or not span.is_recording():
        return

    if error:
        span.set_attribute(SpanAttributeKey.OUTPUTS, json.dumps({"error": str(error)}))
        return

    output_str = serializer(result) if serializer else json.dumps(str(result) if result else None)
    span.set_attribute(SpanAttributeKey.OUTPUTS, output_str)

    # Set CHAT_MESSAGES for chat outputs (as array format for backward compatibility)
    if serializer == _serialize_chat_output and output_str != "null":
        try:
            output_dict = json.loads(output_str)
            if "messages" in output_dict:
                span.set_attribute(
                    SpanAttributeKey.CHAT_MESSAGES, json.dumps(output_dict["messages"])
                )
        except Exception:
            pass


def _create_trace_wrapper(
    parser: Optional[Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]]] = None,
    serializer: Optional[Callable[[Any], str]] = None,
) -> Callable[[Any], Any]:
    """Create a trace wrapper with specific parser and serializer."""

    async def _trace_wrapper(original, *args, **kwargs):
        span = get_current_span()
        _set_span_inputs(span, parser, original, args, kwargs)

        try:
            result = await original(*args, **kwargs)
            _set_span_outputs(span, serializer, result)
            return result
        except Exception as e:
            _set_span_outputs(span, serializer, None, error=e)
            raise

    return _trace_wrapper


def _streaming_not_supported_wrapper(original, *args, **kwargs):
    """Wrapper for streaming methods that logs a debug message."""
    _logger.debug(
        f"Streaming method '{original.__qualname__}' called. "
        "Note: Streaming responses are not currently captured in MLflow traces."
    )
    return original(*args, **kwargs)
