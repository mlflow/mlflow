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
from mlflow.entities.span import LiveSpan, SpanEvent, SpanStatusCode
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

# NB: Use global variable instead of the instance variable of the processor, because sometimes
# multiple span processor instances can be created and we need to share the same map.
_OTEL_SPAN_ID_TO_MLFLOW_SPAN_AND_TOKEN = {}


_logger = logging.getLogger(__name__)


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
        result = {}

        # Check if function object is passed as first positional arg after self
        if len(args) > 1 and args[1] is not None:
            function = args[1]
            if hasattr(function, "name") and hasattr(function, "plugin_name"):
                result["function_name"] = function.name
                result["plugin_name"] = function.plugin_name

        # Check for function_name and plugin_name in kwargs
        for k in ["function_name", "plugin_name"]:
            if k in kwargs and kwargs[k]:
                result[k] = kwargs[k]

        # Check for arguments (can be positional or keyword)
        if len(args) > 2 and args[2] is not None:
            # arguments passed as positional
            arguments = args[2]
            if hasattr(arguments, "__dict__"):
                result["arguments"] = dict(arguments)
        elif "arguments" in kwargs and kwargs["arguments"] is not None:
            arguments = kwargs["arguments"]
            if hasattr(arguments, "__dict__"):
                result["arguments"] = dict(arguments)

        return result
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


def _serialize_chat_output(result: Any) -> Any:
    """Serialize chat completion outputs."""
    try:
        if result is None:
            return None
        if isinstance(result, ChatMessageContent):
            result = [result]
        if isinstance(result, list) and result and isinstance(result[0], ChatMessageContent):
            full_responses = []
            for completion in result:
                full_response: dict[str, Any] = completion.to_dict()

                if isinstance(completion, ChatMessageContent):
                    full_response["finish_reason"] = completion.finish_reason.value
                full_responses.append(full_response)
            return {"messages": full_responses}
        return json.dumps(None)
    except Exception as e:
        _logger.warning(f"Failed to serialize chat result: {e}")
        return None


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
        return result
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


def _set_span_inputs(
    mlflow_span: LiveSpan,
    parser: Optional[Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]]],
    original: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Set input attributes on the span."""
    parsed_inputs = parser(args, kwargs) if parser else {}

    if not parsed_inputs:
        parsed_inputs = {"function": original.__qualname__}
        if args[1:]:  # Skip self
            parsed_inputs["args"] = [getattr(a, "__class__.__name__", str(a)) for a in args[1:]]
        if kwargs:
            parsed_inputs["kwargs"] = {k: str(v) for k, v in kwargs.items()}

    mlflow_span.set_inputs(parsed_inputs)


def _set_span_outputs(
    mlflow_span: LiveSpan,
    serializer: Optional[Callable[[Any], str]],
    result: Any,
    error: Optional[Exception] = None,
) -> None:
    """Set output attributes on the span."""
    output_str = serializer(result)
    mlflow_span.set_outputs(output_str)

    if serializer == _serialize_chat_output and output_str is not None:
        try:
            output_dict = json.loads(output_str)
            if "messages" in output_dict:
                mlflow_span.set_attribute(SpanAttributeKey.CHAT_MESSAGES, output_dict["messages"])
        except Exception:
            pass

    if error:
        mlflow_span.add_event(SpanEvent.from_exception(error))
        mlflow_span.set_status(SpanStatusCode.ERROR)


def _create_trace_wrapper(
    span_type: Optional[SpanType] = None,
    parser: Optional[Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]]] = None,
    serializer: Optional[Callable[[Any], str]] = None,
) -> Callable[[Any], Any]:
    """Create a trace wrapper with specific parser and serializer."""

    async def _trace_wrapper(original, *args, **kwargs):
        span = get_current_span()
        mlflow_span = _get_live_span_from_otel_span_id(span.get_span_context().span_id)

        if not mlflow_span:
            return await original(*args, **kwargs)

        if span_type:
            mlflow_span.set_span_type(span_type)

        _set_span_inputs(mlflow_span, parser, original, args, kwargs)

        try:
            result = await original(*args, **kwargs)
        except Exception as e:
            _set_span_outputs(mlflow_span, serializer, None, error=e)
            raise

        _set_span_outputs(mlflow_span, serializer, result)
        return result

    return _trace_wrapper
