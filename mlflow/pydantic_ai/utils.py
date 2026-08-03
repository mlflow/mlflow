import logging
from dataclasses import asdict, is_dataclass
from typing import Any

from mlflow.tracing.constant import TokenUsageKey

_logger = logging.getLogger(__name__)
_SAFE_PRIMITIVE_TYPES = (str, int, float, bool)


def is_safe_for_serialization(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, _SAFE_PRIMITIVE_TYPES):
        return True
    if isinstance(value, dict):
        return all(is_safe_for_serialization(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(is_safe_for_serialization(item) for item in value)
    return (is_dataclass(value) and not isinstance(value, type)) or isinstance(value, type)


def extract_safe_attributes(instance: Any) -> dict[str, Any]:
    attributes = {}
    for key in dir(instance):
        if key.startswith("_"):
            continue
        try:
            value = getattr(instance, key, None)
        except Exception:
            continue
        if callable(value) and not isinstance(value, type):
            continue
        if isinstance(value, type):
            attributes[key] = value.__name__
        elif is_safe_for_serialization(value):
            attributes[key] = value
    return attributes


def model_request_inputs(request_context) -> dict[str, Any]:
    if request_context is None:
        return {}
    inputs = {
        "messages": getattr(request_context, "messages", None),
        "model_settings": getattr(request_context, "model_settings", None),
        "model_request_parameters": getattr(request_context, "model_request_parameters", None),
    }
    return {key: value for key, value in inputs.items() if value is not None}


def serialize_output(result: Any) -> Any:
    if result is None:
        return None

    if hasattr(result, "new_messages") and callable(result.new_messages):
        try:
            new_messages = result.new_messages()
            serialized_messages = [asdict(msg) for msg in new_messages]

            try:
                serialized_result = asdict(result)
            except Exception:
                # We can't use asdict for StreamedRunResult because its async generator
                serialized_result = dict(result.__dict__) if hasattr(result, "__dict__") else {}

            serialized_result["_new_messages_serialized"] = serialized_messages
            return serialized_result
        except Exception as e:
            _logger.debug("Failed to serialize new_messages: %s", e)

    return result.__dict__ if hasattr(result, "__dict__") else result


def parse_usage(result: Any) -> dict[str, int] | None:
    try:
        if isinstance(result, tuple) and len(result) == 2:
            usage = result[1]
        else:
            usage_attr = getattr(result, "usage", None)
            if usage_attr is None:
                return None

            # Handle both property (RunResult) and method (StreamedRunResult)
            # StreamedRunResult has .usage() as a method
            usage = usage_attr() if callable(usage_attr) else usage_attr

        if usage is None:
            return None

        # input_tokens/output_tokens are the current field names; request_tokens/
        # response_tokens are deprecated aliases kept for backward compatibility.
        input_tokens = getattr(usage, "input_tokens", None)
        if input_tokens is None:
            input_tokens = getattr(usage, "request_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "response_tokens", 0)
        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        return {
            TokenUsageKey.INPUT_TOKENS: input_tokens,
            TokenUsageKey.OUTPUT_TOKENS: output_tokens,
            TokenUsageKey.TOTAL_TOKENS: total_tokens,
        }
    except Exception as e:
        _logger.debug("Failed to parse token usage from output: %s", e)
    return None
