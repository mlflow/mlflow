import logging
from dataclasses import asdict
from typing import Any

from mlflow.tracing.constant import TokenUsageKey

_logger = logging.getLogger(__name__)


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
        total_tokens = getattr(usage, "total_tokens")
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
