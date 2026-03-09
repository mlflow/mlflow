"""
OpenAI-format message converter for GenAI Semantic Convention export.

Handles the OpenAI chat completions message format, also used by Groq and Bedrock
(which share the same request/response shape).
"""

from typing import Any

from mlflow.tracing.constant import GenAiSemconvKey
from mlflow.tracing.export.genai_semconv.converter import GenAiSemconvConverter


def _convert_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a single OpenAI message dict to GenAI semconv format."""
    converted: dict[str, Any] = {}

    if role := msg.get("role"):
        converted["role"] = role

    # Content: string, list of content parts, or None (for tool-call messages)
    if "content" in msg:
        converted["content"] = msg["content"]

    # Tool calls (assistant messages)
    if tool_calls := msg.get("tool_calls"):
        converted["tool_calls"] = [
            {
                "id": tc.get("id"),
                "type": tc.get("type", "function"),
                "function": tc.get("function"),
            }
            for tc in tool_calls
        ]

    # Tool call ID (tool response messages)
    if tool_call_id := msg.get("tool_call_id"):
        converted["tool_call_id"] = tool_call_id

    return converted


class OpenAiSemconvConverter(GenAiSemconvConverter):
    def convert_inputs(self, inputs: dict[str, Any]) -> list[dict] | None:
        messages = inputs.get("messages")
        if not isinstance(messages, list):
            return None
        return [_convert_message(m) for m in messages]

    def convert_outputs(self, outputs: dict[str, Any]) -> list[dict] | None:
        choices = outputs.get("choices")
        if not isinstance(choices, list):
            return None
        result = []
        for choice in choices:
            msg = choice.get("message") or choice.get("delta", {})
            converted = _convert_message(msg)
            if finish_reason := choice.get("finish_reason"):
                converted["finish_reason"] = finish_reason
            result.append(converted)
        return result

    def extract_response_attrs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        attrs = super().extract_response_attrs(outputs)
        # Collect finish_reasons from all choices
        choices = outputs.get("choices")
        if isinstance(choices, list):
            reasons = [c.get("finish_reason") for c in choices if c.get("finish_reason")]
            if reasons:
                attrs[GenAiSemconvKey.RESPONSE_FINISH_REASONS] = reasons
        return attrs
