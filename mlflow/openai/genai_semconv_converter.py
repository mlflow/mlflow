"""
OpenAI-format message converter for GenAI Semantic Convention export.

Handles the OpenAI chat completions message format, also used by Groq and Bedrock
(which share the same request/response shape).
"""

import json
from typing import Any

from mlflow.tracing.constant import GenAiSemconvKey
from mlflow.tracing.export.genai_semconv.converter import GenAiSemconvConverter


def _convert_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a single OpenAI message dict to GenAI semconv format with parts array."""
    role = msg.get("role", "user")
    parts = []

    content = msg.get("content")
    if isinstance(content, str):
        parts.append({"type": "text", "content": content})
    elif isinstance(content, list):
        # OpenAI multimodal: [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
        for item in content:
            match item:
                case {"type": "text", "text": str(text)}:
                    parts.append({"type": "text", "content": text})
                case _:
                    parts.append({"type": "text", "content": json.dumps(item)})
    elif content is not None:
        parts.append({"type": "text", "content": str(content)})

    # Tool calls → tool_call parts
    if tool_calls := msg.get("tool_calls"):
        for tc in tool_calls:
            func = tc.get("function", {})
            args_raw = func.get("arguments", "{}")
            # Parse arguments string to object per semconv spec
            try:
                arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except (json.JSONDecodeError, TypeError):
                arguments = args_raw
            parts.append({
                "type": "tool_call",
                "id": tc.get("id"),
                "name": func.get("name"),
                "arguments": arguments,
            })

    # Tool response → tool_call_response part
    if tool_call_id := msg.get("tool_call_id"):
        # Collect all text content into a single result string
        result = parts[0]["content"] if len(parts) == 1 else json.dumps([p["content"] for p in parts]) if parts else ""
        return {
            "role": role,
            "parts": [{"type": "tool_call_response", "id": tool_call_id, "result": result}],
        }

    return {"role": role, "parts": parts}


class OpenAiSemconvConverter(GenAiSemconvConverter):
    def convert_inputs(self, inputs: dict[str, Any]) -> list[dict] | None:
        messages = inputs.get("messages")
        if not isinstance(messages, list):
            return None
        return [_convert_message(m) for m in messages if m.get("role") != "system"]

    def convert_system_instructions(self, inputs: dict[str, Any]) -> list[dict] | None:
        messages = inputs.get("messages")
        if not isinstance(messages, list):
            return None
        parts = []
        for m in messages:
            if m.get("role") != "system":
                continue
            content = m.get("content")
            if isinstance(content, str):
                parts.append({"type": "text", "content": content})
            elif isinstance(content, list):
                for item in content:
                    match item:
                        case {"type": "text", "text": str(text)}:
                            parts.append({"type": "text", "content": text})
                        case _:
                            parts.append({"type": "text", "content": json.dumps(item)})
            elif content is not None:
                parts.append({"type": "text", "content": str(content)})
        return parts or None

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

    def extract_request_params(self, inputs: dict[str, Any]) -> dict[str, Any]:
        params = super().extract_request_params(inputs)
        # Flatten OpenAI's nested {"type": "function", "function": {...}} tool format
        # to the spec's flat {"type": "function", "name": ..., "description": ..., ...}
        if GenAiSemconvKey.TOOL_DEFINITIONS in params:
            tools = inputs.get("tools", [])
            flattened = []
            for tool in tools:
                flat = {"type": tool.get("type", "function")}
                if func := tool.get("function"):
                    flat.update(func)
                flattened.append(flat)
            params[GenAiSemconvKey.TOOL_DEFINITIONS] = json.dumps(flattened)
        return params

    def extract_response_attrs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        attrs = super().extract_response_attrs(outputs)
        # Collect finish_reasons from all choices
        choices = outputs.get("choices")
        if isinstance(choices, list):
            reasons = [c.get("finish_reason") for c in choices if c.get("finish_reason")]
            if reasons:
                attrs[GenAiSemconvKey.RESPONSE_FINISH_REASONS] = reasons
        return attrs
