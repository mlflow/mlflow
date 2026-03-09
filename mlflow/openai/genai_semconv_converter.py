"""
OpenAI-format message converters for GenAI Semantic Convention export.

Two converters handle the two OpenAI API shapes:
- OpenAIChatCompletionConverter: Chat Completions API (also used by Groq, Bedrock)
- OpenAIResponsesConverter: Responses API
"""

import json
from typing import Any

from mlflow.tracing.constant import GenAiSemconvKey
from mlflow.tracing.export.genai_semconv.converter import GenAiSemconvConverter


class OpenAIChatCompletionConverter(GenAiSemconvConverter):
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
        if GenAiSemconvKey.TOOL_DEFINITIONS in params:
            params[GenAiSemconvKey.TOOL_DEFINITIONS] = _flatten_tools(inputs.get("tools", []))
        return params

    def extract_response_attrs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        attrs = super().extract_response_attrs(outputs)
        choices = outputs.get("choices")
        if isinstance(choices, list):
            reasons = [c.get("finish_reason") for c in choices if c.get("finish_reason")]
            if reasons:
                attrs[GenAiSemconvKey.RESPONSE_FINISH_REASONS] = reasons
        return attrs


class OpenAIResponsesConverter(GenAiSemconvConverter):
    def convert_inputs(self, inputs: dict[str, Any]) -> list[dict] | None:
        input_data = inputs.get("input")
        if input_data is None:
            return None
        if isinstance(input_data, str):
            return [{"role": "user", "parts": [{"type": "text", "content": input_data}]}]
        if isinstance(input_data, list):
            return [self._convert_input_item(item) for item in input_data]
        return None

    def convert_system_instructions(self, inputs: dict[str, Any]) -> list[dict] | None:
        if instructions := inputs.get("instructions"):
            if isinstance(instructions, str):
                return [{"type": "text", "content": instructions}]
        return None

    def convert_outputs(self, outputs: dict[str, Any]) -> list[dict] | None:
        output_items = outputs.get("output")
        if not isinstance(output_items, list):
            return None
        status = outputs.get("status")
        result = []
        for item in output_items:
            converted = self._convert_output_item(item)
            if status:
                converted["finish_reason"] = status
            result.append(converted)
        return result

    def extract_request_params(self, inputs: dict[str, Any]) -> dict[str, Any]:
        params = super().extract_request_params(inputs)
        if GenAiSemconvKey.TOOL_DEFINITIONS in params:
            params[GenAiSemconvKey.TOOL_DEFINITIONS] = _flatten_tools(inputs.get("tools", []))
        return params

    def extract_response_attrs(self, outputs: dict[str, Any]) -> dict[str, Any]:
        attrs = super().extract_response_attrs(outputs)
        if status := outputs.get("status"):
            attrs[GenAiSemconvKey.RESPONSE_FINISH_REASONS] = [status]
        return attrs

    @staticmethod
    def _convert_input_item(item: dict[str, Any]) -> dict[str, Any]:
        match item:
            case {"type": "function_call", "call_id": call_id, "name": name, "arguments": args}:
                return {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": call_id,
                            "name": name,
                            "arguments": _parse_tool_arguments(args),
                        }
                    ],
                }
            case {"type": "function_call_output", "call_id": call_id, "output": output}:
                return {
                    "role": "tool",
                    "parts": [{"type": "tool_call_response", "id": call_id, "result": output}],
                }
            case _:
                return _convert_message(item)

    @staticmethod
    def _convert_output_item(item: dict[str, Any]) -> dict[str, Any]:
        match item:
            case {"type": "message", "role": role, "content": list(content_items)}:
                parts = []
                for ci in content_items:
                    match ci:
                        case {"type": "output_text", "text": str(text)}:
                            parts.append({"type": "text", "content": text})
                        case _:
                            parts.append({"type": "text", "content": json.dumps(ci)})
                return {"role": role, "parts": parts}
            case {"type": "function_call", "call_id": call_id, "name": name, "arguments": args}:
                return {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": call_id,
                            "name": name,
                            "arguments": _parse_tool_arguments(args),
                        }
                    ],
                }
            case _:
                return {
                    "role": "assistant",
                    "parts": [{"type": "text", "content": json.dumps(item)}],
                }


def _convert_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a single OpenAI chat message dict to GenAI semconv format with parts array."""
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
            try:
                arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except (json.JSONDecodeError, TypeError):
                arguments = args_raw
            parts.append(
                {
                    "type": "tool_call",
                    "id": tc.get("id"),
                    "name": func.get("name"),
                    "arguments": arguments,
                }
            )

    # Tool response → tool_call_response part
    if tool_call_id := msg.get("tool_call_id"):
        result = (
            parts[0]["content"]
            if len(parts) == 1
            else json.dumps([p["content"] for p in parts])
            if parts
            else ""
        )
        return {
            "role": role,
            "parts": [{"type": "tool_call_response", "id": tool_call_id, "result": result}],
        }

    return {"role": role, "parts": parts}


def _parse_tool_arguments(args: Any) -> Any:
    try:
        return json.loads(args) if isinstance(args, str) else args
    except (json.JSONDecodeError, TypeError):
        return args


def _flatten_tools(tools: list[dict]) -> str:
    """Flatten OpenAI's nested tool format to the semconv flat format."""
    flattened = []
    for tool in tools:
        flat = {"type": tool.get("type", "function")}
        if func := tool.get("function"):
            flat.update(func)
        else:
            # Responses API: tools are already flat
            flat.update({k: v for k, v in tool.items() if k != "type"})
        flattened.append(flat)
    return json.dumps(flattened)
