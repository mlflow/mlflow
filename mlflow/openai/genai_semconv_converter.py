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
    def convert_inputs(self, inputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        messages = inputs.get("messages")
        if not isinstance(messages, list):
            return None
        return [_convert_message(m) for m in messages if m.get("role") != "system"]

    def convert_system_instructions(self, inputs: dict[str, Any]) -> list[dict[str, Any]] | None:
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

    def convert_outputs(self, outputs: dict[str, Any]) -> list[dict[str, Any]] | None:
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
            if reasons := [c.get("finish_reason") for c in choices if c.get("finish_reason")]:
                attrs[GenAiSemconvKey.RESPONSE_FINISH_REASONS] = reasons
        return attrs


class OpenAIResponsesConverter(GenAiSemconvConverter):
    def convert_inputs(self, inputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        input_data = inputs.get("input")
        if input_data is None:
            return None
        if isinstance(input_data, str):
            return [{"role": "user", "parts": [{"type": "text", "content": input_data}]}]
        if isinstance(input_data, list):
            return [self._convert_input_item(item) for item in input_data]
        return None

    def convert_system_instructions(self, inputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        if instructions := inputs.get("instructions"):
            if isinstance(instructions, str):
                return [{"type": "text", "content": instructions}]
        return None

    def convert_outputs(self, outputs: dict[str, Any]) -> list[dict[str, Any]] | None:
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
        item_type = item.get("type")
        if item_type == "function_call":
            return {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": item["call_id"],
                        "name": item["name"],
                        "arguments": _parse_tool_arguments(item["arguments"]),
                    }
                ],
            }
        elif item_type == "function_call_output":
            return {
                "role": "tool",
                "parts": [
                    {"type": "tool_call_response", "id": item["call_id"], "result": item["output"]}
                ],
            }
        else:
            return _convert_message(item)

    @staticmethod
    def _convert_output_item(item: dict[str, Any]) -> dict[str, Any]:
        item_type = item.get("type")
        if item_type == "message":
            parts = []
            for ci in item.get("content", []):
                if ci.get("type") == "output_text":
                    parts.append({"type": "text", "content": ci["text"]})
                else:
                    parts.append({"type": "text", "content": json.dumps(ci)})
            return {"role": item.get("role", "assistant"), "parts": parts}
        elif item_type == "function_call":
            return {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": item["call_id"],
                        "name": item["name"],
                        "arguments": _parse_tool_arguments(item["arguments"]),
                    }
                ],
            }
        else:
            return {
                "role": "assistant",
                "parts": [{"type": "text", "content": json.dumps(item)}],
            }


def _convert_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a single OpenAI chat message dict to GenAI semconv format with parts array."""
    role = msg.get("role", "user")
    parts = _convert_content(msg.get("content"))

    if tool_calls := msg.get("tool_calls"):
        parts.extend(_convert_tool_call(tc) for tc in tool_calls)

    if tool_call_id := msg.get("tool_call_id"):
        return _convert_tool_response(role, tool_call_id, parts)

    return {"role": role, "parts": parts}


def _convert_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    if isinstance(content, list):
        parts = []
        for item in content:
            item_type = item.get("type") if isinstance(item, dict) else None
            if item_type in ("text", "input_text"):
                parts.append({"type": "text", "content": item["text"]})
            elif item_type == "image_url":
                # Chat completion format
                parts.append(_convert_image_url(item["image_url"]["url"]))
            elif item_type == "input_image":
                # Responses API format
                parts.append(_convert_image_url(item["image_url"]))
            elif item_type == "input_audio":
                audio = item["input_audio"]
                parts.append({
                    "type": "blob",
                    "modality": "audio",
                    "mime_type": f"audio/{audio['format']}",
                    "content": audio["data"],
                })
            else:
                parts.append({"type": "text", "content": json.dumps(item)})
        return parts
    if content is not None:
        return [{"type": "text", "content": str(content)}]
    return []


def _convert_image_url(url: str) -> dict[str, Any]:
    if url.startswith("data:"):
        # Parse "data:<mime_type>;base64,<content>"
        header, _, data = url.partition(",")
        mime_type = header.removeprefix("data:").removesuffix(";base64")
        return {
            "type": "blob",
            "modality": "image",
            "mime_type": mime_type,
            "content": data,
        }
    return {"type": "uri", "modality": "image", "uri": url}


def _convert_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    func = tc.get("function", {})
    return {
        "type": "tool_call",
        "id": tc.get("id"),
        "name": func.get("name"),
        "arguments": _parse_tool_arguments(func.get("arguments", "{}")),
    }


def _convert_tool_response(
    role: str, tool_call_id: str, parts: list[dict[str, Any]]
) -> dict[str, Any]:
    result = parts[0].get("content") if parts else None
    return {
        "role": role,
        "parts": [{"type": "tool_call_response", "id": tool_call_id, "result": result}],
    }


def _parse_tool_arguments(args: Any) -> Any:
    try:
        return json.loads(args) if isinstance(args, str) else args
    except (json.JSONDecodeError, TypeError):
        return args


def _flatten_tools(tools: list[dict[str, Any]]) -> str:
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
