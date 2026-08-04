import json
from typing import Any

from mlflow.tracing.constant import GenAiSemconvKey
from mlflow.tracing.export.genai_semconv.converter import GenAiSemconvConverter


class AnthropicConverter(GenAiSemconvConverter):
    def convert_inputs(self, inputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        messages = inputs.get("messages")
        if not isinstance(messages, list):
            return None
        return [_convert_message(m) for m in messages]

    def convert_system_instructions(self, inputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        system = inputs.get("system")
        if isinstance(system, str):
            return [{"type": "text", "content": system}]
        if isinstance(system, list):
            return [_convert_block(b) for b in system]
        return None

    def convert_outputs(self, outputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        content = outputs.get("content")
        if not isinstance(content, list):
            return None
        parts = [_convert_block(b) for b in content]
        return [{"role": outputs.get("role", "assistant"), "parts": parts}]

    def extract_request_params(self, inputs: dict[str, Any]) -> dict[str, Any]:
        params = super().extract_request_params(inputs)
        if (stop_sequences := inputs.get("stop_sequences")) is not None:
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            params[GenAiSemconvKey.REQUEST_STOP_SEQUENCES] = stop_sequences
        if GenAiSemconvKey.TOOL_DEFINITIONS in params:
            params[GenAiSemconvKey.TOOL_DEFINITIONS] = json.dumps(inputs.get("tools", []))
        return params


def _convert_message(msg: dict[str, Any]) -> dict[str, Any]:
    role = msg.get("role", "user")
    content = msg.get("content")

    if isinstance(content, str):
        return {"role": role, "parts": [{"type": "text", "content": content}]}

    if isinstance(content, list):
        parts = []
        has_tool_result = False
        for block in content:
            converted = _convert_block(block)
            parts.append(converted)
            if converted.get("type") == "tool_call_response":
                has_tool_result = True
        # Anthropic uses "user" role for tool result. Override it to "tool"
        if has_tool_result and len(parts) == 1:
            return {"role": "tool", "parts": parts}
        return {"role": role, "parts": parts}

    return {"role": role, "parts": []}


def _convert_block(block: dict[str, Any]) -> dict[str, Any]:
    block_type = block.get("type")
    match block_type:
        case "text":
            return {"type": "text", "content": block.get("text", "")}
        case "image" | "document":
            source = block.get("source", {})
            source_type = source.get("type")
            if source_type == "base64":
                return {
                    "type": "blob",
                    "modality": block_type,
                    "mime_type": source.get("media_type", ""),
                    "content": source.get("data", ""),
                }
            if source_type == "url":
                return {
                    "type": "uri",
                    "modality": block_type,
                    "uri": source.get("url", ""),
                }
            return {"type": "text", "content": json.dumps(block)}
        case "tool_use":
            return {
                "type": "tool_call",
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "arguments": block.get("input"),
            }
        case "tool_result":
            return {
                "type": "tool_call_response",
                "id": block.get("tool_use_id", ""),
                "result": block.get("content", ""),
            }
        case _:
            # Fallback to text with dumped content block
            return {"type": "text", "content": json.dumps(block)}
