"""
Bedrock Converse API message converter for GenAI Semantic Convention export.

Translates Bedrock's Converse API format (content blocks with text, toolUse,
toolResult, image) into the GenAI semconv parts array format.
"""

import base64
import json
from typing import Any

from mlflow.tracing.constant import GenAiSemconvKey
from mlflow.tracing.export.genai_semconv.converter import GenAiSemconvConverter

_INFERENCE_CONFIG_KEY_MAPPING = {
    "temperature": GenAiSemconvKey.REQUEST_TEMPERATURE,
    "maxTokens": GenAiSemconvKey.REQUEST_MAX_TOKENS,
    "topP": GenAiSemconvKey.REQUEST_TOP_P,
    "stopSequences": GenAiSemconvKey.REQUEST_STOP_SEQUENCES,
}


class BedrockConverseConverter(GenAiSemconvConverter):
    def convert_inputs(self, inputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        messages = inputs.get("messages")
        if not isinstance(messages, list):
            return None
        return [_convert_message(m) for m in messages]

    def convert_system_instructions(self, inputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        system = inputs.get("system")
        if not isinstance(system, list):
            return None
        parts = [
            {"type": "text", "content": text} for block in system if (text := block.get("text"))
        ]
        return parts or None

    def convert_outputs(self, outputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        match outputs:
            case {"output": {"message": dict() as message}}:
                return [_convert_message(message)]
            case _:
                return None

    def extract_request_params(self, inputs: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if isinstance(config := inputs.get("inferenceConfig"), dict):
            for bedrock_key, semconv_key in _INFERENCE_CONFIG_KEY_MAPPING.items():
                if (value := config.get(bedrock_key)) is not None:
                    params[semconv_key] = value

        if isinstance(tool_config := inputs.get("toolConfig"), dict):
            if tools := tool_config.get("tools"):
                params[GenAiSemconvKey.TOOL_DEFINITIONS] = json.dumps(_flatten_tools(tools))
        return params


def _convert_message(msg: dict[str, Any]) -> dict[str, Any]:
    role = msg.get("role", "user")
    content = msg.get("content")
    if not isinstance(content, list):
        return {"role": role, "parts": []}

    parts = []
    has_tool_result = False

    for block in content:
        if "text" in block:
            parts.append({"type": "text", "content": block["text"]})
        elif tool_use := block.get("toolUse"):
            arguments = tool_use.get("input", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, TypeError):
                    pass
            parts.append({
                "type": "tool_call",
                "id": tool_use.get("toolUseId"),
                "name": tool_use.get("name"),
                "arguments": arguments,
            })
        elif tool_result := block.get("toolResult"):
            has_tool_result = True
            result_content = tool_result.get("content", [])
            parts.append({
                "type": "tool_call_response",
                "id": tool_result.get("toolUseId"),
                "result": _extract_tool_result(result_content),
            })
        elif image := block.get("image"):
            parts.append(_convert_image(image))

    if has_tool_result:
        role = "tool"

    return {"role": role, "parts": parts}


def _extract_tool_result(content: list[dict[str, Any]]) -> str | None:
    if not content:
        return None
    results = []
    for item in content:
        if (json_val := item.get("json")) is not None:
            results.append(json.dumps(json_val))
        elif text := item.get("text"):
            results.append(text)
    match results:
        case [single]:
            return single
        case [_, *_]:
            return json.dumps(results)
        case _:
            return None


def _convert_image(image: dict[str, Any]) -> dict[str, Any]:
    fmt = image.get("format", "png")
    source = image.get("source", {})
    image_bytes = source.get("bytes")
    if image_bytes is None:
        return {"type": "text", "content": json.dumps(image)}
    if isinstance(image_bytes, (bytes, bytearray)):
        data = base64.b64encode(image_bytes).decode("utf-8")
    else:
        # Bedrock should always return bytes, but casting everything else to string for safety
        data = str(image_bytes)
    return {
        "type": "blob",
        "modality": "image",
        "mime_type": f"image/{fmt}",
        "content": data,
    }


def _flatten_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for tool in tools:
        if tool_spec := tool.get("toolSpec"):
            flat: dict[str, Any] = {"type": "function", "name": tool_spec["name"]}
            if desc := tool_spec.get("description"):
                flat["description"] = desc
            if input_schema := tool_spec.get("inputSchema"):
                flat["parameters"] = input_schema.get("json")
            flattened.append(flat)
    return flattened
