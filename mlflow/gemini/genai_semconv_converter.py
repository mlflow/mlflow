import json
from typing import Any

from mlflow.tracing.constant import GenAiSemconvKey
from mlflow.tracing.export.genai_semconv.converter import GenAiSemconvConverter


class GeminiConverter(GenAiSemconvConverter):
    def convert_inputs(self, inputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        contents = inputs.get("contents")
        if contents is None:
            return None

        if isinstance(contents, str):
            return [{"role": "user", "parts": [{"type": "text", "content": contents}]}]

        if isinstance(contents, list):
            # Check if this is a list of Content dicts (have "role" key)
            # or a flat list of Part dicts/strings (no "role" key)
            if contents and isinstance(contents[0], dict) and "role" in contents[0]:
                return [_convert_content_dict(c) for c in contents]
            # Flat list of parts → single user message
            parts = [_convert_part(p) for p in contents]
            return [{"role": "user", "parts": parts}]

        return None

    def convert_system_instructions(self, inputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        config = inputs.get("config")
        if not isinstance(config, dict):
            return None
        system_instruction = config.get("system_instruction")

        if system_instruction is None:
            return None
        elif isinstance(system_instruction, str):
            return [{"type": "text", "content": system_instruction}]
        elif isinstance(system_instruction, dict):
            parts = system_instruction.get("parts", [])
            return [_convert_part(p) for p in parts]

    def convert_outputs(self, outputs: dict[str, Any]) -> list[dict[str, Any]] | None:
        candidates = outputs.get("candidates")
        if not isinstance(candidates, list):
            return None
        result = []
        for candidate in candidates:
            content = candidate.get("content", {})
            parts_list = content.get("parts", [])
            role = content.get("role", "user")
            role = "assistant" if role == "model" else role
            parts = [_convert_part(p) for p in parts_list]
            msg = {"role": role, "parts": parts}
            result.append(msg)
        return result

    def extract_request_params(self, inputs: dict[str, Any]) -> dict[str, Any]:
        config = inputs.get("config")
        if not isinstance(config, dict):
            return {}
        # Remap Gemini-specific keys to the names the base class expects
        normalized = {**config}
        if "max_output_tokens" in normalized:
            normalized["max_tokens"] = normalized.pop("max_output_tokens")
        if "stop_sequences" in normalized:
            normalized["stop"] = normalized.pop("stop_sequences")
        params = super().extract_request_params(normalized)
        # Tools are set separately via set_span_chat_tools → mlflow.chat.tools,
        # so remove the raw (non-serializable) tool references from params.
        params.pop(GenAiSemconvKey.TOOL_DEFINITIONS, None)
        return params


def _convert_content_dict(content: dict[str, Any]) -> dict[str, Any]:
    role = content.get("role", "user")
    role = "assistant" if role == "model" else role
    parts = [_convert_part(p) for p in content.get("parts", [])]

    # function_response parts → role "tool"
    if parts and all(p.get("type") == "tool_call_response" for p in parts):
        return {"role": "tool", "parts": parts}

    return {"role": role, "parts": parts}


def _convert_part(part: Any) -> dict[str, Any]:
    if isinstance(part, str):
        return {"type": "text", "content": part}
    if not isinstance(part, dict):
        return {"type": "text", "content": str(part)}

    if (text := part.get("text")) is not None:
        return {"type": "text", "content": text}
    elif inline := part.get("inline_data"):
        mime_type = inline.get("mime_type", "")
        result = {
            "type": "blob",
            "mime_type": mime_type,
            "content": inline.get("data", ""),
        }
        if mime_type:
            result["modality"] = mime_type.split("/")[0]
        return result
    elif file_data := part.get("file_data"):
        mime_type = file_data.get("mime_type", "")
        result = {
            "type": "uri",
            "mime_type": mime_type,
            "uri": file_data.get("file_uri", ""),
        }
        if mime_type:
            result["modality"] = mime_type.split("/")[0]
        return result
    elif fc := part.get("function_call"):
        return {
            "type": "tool_call",
            "name": fc.get("name", ""),
            "arguments": fc.get("args", {}),
        }
    elif fr := part.get("function_response"):
        return {
            "type": "tool_call_response",
            "name": fr.get("name", ""),
            "result": fr.get("response", {}),
        }
    return {"type": "text", "content": json.dumps(part)}
