"""Custom View delivery for structured-output providers (Claude Code and Codex).

This module converts a structured response envelope into a terminal ``client_tool_call``.
Native-tool providers use ``tool_executor`` instead.
"""

import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from mlflow.assistant.types import Event, Message, TextBlock, ToolUseBlock

RENDER_CUSTOM_VIEW_TOOL_NAME = "render_custom_view"

CUSTOM_VIEW_RESPONSE_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["message", RENDER_CUSTOM_VIEW_TOOL_NAME]},
        "text": {
            "type": "string",
            "description": (
                "The conversational response shown in chat. For a rendered view, briefly "
                "describe what was created or changed."
            ),
        },
        "title": {
            "type": "string",
            "description": (
                "A short display title when type is render_custom_view; otherwise an empty string."
            ),
        },
        "messages": {
            "type": "array",
            "description": (
                "The complete A2UI message list when type is render_custom_view; otherwise empty."
            ),
            "items": {"type": "object"},
        },
    },
    "required": ["type", "text", "title", "messages"],
    "additionalProperties": False,
}

# Codex's --output-schema uses strict OpenAI structured outputs, where every object
# must declare a closed set of properties. A2UI messages are intentionally open-ended,
# so encode that array as JSON within the otherwise strict response envelope.
STRINGIFIED_CUSTOM_VIEW_RESPONSE_SCHEMA: dict[str, Any] = {
    **CUSTOM_VIEW_RESPONSE_SCHEMA,
    "properties": {
        **CUSTOM_VIEW_RESPONSE_SCHEMA["properties"],
        "messages": {
            "type": "string",
            "description": (
                "The complete A2UI message array encoded as JSON when type is "
                "render_custom_view; otherwise the JSON string '[]'."
            ),
        },
    },
}

CUSTOM_VIEW_STRUCTURED_OUTPUT_INSTRUCTIONS = f"""\
Your final response MUST match the provided JSON schema. Set "type" to
"{RENDER_CUSTOM_VIEW_TOOL_NAME}" only when the user asks to build or modify the custom view;
put the complete A2UI specification in "messages" and a short view name in "title". For any
other response, set "type" to "message", answer in "text", and return an empty "title" and
"messages" array. Do not call a render tool and do not wrap the JSON in a code fence.
"""

STRINGIFIED_CUSTOM_VIEW_STRUCTURED_OUTPUT_INSTRUCTIONS = f"""\
{CUSTOM_VIEW_STRUCTURED_OUTPUT_INSTRUCTIONS}
For this provider, the response schema defines "messages" as a string. JSON-encode the complete
A2UI message array into that string. For a normal message, return "messages": "[]". This is only
the transport encoding: the decoded value must be the same A2UI message array described above.
The decoded string must start with "[", end with "]", and contain nothing after that final "]".
"""


def _parse_stringified_messages(value: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as original_error:
        try:
            parsed, end = json.JSONDecoder().raw_decode(value)
        except json.JSONDecodeError:
            raise ValueError("messages must be a JSON-encoded array") from original_error
        else:
            # Codex can close the strict outer response envelope inside the string,
            # leaving a complete A2UI array followed by an extra `}`. Discard only
            # a short suffix of unmatched closing delimiters; never repair content.
            trailing = value[end:].strip()
            if isinstance(parsed, list) and 0 < len(trailing) <= 4 and set(trailing) <= {"}", "]"}:
                return parsed
        raise ValueError("messages must be a JSON-encoded array") from original_error
    if not isinstance(parsed, list):
        raise ValueError("messages must be a JSON-encoded array")
    return parsed


class CustomViewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["message", "render_custom_view"]
    text: str
    title: str
    messages: list[dict[str, Any]]

    @field_validator("messages", mode="before")
    @classmethod
    def parse_stringified_messages(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _parse_stringified_messages(value)
        return value


def parse_custom_view_response(value: Any) -> CustomViewResponse:
    if isinstance(value, str):
        value = json.loads(value)
    return CustomViewResponse.model_validate(value)


def custom_view_response_events(response: CustomViewResponse) -> list[Event]:
    events = []
    if response.text:
        events.append(
            Event.from_message(Message(role="assistant", content=[TextBlock(text=response.text)]))
        )

    if response.type == RENDER_CUSTOM_VIEW_TOOL_NAME:
        request_id = str(uuid.uuid4())
        tool_input = {"title": response.title, "messages": response.messages}
        events.extend([
            Event.from_message(
                Message(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id=request_id,
                            name=RENDER_CUSTOM_VIEW_TOOL_NAME,
                            input=tool_input,
                        )
                    ],
                )
            ),
            Event.from_client_tool_call(
                request_id,
                RENDER_CUSTOM_VIEW_TOOL_NAME,
                tool_input,
                continuation="terminal",
            ),
        ])
    return events


def is_custom_view_request(context: dict[str, Any] | None) -> bool:
    return bool(context and context.get("customTraceView"))
