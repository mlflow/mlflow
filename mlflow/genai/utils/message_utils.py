from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def serialize_messages_to_prompts(
    messages: list[Any],
) -> tuple[str, str | None]:
    """
    Serialize messages to user_prompt and system_prompt strings.

    Handles both litellm Message objects and message dicts with 'role' and 'content' keys.
    This is needed because call_chat_completions only accepts string prompts.

    Args:
        messages: List of message objects or dicts.

    Returns:
        Tuple of (user_prompt, system_prompt). system_prompt may be None.
    """
    system_prompt = None
    user_parts = []

    for msg in messages:
        # Handle both object attributes and dict keys
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            name = msg.get("name")
        else:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", None)
            name = getattr(msg, "name", None)

        if role == "system":
            system_prompt = content
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            if tool_calls:
                user_parts.append("Assistant: [Called tools]")
            elif content:
                user_parts.append(f"Assistant: {content}")
        elif role == "tool":
            if name:
                user_parts.append(f"Tool {name}: {content}")
            else:
                user_parts.append(f"tool: {content}")
        else:
            user_parts.append(f"{role}: {content}")

    user_prompt = "\n\n".join(user_parts)
    return user_prompt, system_prompt


# Backwards compatibility aliases
def serialize_messages_to_databricks_prompts(
    messages: list[Any],
) -> tuple[str, str | None]:
    return serialize_messages_to_prompts(messages)


def serialize_chat_messages_to_prompts(
    messages: list[dict[str, Any]],
) -> tuple[str, str | None]:
    return serialize_messages_to_prompts(messages)


def _enforce_strict_json_schema(node: Any) -> None:
    """Recursively set ``additionalProperties: false`` on every object schema.

    OpenAI's strict structured output API (used by the MLflow AI Gateway and the
    ``openai``/``azure`` providers) rejects a ``json_schema`` response format
    unless every object - including those nested under ``$defs``, ``properties``,
    array ``items``, or combinators like ``anyOf`` - declares
    ``additionalProperties: false``. Pydantic's ``model_json_schema()`` does not
    emit this field, so we add it in place before sending the request.

    An object node is detected either by an explicit ``type: object`` or by the
    presence of ``properties``, since Pydantic does not always emit ``type`` on
    object schemas.
    """
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
        for value in node.values():
            _enforce_strict_json_schema(value)
    elif isinstance(node, list):
        for item in node:
            _enforce_strict_json_schema(item)


def pydantic_to_response_format(cls: type[BaseModel]) -> dict[str, Any]:
    schema = cls.model_json_schema()
    _enforce_strict_json_schema(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": cls.__name__,
            "schema": schema,
            "strict": True,
        },
    }
