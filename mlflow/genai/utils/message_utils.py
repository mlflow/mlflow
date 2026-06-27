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


# Providers that support OpenAI-style strict JSON schema enforcement via
# response_format={"type": "json_schema", "json_schema": {..., "strict": true}}
# with object schemas requiring additionalProperties: false.
#
# - openai: https://platform.openai.com/docs/guides/structured-outputs
# - azure: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs
# - databricks: https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference
#   (JsonSchemaObject.strict — "Whether to enable strict schema adherence when
#   generating the output")
# - "endpoints" is this codebase's internal URI scheme for Databricks serving endpoints
#   (endpoints:/my-endpoint → provider="endpoints"); same OpenAI-compatible API as "databricks".
#
# Other providers (anthropic, gemini, bedrock, etc.) may expose structured output
# through provider-native formats or adapter-specific translations, so enforcement
# semantics should be handled provider-specifically.
_STRICT_JSON_SCHEMA_PROVIDERS: frozenset[str] = frozenset({
    "openai",
    "azure",
    "databricks",
    "endpoints",
})


def pydantic_to_response_format(
    cls: type[BaseModel], provider: str | None = None
) -> dict[str, Any]:
    strict = provider is not None and provider in _STRICT_JSON_SCHEMA_PROVIDERS
    return {
        "type": "json_schema",
        "json_schema": {
            "name": cls.__name__,
            "schema": {
                **cls.model_json_schema(),
                **({"additionalProperties": False} if strict else {}),
            },
            **({"strict": True} if strict else {}),
        },
    }
