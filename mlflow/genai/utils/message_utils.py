from __future__ import annotations

from typing import Any


def serialize_messages_to_databricks_prompts(
    messages: list[Any],
) -> tuple[str, str | None]:
    """
    Serialize litellm Messages to user_prompt and system_prompt for Databricks.

    This is needed because call_chat_completions only accepts string prompts.

    Args:
        messages: List of litellm Message objects.

    Returns:
        Tuple of (user_prompt, system_prompt).
    """
    system_prompt = None
    user_parts = []

    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_parts.append(msg.content)
        elif msg.role == "assistant":
            if msg.tool_calls:
                user_parts.append("Assistant: [Called tools]")
            elif msg.content:
                user_parts.append(f"Assistant: {msg.content}")
        elif msg.role == "tool":
            user_parts.append(f"Tool {msg.name}: {msg.content}")

    user_prompt = "\n\n".join(user_parts)
    return user_prompt, system_prompt


def serialize_chat_messages_to_prompts(
    messages: list[dict[str, Any]],
) -> tuple[str, str | None]:
    """
    Serialize chat messages (as dicts) to user_prompt and system_prompt strings.

    This is used by third-party integrations that receive messages as dicts
    and need to convert them for Databricks endpoints.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        Tuple of (user_prompt, system_prompt). system_prompt may be None.
    """
    system_prompt = None
    user_parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_prompt = content
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            if content:
                user_parts.append(f"Assistant: {content}")
        else:
            user_parts.append(f"{role}: {content}")

    user_prompt = "\n\n".join(user_parts)
    return user_prompt, system_prompt
