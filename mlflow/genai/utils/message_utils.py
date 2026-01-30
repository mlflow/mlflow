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
