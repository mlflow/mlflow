"""Lightweight message types for the judge tool-calling loop.

These replace the litellm.Message and litellm.ChatCompletionMessageToolCall
types that were previously used as the data model throughout the judge code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class JudgeMessage:
    """A message in the judge conversation history.

    Uses the OpenAI message format. tool_calls entries are plain dicts
    in OpenAI format: {"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}
    """

    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
