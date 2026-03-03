import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# Message interface between assistant providers and the assistant client
# Inspired by https://github.com/anthropics/claude-agent-sdk-python/blob/29c12cd80b256e88f321b2b8f1f5a88445077aa5/src/claude_agent_sdk/types.py


class TextBlock(BaseModel):
    """Text content block."""

    text: str


class ThinkingBlock(BaseModel):
    """Thinking content block."""

    thinking: str
    signature: str


class ToolUseBlock(BaseModel):
    """Tool use content block."""

    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(BaseModel):
    """Tool result content block."""

    tool_use_id: str
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None


ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock


class Message(BaseModel):
    """Structured message representation for assistant conversations.

    Uses standard chat message format with role and content fields.
    Can be extended in the future to support multi-modal content.
    """

    role: Literal["user", "assistant", "system"] = Field(description="Role of the message sender")
    content: str | list[ContentBlock] = Field(description="Content of the message")


class EventType(str, Enum):
    MESSAGE = "message"
    STREAM_EVENT = "stream_event"
    DONE = "done"
    ERROR = "error"
    INTERRUPTED = "interrupted"

    def __str__(self):
        return self.value


class Event(BaseModel):
    """A common event format parsed from the raw assistant provider output."""

    type: EventType
    data: dict[str, Any]

    def to_sse_event(self) -> str:
        """Convert the event to an SSE event string."""
        return f"event: {self.type}\ndata: {json.dumps(self.data)}\n\n"

    @classmethod
    def from_error(cls, error: str) -> "Event":
        return cls(type=EventType.ERROR, data={"error": error})

    @classmethod
    def from_message(cls, message: Message) -> "Event":
        return cls(type=EventType.MESSAGE, data={"message": message.model_dump()})

    @classmethod
    def from_stream_event(cls, event: dict[str, Any]) -> "Event":
        return cls(type=EventType.STREAM_EVENT, data={"event": event})

    @classmethod
    def from_result(cls, result: Any, session_id: str) -> "Event":
        return cls(type=EventType.DONE, data={"result": result, "session_id": session_id})

    @classmethod
    def from_interrupted(cls) -> "Event":
        return cls(type=EventType.INTERRUPTED, data={"message": "Assistant was interrupted"})
