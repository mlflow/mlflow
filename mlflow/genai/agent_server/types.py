"""Type definitions for the agent server module."""

from typing import Literal

AgentType = Literal["agent/v1/responses", "agent/v1/chat", "agent/v2/chat"]
