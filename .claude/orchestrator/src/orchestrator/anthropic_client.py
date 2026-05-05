"""Async wrapper around the Anthropic Messages API.

Holds model selection (per-step Sonnet/Opus picks) and provides a single
retry-aware entry point so the orchestrator code does not need to think about
transient 429/500 responses.

Prompt caching is enabled on the system prompt so the agent definition (the
fixed part of every call) lands in the cache and subsequent calls within the
same session pay the cache-read rate instead of the full input rate.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import anthropic
from anthropic import AsyncAnthropic

_logger = logging.getLogger(__name__)


class Role(str, Enum):
    SONNET = "sonnet"
    OPUS = "opus"


_MODEL_IDS: dict[Role, str] = {
    Role.SONNET: "claude-sonnet-4-6",
    Role.OPUS: "claude-opus-4-7",
}


@dataclass(frozen=True)
class ModelChoice:
    """Selects which Anthropic model handles a given orchestrator step.

    The default in M1 is Sonnet for every call. When invoked with `--hybrid`,
    the spotter-discovery step swaps to Opus while the reviewer calls stay on
    Sonnet.
    """

    spotter: Role = Role.SONNET
    reviewer_standalone: Role = Role.SONNET
    reviewer_opinion: Role = Role.SONNET
    cluster_judge: Role = Role.SONNET
    semantic_dedup_judge: Role = Role.SONNET

    @classmethod
    def default(cls) -> ModelChoice:
        return cls()

    @classmethod
    def hybrid(cls) -> ModelChoice:
        return cls(spotter=Role.OPUS)


@dataclass(frozen=True)
class AnthropicConfig:
    api_key: str
    max_retries: int = 3
    timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> AnthropicConfig:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. The orchestrator requires direct API "
                "access; Claude Code subscription credentials are not used in production."
            )
        return cls(api_key=key)


def model_id(role: Role) -> str:
    return _MODEL_IDS[role]


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class CompletionResult:
    text: str
    role: Role
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int


class AnthropicClient:
    """Thin async wrapper. One instance per orchestrator run."""

    def __init__(self, config: AnthropicConfig) -> None:
        self._config = config
        self._client = AsyncAnthropic(api_key=config.api_key, timeout=config.timeout_seconds)

    async def complete(
        self,
        *,
        role: Role,
        system: str,
        messages: list[Message],
        max_tokens: int = 8192,
        cache_system: bool = True,
    ) -> CompletionResult:
        """Send a Messages API request and return the assistant's text reply.

        `cache_system=True` marks the system prompt for prompt caching so the
        agent definition (~5-10K tokens of fixed content) is billed at cache-read
        rate on subsequent calls within the same orchestrator run.
        """
        for attempt in range(self._config.max_retries):
            try:
                response = await self._client.messages.create(
                    model=model_id(role),
                    max_tokens=max_tokens,
                    system=[
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"} if cache_system else None,
                        }
                    ]
                    if cache_system
                    else system,
                    messages=[{"role": m.role, "content": m.content} for m in messages],
                )
                text_block = next(
                    (b for b in response.content if b.type == "text"),
                    None,
                )
                if text_block is None:
                    raise RuntimeError(
                        f"Anthropic response had no text block: {response.content!r}"
                    )
                usage = response.usage
                return CompletionResult(
                    text=text_block.text,
                    role=role,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                )
            except (anthropic.APITimeoutError, anthropic.RateLimitError) as e:
                if attempt == self._config.max_retries - 1:
                    raise
                backoff = 2**attempt
                _logger.warning(
                    "Anthropic retryable error on attempt %d/%d: %s. Retrying in %ds.",
                    attempt + 1,
                    self._config.max_retries,
                    e,
                    backoff,
                )
                await asyncio.sleep(backoff)
        raise RuntimeError("unreachable")
