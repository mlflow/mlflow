"""Thin wrapper around the Anthropic Messages API.

Two responsibilities:
  1. Hold the model selection (Sonnet by default; Opus when `--hybrid` is set
     for the spotter-discovery call only).
  2. Provide a single retry-aware entry point so the orchestrator code does
     not need to think about transient 429/500 responses.

Stack 1 ships only the data classes and the construction surface. The actual
Messages API call lands in Stack 2 alongside the orchestrator flow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


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
    the spotter-discovery step swaps to Opus while the kjc9 standalone and
    kjc9 opinion calls stay on Sonnet.
    """

    spotter: Role = Role.SONNET
    kjc9_standalone: Role = Role.SONNET
    kjc9_opinion: Role = Role.SONNET
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
                "ANTHROPIC_API_KEY is not set. The orchestrator requires direct API access; "
                "Claude Code subscription credentials are not used in production."
            )
        return cls(api_key=key)


def model_id(role: Role) -> str:
    return _MODEL_IDS[role]
