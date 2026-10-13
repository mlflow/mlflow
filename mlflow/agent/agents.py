"""Registry of coding agent CLIs supported by ``mlflow agent setup``.

To support a new agent, append an :class:`AgentTool` entry to :data:`AGENTS`.
That is the only place per-agent variation lives.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Literal

AgentName = Literal["claude", "codex", "opencode"]


@dataclass(frozen=True)
class AgentTool:
    name: AgentName
    display_name: str
    binary: str
    # Repo-relative directory where this agent reads SKILL.md from.
    skills_dir: str
    # Args inserted between the binary and the prompt at launch.
    interactive_args: tuple[str, ...] = ()

    def is_installed(self) -> bool:
        return shutil.which(self.binary) is not None


AGENTS: dict[AgentName, AgentTool] = {
    "claude": AgentTool(
        name="claude",
        display_name="Claude Code",
        binary="claude",
        skills_dir=".claude/skills",
    ),
    "codex": AgentTool(
        name="codex",
        display_name="OpenAI Codex",
        binary="codex",
        skills_dir=".agents/skills",
    ),
    "opencode": AgentTool(
        name="opencode",
        display_name="OpenCode",
        binary="opencode",
        skills_dir=".agents/skills",
        interactive_args=("--prompt",),
    ),
}


def get_agent(name: AgentName) -> AgentTool:
    if agent := AGENTS.get(name):
        return agent
    available = ", ".join(sorted(AGENTS))
    raise ValueError(f"Unknown agent {name!r}. Available: {available}")


def detect_installed() -> list[AgentTool]:
    return [a for a in AGENTS.values() if a.is_installed()]
