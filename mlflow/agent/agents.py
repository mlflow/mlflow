"""Registry of coding agent CLIs supported by ``mlflow agent setup``.

To support a new agent, append an :class:`AgentTool` entry to :data:`AGENTS`.
That is the only place per-agent variation lives.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

AgentName = Literal["claude", "codex", "opencode"]


@dataclass(frozen=True)
class AgentTool:
    name: AgentName
    display_name: str
    binary: str
    # Repo-relative directory where this agent reads SKILL.md from.
    skills_dir: str
    # Home-relative directory where this agent reads user-global SKILL.md from.
    global_skills_dir: str
    # Environment variables the agent exports into the processes it spawns. Any one
    # of them being set means MLflow is running underneath that agent. Empty means
    # we know of no reliable marker, so the agent is never detected this way.
    env_markers: tuple[str, ...] = ()
    # Args inserted between the binary and the prompt at launch.
    interactive_args: tuple[str, ...] = ()

    def is_installed(self) -> bool:
        return shutil.which(self.binary) is not None

    def is_active(self) -> bool:
        """Whether this agent is the one driving the current process."""
        return any(os.environ.get(marker) for marker in self.env_markers)

    def skill_dirs(self, repo_root: Path) -> tuple[Path, ...]:
        """Directories this agent loads skills from, project first.

        Both project layouts are listed: ``skills_dir`` is where the agent looks
        natively, while ``mlflow agent setup``'s assistant flow writes project
        skills under ``global_skills_dir`` (`setup/cli.py`). They differ for
        Codex, and a skill installed either way must suppress the hint.
        """
        return (
            repo_root / self.skills_dir,
            repo_root / self.global_skills_dir,
            Path.home() / self.global_skills_dir,
        )


AGENTS: dict[AgentName, AgentTool] = {
    "claude": AgentTool(
        name="claude",
        display_name="Claude Code",
        binary="claude",
        skills_dir=".claude/skills",
        global_skills_dir=".claude/skills",
        env_markers=("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"),
    ),
    "codex": AgentTool(
        name="codex",
        display_name="OpenAI Codex",
        binary="codex",
        skills_dir=".agents/skills",
        global_skills_dir=".codex/skills",
        # Set by codex-rs when it spawns a sandboxed shell tool call. See
        # CODEX_SANDBOX_ENV_VAR in codex-rs/core/src/spawn.rs.
        env_markers=("CODEX_SANDBOX", "CODEX_SANDBOX_NETWORK_DISABLED"),
    ),
    "opencode": AgentTool(
        name="opencode",
        display_name="OpenCode",
        binary="opencode",
        skills_dir=".agents/skills",
        global_skills_dir=".config/opencode/skills",
        # No documented env marker, so OpenCode is never auto-detected.
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


def detect_active() -> AgentTool | None:
    """Return the coding agent driving this process, or ``None`` for a human shell."""
    return next((a for a in AGENTS.values() if a.is_active()), None)
