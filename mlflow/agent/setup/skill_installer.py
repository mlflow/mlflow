from __future__ import annotations

from pathlib import Path

from mlflow.agent.agents import AgentTool
from mlflow.assistant.skill_installer import install_skills as _install_skills


def skills_dest(repo_root: Path, agent: AgentTool) -> Path:
    return repo_root / agent.skills_dir


def install_skills(repo_root: Path, agent: AgentTool) -> list[str]:
    """Copy the curated MLflow skills into the agent's skills dir."""
    return _install_skills(skills_dest(repo_root, agent))
