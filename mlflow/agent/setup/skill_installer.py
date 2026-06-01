"""Install the bundled MLflow skills into a repo for the chosen agent."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from mlflow.agent.agents import AgentTool
from mlflow.assistant.skill_installer import install_skills as _install_skills


def _read_template(filename: str) -> str:
    return resources.files("mlflow.agent.setup.templates").joinpath(filename).read_text()


def skills_dest(repo_root: Path, agent: AgentTool) -> Path:
    return repo_root / agent.skills_dir


def install_skills(repo_root: Path, agent: AgentTool) -> list[str]:
    """Copy the curated MLflow skills into the agent's skills dir."""
    return _install_skills(skills_dest(repo_root, agent))


def build_task(
    repo_root: Path,
    agent: AgentTool,
    tracking_uri: str,
) -> str:
    """Compose the first user message handed to the agent.

    The shell (rules, execution requirements, verify, final summary) lives in
    ``instrument-task.md`` and is language-agnostic. The language-specific
    steps (install, tracking URI wiring, autolog snippet) come from
    ``<language>.md`` and are interpolated via ``{language_steps}``.
    """
    language_steps = _read_template("python.md").format(
        skills_dir=agent.skills_dir,
        tracking_uri=tracking_uri,
    )
    return _read_template("instrument-task.md").format(
        repo_root=repo_root,
        skills_dir=agent.skills_dir,
        tracking_uri=tracking_uri,
        language_steps=language_steps,
    )
