from __future__ import annotations

from importlib import resources
from pathlib import Path

from mlflow.agent.agents import AgentTool


def _read_template(filename: str) -> str:
    return resources.files("mlflow.agent.setup.templates").joinpath(filename).read_text()


def build_task(
    repo_root: Path,
    agent: AgentTool,
    tracking_uri: str,
    *,
    started_local_server: bool = False,
) -> str:
    """Compose the first user message handed to the agent.

    The shell (rules, execution requirements, verify, final summary) lives in
    ``instrument-task.md`` and is language-agnostic. The language-specific
    steps (install, tracking URI wiring, autolog snippet) come from
    ``<language>.md`` and are interpolated via ``{language_steps}``.

    When ``started_local_server`` is ``True``, ``tracking_uri`` is a
    ``http://127.0.0.1:<port>`` URL picked by the CLI and the agent is
    instructed to start a local MLflow server bound to that URL.
    """
    if started_local_server:
        port = tracking_uri.rsplit(":", 1)[1]
        server_setup = _read_template("local-server.md").format(
            tracking_uri=tracking_uri, port=port
        )
    else:
        server_setup = ""
    language_steps = _read_template("python.md").format(
        skills_dir=agent.skills_dir,
        tracking_uri=tracking_uri,
        server_setup=server_setup,
    )
    return _read_template("instrument-task.md").format(
        repo_root=repo_root,
        skills_dir=agent.skills_dir,
        tracking_uri=f"`{tracking_uri}`",
        language_steps=language_steps,
    )
