from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import mlflow.assistant.skills as _skills_pkg
from mlflow.agent.agents import AgentTool

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def _read_template(filename: str) -> str:
    return resources.files("mlflow.agent.setup.templates").joinpath(filename).read_text()


def _render(template: str, **values: str) -> str:
    def replace(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in values:
            raise KeyError(f"Missing template value: {key!r}")
        return values[key]

    return _PLACEHOLDER.sub(replace, template)


def _bundled_skills_root() -> Path:
    return Path(_skills_pkg.__path__[0])


def build_prompt(
    repo_root: Path,
    agent: AgentTool,
    tracking_uri: str,
    *,
    local_server_port: int | None = None,
    skills_installed: bool = True,
) -> str:
    """Compose the first user message handed to the agent.

    The shell (rules, execution requirements, verify, final summary) lives in
    ``instrument.md`` and is language-agnostic. The language-specific
    steps (install, tracking URI wiring, autolog snippet) come from
    ``<language>.md`` and are interpolated via ``{{ language_steps }}``.

    When ``local_server_port`` is not ``None``, the CLI picked it and built
    ``tracking_uri = http://127.0.0.1:<port>``; the agent is instructed to
    start a local MLflow server on that port.

    When ``skills_installed`` is ``False``, ``{{ skills_dir }}`` is
    redirected to the bundled skill location inside the MLflow install so
    the agent can still consult them without writing to the repo.
    """
    if skills_installed:
        skills_dir = agent.skills_dir
        skills_intro = (
            f"A set of MLflow skills has been installed at `{skills_dir}/`. "
            "Consult them for\nguidance."
        )
        no_overwrite_bullet = (
            "**Do not create setup-only files in the repo.** No scratch dirs, no agent\n"
            f"  task files. The skills at `{skills_dir}/` are already installed; do not\n"
            "  overwrite them."
        )
    else:
        skills_dir = _bundled_skills_root().as_posix()
        skills_intro = (
            f"MLflow skills are bundled at `{skills_dir}/`. Consult them in place. "
            "Do not\ncopy them into the repo."
        )
        no_overwrite_bullet = (
            "**Do not create setup-only files in the repo.** No scratch dirs, no agent\n"
            "  task files."
        )

    if local_server_port is not None:
        server_setup = _render(
            _read_template("local-server.md"),
            tracking_uri=tracking_uri,
            port=str(local_server_port),
        )
    else:
        server_setup = ""
    language_steps = _render(
        _read_template("python.md"),
        skills_dir=skills_dir,
        tracking_uri=tracking_uri,
        server_setup=server_setup,
    )
    return _render(
        _read_template("instrument.md"),
        repo_root=str(repo_root),
        skills_intro=skills_intro,
        no_overwrite_bullet=no_overwrite_bullet,
        tracking_uri=f"`{tracking_uri}`",
        language_steps=language_steps,
    )
