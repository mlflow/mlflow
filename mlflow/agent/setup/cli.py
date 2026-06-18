from __future__ import annotations

import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import click

from mlflow.agent.agents import AGENTS, AgentName, AgentTool, detect_installed, get_agent
from mlflow.agent.setup.prompt import build_prompt
from mlflow.agent.setup.select import arrow_select
from mlflow.assistant.config import AssistantConfig, SkillsConfig
from mlflow.assistant.skill_installer import install_skills
from mlflow.telemetry.events import AgentSetupEvent
from mlflow.telemetry.track import _record_event
from mlflow.tracking import MlflowClient


def _resolve_experiment_id(tracking_uri: str, ref: str) -> str:
    """Return an experiment ID. Path inputs are looked up (or created) via the workspace."""
    if not ref.startswith("/"):
        return ref
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(ref)
    if exp is not None:
        return exp.experiment_id
    experiment_id = client.create_experiment(ref)
    click.secho(f"Created experiment {ref!r} (ID {experiment_id}).", fg="green", err=True)
    return experiment_id


def _find_available_port(start: int = 5000, end: int = 5100) -> int:
    for port in range(start, end):
        with socket.socket() as s:
            try:
                s.bind(("", port))
            except OSError:
                continue
            return port
    raise click.ClickException(f"No available port found in {start}-{end - 1}.")


def _git_root(start: Path) -> tuple[Path | None, str | None]:
    """Return (repo_root, reason); the reason explains why repo_root is None."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None, "Git is not installed."
    except subprocess.CalledProcessError:
        return None, "Not inside a git repository."
    return Path(out.stdout.strip()), None


def _choose_agent(preferred: AgentName | None) -> AgentTool:
    if preferred:
        agent = get_agent(preferred)
        if not agent.is_installed():
            raise click.ClickException(
                f"{agent.display_name} CLI ({agent.binary!r}) not found on PATH."
            )
        return agent

    installed = detect_installed()
    match installed:
        case []:
            available = ", ".join(a.display_name for a in AGENTS.values())
            raise click.ClickException(
                f"No supported agent CLI found on PATH. Install one of: {available}."
            )
        case [only]:
            click.echo(f"Using {only.display_name} (only installed agent detected).", err=True)
            return only
        case _:
            idx = arrow_select(
                "Multiple agents detected. Select one:",
                [a.display_name for a in installed],
            )
            return installed[idx]


@dataclass(frozen=True)
class _AssistantTarget:
    """The in-app Assistant provider that a coding agent maps to."""

    config_name: str  # key under AssistantConfig.providers
    skills_subdir: str  # global skills dir relative to home (mirrors provider.resolve_skills_path)


# Coding agents whose CLI doubles as an in-app MLflow Assistant provider.
_ASSISTANT_PROVIDERS: dict[AgentName, _AssistantTarget] = {
    "claude": _AssistantTarget(config_name="claude_code", skills_subdir=".claude/skills"),
    "codex": _AssistantTarget(config_name="codex", skills_subdir=".codex/skills"),
}


def _is_localhost_tracking_uri(tracking_uri: str) -> bool:
    """Whether the localhost-only Assistant API can reach this tracking server."""
    parsed = urlparse(tracking_uri if "://" in tracking_uri else f"http://{tracking_uri}")
    host = (parsed.hostname or "").lower()
    return host in ("localhost", "::1") or host.startswith("127.")


def _offer_assistant_setup(agent: AgentTool, tracking_uri: str) -> bool | None:
    """Optionally select `agent` as the in-app MLflow Assistant provider.

    Selects the matching provider in the Assistant config and offers to install its
    skills into the global user directory. Only offered for agents that have an
    Assistant provider and when the tracking server is reachable from localhost (the
    Assistant API is localhost-only). An existing provider configuration is preserved
    (model and skills location are left untouched); only the selection is updated.

    Skills are installed globally because `agent setup` does not create an
    experiment->path mapping, and the in-app Assistant resolves project-level skills
    from that mapping; a project-level install would therefore not be discovered.

    Returns None when the Assistant isn't applicable (no matching provider or the
    tracking server isn't reachable from localhost), False when offered but declined,
    and True when configured.
    """
    target = _ASSISTANT_PROVIDERS.get(agent.name)
    if target is None or not _is_localhost_tracking_uri(tracking_uri):
        return None

    if not click.confirm(
        click.style(
            f"Also enable the in-app MLflow Assistant with {agent.display_name}?",
            fg="cyan",
            bold=True,
        ),
        default=True,
        err=True,
    ):
        return False

    config = AssistantConfig.load()
    if existing := config.providers.get(target.config_name):
        # Preserve the user's existing model and skills location; only select it.
        config.set_provider(target.config_name, model=existing.model)
        click.secho(
            f"Selected {agent.display_name} as the MLflow Assistant provider "
            "(kept your existing configuration).",
            fg="green",
            err=True,
        )
    else:
        config.set_provider(target.config_name, model="default")
        config.providers[target.config_name].skills = SkillsConfig(type="global")
        click.secho(f"Enabled the MLflow Assistant ({agent.display_name}).", fg="green", err=True)
    config.save()

    skills_dest = Path.home() / target.skills_subdir
    if click.confirm(
        click.style(f"Install MLflow skills to {skills_dest}?", fg="cyan", bold=True),
        default=True,
        err=True,
    ):
        installed = install_skills(skills_dest)
        click.secho(f"Installed {len(installed)} skill(s) to {skills_dest}.", fg="green", err=True)

    return True


def _run_setup(
    agent_name: AgentName | None,
    print_prompt: bool,
    payload: dict[str, Any],
) -> tuple[list[str], Path] | None:
    """Run the interactive setup flow and return the agent launch command, or None for --print."""
    repo_root, reason = _git_root(Path.cwd())
    if repo_root is None:
        click.secho(
            f"{reason} The agent's edits cannot be reviewed or reverted with git.",
            fg="yellow",
            err=True,
        )
        repo_root = Path.cwd()

    agent = _choose_agent(agent_name)
    payload["agent"] = agent.name

    skills_dest = repo_root / agent.skills_dir
    skills_choice = arrow_select(
        f"Install MLflow skills at {agent.skills_dir}/ (this project)?",
        ["Install", "Skip"],
    )
    skills_installed = skills_choice == 0
    payload["skills_install_confirmed"] = skills_installed
    if skills_installed:
        installed = install_skills(skills_dest)
        click.secho(
            f"Wrote {len(installed)} skill(s) to {agent.skills_dir}/:", fg="green", err=True
        )
        for name in installed:
            click.echo(f"  - {name}", err=True)
    else:
        click.secho("Skipping skill installation.", fg="yellow", err=True)

    backend_choice = arrow_select(
        "Tracking backend:",
        [
            "Start a new local server",
            "Databricks workspace",
            "Existing server URL (e.g. http://localhost:5000)",
        ],
    )
    experiment_id: str | None = None
    local_server_port: int | None = None
    match backend_choice:
        case 0:
            local_server_port = _find_available_port()
            tracking_uri = f"http://127.0.0.1:{local_server_port}"
            click.secho(f"Picked local tracking URI: {tracking_uri}", fg="green", err=True)
        case 1:
            profile = click.prompt(
                click.style(
                    "Databricks configuration profile, or empty for default",
                    fg="cyan",
                    bold=True,
                ),
                default="",
                show_default=False,
                err=True,
            ).strip()
            tracking_uri = f"databricks://{profile}" if profile else "databricks"
            experiment_ref = click.prompt(
                click.style(
                    "Experiment ID, or path (auto-created if it doesn't exist)",
                    fg="cyan",
                    bold=True,
                ),
                err=True,
            ).strip()
            experiment_id = _resolve_experiment_id(tracking_uri, experiment_ref)
        case _:
            tracking_uri = click.prompt(
                click.style("Tracking server URL", fg="cyan", bold=True),
                err=True,
            ).strip()

    payload["assistant_configured"] = _offer_assistant_setup(agent, tracking_uri)

    prompt = build_prompt(
        repo_root,
        agent,
        tracking_uri,
        local_server_port=local_server_port,
        experiment_id=experiment_id,
        skills_installed=skills_installed,
    )

    if print_prompt:
        click.echo(prompt)
        return None

    cmd = [agent.binary, *agent.interactive_args, prompt]
    click.echo(err=True)
    click.secho(f"Launching {agent.display_name}...", fg="cyan", err=True)
    return cmd, repo_root


@click.command("setup")
@click.option(
    "--agent",
    "agent_name",
    type=click.Choice(sorted(AGENTS)),
    default=None,
    help="Coding agent to set up. If omitted, picks from installed agents.",
)
@click.option(
    "--print",
    "print_prompt",
    is_flag=True,
    default=False,
    help=(
        "Print the composed task prompt to stdout and exit without launching the agent. "
        "Useful for passing the prompt into a custom invocation, e.g. "
        '`claude --permission-mode auto "$(mlflow agent setup --agent claude --print)"`.'
    ),
)
def setup(
    agent_name: AgentName | None,
    print_prompt: bool,
):
    """[Experimental] Install MLflow skills and launch a coding agent to instrument this repo."""
    click.secho(
        "[Experimental] `mlflow agent setup` is experimental and may change without notice.",
        fg="yellow",
        err=True,
    )

    success = False
    payload = {
        "agent": None,
        "print_prompt": print_prompt,
        "skills_install_confirmed": None,
        "assistant_configured": None,
    }
    try:
        launch = _run_setup(agent_name, print_prompt, payload)
        success = True
    finally:
        # Record before handing off to the agent's TUI so a force-aborted session
        # (kill -9, terminal closed) doesn't drop the setup event.
        _record_event(AgentSetupEvent, payload, success=success)

    if launch is None:
        return

    cmd, cwd = launch
    # Inherit stdio so the agent's TUI takes over until the user exits.
    result = subprocess.run(cmd, cwd=cwd)
    sys.exit(result.returncode)
