from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import click

from mlflow.agent.agents import AGENTS, AgentName, AgentTool, detect_installed, get_agent
from mlflow.agent.setup.prompt import build_prompt
from mlflow.agent.setup.select import arrow_select
from mlflow.assistant.skill_installer import install_skills
from mlflow.telemetry.events import AgentSetupEvent
from mlflow.telemetry.track import _record_event


def _find_available_port(start: int = 5000, end: int = 5100) -> int:
    for port in range(start, end):
        with socket.socket() as s:
            try:
                s.bind(("", port))
            except OSError:
                continue
            return port
    raise click.ClickException(f"No available port found in {start}-{end - 1}.")


def _git_root(start: Path) -> Path | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return Path(out.stdout.strip())


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


def _run_setup(
    agent_name: AgentName | None,
    print_prompt: bool,
    payload: dict[str, Any],
) -> tuple[list[str], Path] | None:
    """Run the interactive setup flow and return the agent launch command, or None for --print."""
    repo_root = _git_root(Path.cwd())
    if repo_root is None:
        raise click.ClickException("`mlflow agent setup` must be run inside a git working tree.")

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

    tracking_uri_input = click.prompt(
        click.style(
            "Tracking URI (leave empty to let the agent start a local server, "
            "or 'databricks' for a Databricks workspace)",
            fg="cyan",
            bold=True,
        ),
        default="",
        show_default=False,
        err=True,
    ).strip()
    experiment_path: str | None = None
    if tracking_uri_input:
        tracking_uri = tracking_uri_input
        local_server_port: int | None = None
        if tracking_uri == "databricks":
            experiment_path = click.prompt(
                click.style(
                    "Workspace experiment path (e.g. /Users/you@example.com/my-app)",
                    fg="cyan",
                    bold=True,
                ),
                err=True,
            ).strip()
            if not experiment_path.startswith("/"):
                raise click.ClickException(
                    "Databricks experiment path must start with '/' "
                    "(e.g. /Users/you@example.com/my-app)."
                )
    else:
        local_server_port = _find_available_port()
        tracking_uri = f"http://127.0.0.1:{local_server_port}"
        click.secho(f"Picked local tracking URI: {tracking_uri}", fg="green", err=True)

    prompt = build_prompt(
        repo_root,
        agent,
        tracking_uri,
        local_server_port=local_server_port,
        experiment_path=experiment_path,
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
        "Print the composed task prompt to stdout and skip launching the agent. "
        "Lets you pipe into a custom invocation, e.g. "
        "`mlflow agent setup --print | claude --permission-mode auto`."
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
