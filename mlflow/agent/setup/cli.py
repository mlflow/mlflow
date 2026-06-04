from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import click

from mlflow.agent.agents import AGENTS, AgentName, AgentTool, detect_installed, get_agent
from mlflow.agent.setup.task import build_task
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
            click.secho("Multiple agents detected:", bold=True, err=True)
            for i, a in enumerate(installed, 1):
                click.echo(f"  {click.style(str(i), fg='cyan')}. {a.display_name}", err=True)
            choice = click.prompt(
                click.style("Select agent", fg="cyan", bold=True),
                type=click.IntRange(1, len(installed)),
                default=1,
                err=True,
            )
            return installed[choice - 1]


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

    repo_root = _git_root(Path.cwd())
    if repo_root is None:
        raise click.ClickException(
            "`mlflow agent setup` must be run inside a git repository (--local scope)."
        )

    agent = _choose_agent(agent_name)
    _record_event(AgentSetupEvent, {"agent": agent.name, "print_prompt": print_prompt})

    skills_dest = repo_root / agent.skills_dir
    skills_installed = click.confirm(
        click.style(
            f"Install MLflow skills at {agent.skills_dir}/ (this project)?",
            fg="cyan",
            bold=True,
        ),
        default=True,
        err=True,
    )
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
            "Tracking URI (leave empty to let the agent start a local server)",
            fg="cyan",
            bold=True,
        ),
        default="",
        show_default=False,
        err=True,
    ).strip()
    if tracking_uri_input:
        tracking_uri = tracking_uri_input
        local_server_port: int | None = None
    else:
        local_server_port = _find_available_port()
        tracking_uri = f"http://127.0.0.1:{local_server_port}"
        click.secho(f"Picked local tracking URI: {tracking_uri}", fg="green", err=True)

    task = build_task(
        repo_root,
        agent,
        tracking_uri,
        local_server_port=local_server_port,
        skills_installed=skills_installed,
    )

    if print_prompt:
        click.echo(task)
        return

    cmd = [agent.binary, *agent.interactive_args, task]
    click.echo(err=True)
    click.secho(f"Launching {agent.display_name}...", fg="cyan", err=True)
    # Inherit stdio so the agent's TUI takes over until the user exits.
    result = subprocess.run(cmd, cwd=repo_root)
    sys.exit(result.returncode)
