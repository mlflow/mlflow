"""CLI commands for managing MLflow AI commands."""

import click

from mlflow.ai_commands.ai_command_utils import (
    get_command,
    get_command_body,
    list_commands,
    parse_frontmatter,
)
from mlflow.telemetry.events import AiCommandRunEvent
from mlflow.telemetry.track import _record_event

__all__ = ["get_command", "get_command_body", "list_commands", "parse_frontmatter", "commands"]


@click.group("ai-commands")
def commands() -> None:
    """Manage MLflow AI commands for LLMs."""


@commands.command("list")
@click.option("--namespace", help="Filter commands by namespace")
def list_cmd(namespace: str | None) -> None:
    """List all available AI commands."""
    cmd_list = list_commands(namespace)

    if not cmd_list:
        if namespace:
            click.echo(f"No AI commands found in namespace '{namespace}'")
        else:
            click.echo("No AI commands found")
        return

    for cmd in cmd_list:
        click.echo(f"{cmd['key']}: {cmd['description']}")


@commands.command("get")
@click.argument("key")
def get_cmd(key: str) -> None:
    """Get a specific AI command by key."""
    try:
        content = get_command(key)
        click.echo(content)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@commands.command("run")
@click.argument("key")
def run_cmd(key: str) -> None:
    """Get a command formatted for execution by an AI assistant."""
    try:
        _record_event(AiCommandRunEvent, {"command_key": key, "context": "cli"})

        content = get_command(key)
        _, body = parse_frontmatter(content)

        # Add prefix instructing the assistant to execute the workflow
        prefix = (
            "The user has run an MLflow AI command via CLI. "
            "Start executing the workflow immediately without any preamble.\n\n"
        )

        click.echo(prefix + body)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
