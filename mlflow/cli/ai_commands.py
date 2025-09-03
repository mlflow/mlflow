"""CLI commands for managing MLflow commands."""

import click

from mlflow.commands import get_command, list_commands


@click.group("ai-commands")
def commands():
    """Manage MLflow AI commands for LLMs."""


@commands.command("list")
@click.option("--namespace", help="Filter commands by namespace")
def list_cmd(namespace):
    """List all available commands."""
    cmd_list = list_commands(namespace)

    if not cmd_list:
        if namespace:
            click.echo(f"No commands found in namespace '{namespace}'")
        else:
            click.echo("No commands found")
        return

    for cmd in cmd_list:
        click.echo(f"{cmd['key']}: {cmd['description']}")


@commands.command("get")
@click.argument("key")
def get_cmd(key):
    """Get a specific command by key."""
    try:
        content = get_command(key)
        click.echo(content)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
