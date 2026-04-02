"""MLflow CLI commands for Codex CLI integration."""

from pathlib import Path

import click

from mlflow.codex.config import (
    CODEX_DIR_NAME,
    get_tracing_status,
)
from mlflow.codex.hooks import (
    disable_tracing_hooks,
    setup_hooks_config,
    stop_hook_handler,
)


@click.group("codex", invoke_without_command=True)
@click.option(
    "--directory",
    "-d",
    default=".",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Directory to set up tracing in (default: current directory)",
)
@click.option(
    "--tracking-uri", "-u", help="MLflow tracking URI (e.g., 'databricks' or 'file://mlruns')"
)
@click.option("--experiment-id", "-e", help="MLflow experiment ID")
@click.option("--experiment-name", "-n", help="MLflow experiment name")
@click.option("--disable", is_flag=True, help="Disable Codex tracing in the specified directory")
@click.option("--status", is_flag=True, help="Show current tracing status")
@click.pass_context
def codex(
    ctx: click.Context,
    directory: str,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
    disable: bool,
    status: bool,
) -> None:
    """Set up Codex CLI tracing in a directory.

    This command configures Codex CLI hooks to automatically trace conversations
    to MLflow. After setup, use the regular 'codex' command and traces will be
    automatically created.

    Examples:

      mlflow autolog codex
      mlflow autolog codex -d ~/my-project
      mlflow autolog codex -u databricks -e 123456789
      mlflow autolog codex --disable
    """
    if ctx.invoked_subcommand is not None:
        return

    target_dir = Path(directory).resolve()
    codex_dir = target_dir / CODEX_DIR_NAME

    if status:
        _show_status(target_dir, codex_dir)
        return

    if disable:
        if disable_tracing_hooks(codex_dir):
            click.echo("Codex tracing disabled")
        else:
            click.echo("No Codex configuration found - tracing was not enabled")
        return

    click.echo(f"Configuring Codex tracing in: {target_dir}")

    codex_dir.mkdir(parents=True, exist_ok=True)
    setup_hooks_config(codex_dir)
    click.echo("Codex CLI hooks configured")

    click.echo(f"\nDirectory: {target_dir}")
    if tracking_uri:
        click.echo(f"Tracking URI: {tracking_uri}")
    if experiment_id:
        click.echo(f"Experiment ID: {experiment_id}")
    elif experiment_name:
        click.echo(f"Experiment Name: {experiment_name}")

    click.echo("\nRun 'codex' in this directory to start tracing.")


def _show_status(target_dir: Path, codex_dir: Path) -> None:
    click.echo(f"Codex tracing status in: {target_dir}")

    tracing_status = get_tracing_status(codex_dir)

    if not tracing_status.enabled:
        click.echo("Codex tracing is not enabled")
        if tracing_status.reason:
            click.echo(f"  Reason: {tracing_status.reason}")
        return

    click.echo("Codex tracing is ENABLED")


@codex.command("stop-hook", hidden=True)
def stop_hook() -> None:
    """Hook handler invoked when a Codex CLI conversation ends."""
    stop_hook_handler()
