"""MLflow CLI commands for Qwen Code integration."""

from pathlib import Path

import click

from mlflow.qwen_code.config import (
    QWEN_DIR_NAME,
    get_tracing_status,
    setup_environment_config,
)
from mlflow.qwen_code.hooks import (
    disable_tracing_hooks,
    setup_hooks_config,
    stop_hook_handler,
)


@click.group("qwen-code", invoke_without_command=True)
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
@click.option("--disable", is_flag=True, help="Disable Qwen tracing in the specified directory")
@click.option("--status", is_flag=True, help="Show current tracing status")
@click.pass_context
def qwen_code(
    ctx: click.Context,
    directory: str,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
    disable: bool,
    status: bool,
) -> None:
    """Set up Qwen Code tracing in a directory.

    This command configures Qwen Code hooks to automatically trace conversations
    to MLflow. After setup, use the regular 'qwen' command and traces will be
    automatically created.

    Examples:

      mlflow autolog qwen-code
      mlflow autolog qwen-code -d ~/my-project
      mlflow autolog qwen-code -u databricks -e 123456789
      mlflow autolog qwen-code --disable
    """
    if ctx.invoked_subcommand is not None:
        return

    target_dir = Path(directory).resolve()
    qwen_dir = target_dir / QWEN_DIR_NAME
    settings_file = qwen_dir / "settings.json"

    if status:
        _show_status(target_dir, settings_file)
        return

    if disable:
        if disable_tracing_hooks(qwen_dir):
            click.echo("Qwen Code tracing disabled")
        else:
            click.echo("No Qwen Code configuration found - tracing was not enabled")
        return

    click.echo(f"Configuring Qwen Code tracing in: {target_dir}")

    qwen_dir.mkdir(parents=True, exist_ok=True)
    setup_hooks_config(qwen_dir)
    click.echo("Qwen Code hooks configured")

    setup_environment_config(settings_file, tracking_uri, experiment_id, experiment_name)

    click.echo(f"\nDirectory: {target_dir}")
    if tracking_uri:
        click.echo(f"Tracking URI: {tracking_uri}")
    if experiment_id:
        click.echo(f"Experiment ID: {experiment_id}")
    elif experiment_name:
        click.echo(f"Experiment Name: {experiment_name}")

    click.echo("\nRun 'qwen' in this directory to start tracing.")


def _show_status(target_dir: Path, settings_file: Path) -> None:
    click.echo(f"Qwen Code tracing status in: {target_dir}")

    tracing_status = get_tracing_status(settings_file)

    if not tracing_status.enabled:
        click.echo("Qwen Code tracing is not enabled")
        if tracing_status.reason:
            click.echo(f"  Reason: {tracing_status.reason}")
        return

    click.echo("Qwen Code tracing is ENABLED")
    if tracing_status.tracking_uri:
        click.echo(f"Tracking URI: {tracing_status.tracking_uri}")


@qwen_code.command("stop-hook", hidden=True)
def stop_hook() -> None:
    """Hook handler invoked when a Qwen Code conversation ends."""
    stop_hook_handler()
