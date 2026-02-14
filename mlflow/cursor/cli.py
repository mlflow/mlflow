"""MLflow CLI commands for Cursor integration."""

from pathlib import Path

import click

from mlflow.cursor.config import (
    CURSOR_DIR,
    get_cursor_hooks_path,
    get_tracing_status,
    setup_environment_config,
)
from mlflow.cursor.hooks import disable_tracing_hooks, setup_hooks_config


@click.group("autolog")
def commands():
    """Commands for autologging with MLflow."""


@commands.command("cursor")
@click.argument("directory", default=".", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "--tracking-uri",
    "-u",
    help="MLflow tracking URI (e.g., 'databricks' or 'http://localhost:5000')",
)
@click.option("--experiment-id", "-e", help="MLflow experiment ID")
@click.option("--experiment-name", "-n", help="MLflow experiment name")
@click.option("--disable", is_flag=True, help="Disable Cursor tracing in the specified directory")
@click.option("--status", is_flag=True, help="Show current tracing status")
def cursor(
    directory: str,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
    disable: bool,
    status: bool,
) -> None:
    """Set up Cursor tracing in a directory.

    This command configures Cursor hooks to automatically trace AI agent interactions
    to MLflow. After setup, your Cursor agent conversations will be automatically
    traced and logged to the configured MLflow experiment.

    DIRECTORY: Directory to set up tracing in (default: current directory)

    Examples:

      # Set up tracing in current directory with local storage
      mlflow autolog cursor

      # Set up tracing in a specific project directory
      mlflow autolog cursor ~/my-project

      # Set up tracing with Databricks
      mlflow autolog cursor -u databricks -e 123456789

      # Set up tracing with a remote MLflow server
      mlflow autolog cursor -u http://localhost:5000

      # Check tracing status
      mlflow autolog cursor --status

      # Disable tracing in current directory
      mlflow autolog cursor --disable
    """
    target_dir = Path(directory).resolve()
    cursor_dir = target_dir / CURSOR_DIR
    hooks_file = get_cursor_hooks_path(target_dir)
    env_file = cursor_dir / ".env"

    if status:
        _show_status(target_dir)
        return

    if disable:
        _handle_disable(hooks_file, env_file)
        return

    click.echo(f"Configuring Cursor tracing in: {target_dir}")

    # Create .cursor directory and set up hooks
    cursor_dir.mkdir(parents=True, exist_ok=True)
    setup_hooks_config(hooks_file)
    click.echo("Cursor hooks configured")

    # Set up environment variables
    setup_environment_config(target_dir, tracking_uri, experiment_id, experiment_name)
    click.echo("MLflow environment configured")

    # Show final status
    _show_setup_status(target_dir, tracking_uri, experiment_id, experiment_name)


def _handle_disable(hooks_file: Path, env_file: Path) -> None:
    """Handle disable command."""
    if disable_tracing_hooks(hooks_file, env_file):
        click.echo("Cursor tracing disabled")
    else:
        click.echo("No Cursor configuration found - tracing was not enabled")


def _show_status(target_dir: Path) -> None:
    """Show current tracing status."""
    click.echo(f"Cursor tracing status in: {target_dir}")

    status = get_tracing_status(target_dir)

    if not status.enabled:
        click.echo("Cursor tracing is not enabled")
        if status.reason:
            click.echo(f"  Reason: {status.reason}")
        return

    click.echo("Cursor tracing is ENABLED")
    click.echo(f"  Tracking URI: {status.tracking_uri or 'default'}")

    if status.experiment_id:
        click.echo(f"  Experiment ID: {status.experiment_id}")
    elif status.experiment_name:
        click.echo(f"  Experiment Name: {status.experiment_name}")
    else:
        click.echo("  Experiment: Default (experiment 0)")


def _show_setup_status(
    target_dir: Path,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
) -> None:
    """Show setup completion status."""
    current_dir = Path.cwd().resolve()

    click.echo("\n" + "=" * 50)
    click.echo("Cursor Tracing Setup Complete!")
    click.echo("=" * 50)

    click.echo(f"Directory: {target_dir}")

    # Show tracking configuration
    if tracking_uri:
        click.echo(f"Tracking URI: {tracking_uri}")
    else:
        click.echo("Tracking URI: default")

    if experiment_id:
        click.echo(f"Experiment ID: {experiment_id}")
    elif experiment_name:
        click.echo(f"Experiment Name: {experiment_name}")
    else:
        click.echo("Experiment: Default (experiment 0)")

    # Show next steps
    click.echo("\n" + "=" * 30)
    click.echo("Next Steps:")
    click.echo("=" * 30)

    if target_dir != current_dir:
        click.echo(f"1. cd {target_dir}")
        click.echo("2. Open the directory in Cursor")
        click.echo("3. Start chatting with Cursor Agent - traces will be captured automatically")
    else:
        click.echo("1. Open this directory in Cursor")
        click.echo("2. Start chatting with Cursor Agent - traces will be captured automatically")

    click.echo("\nView your traces:")
    if tracking_uri == "databricks":
        click.echo("  Open your Databricks workspace")
    elif tracking_uri and tracking_uri.startswith(("http://", "https://")):
        click.echo(f"  Open {tracking_uri} in your browser")
    elif tracking_uri and tracking_uri.startswith("sqlite://"):
        click.echo(f"  mlflow server --backend-store-uri {tracking_uri}")
    else:
        click.echo("  mlflow server")

    click.echo("\nTo disable tracing later:")
    click.echo("  mlflow autolog cursor --disable")
