"""MLflow CLI commands for Claude Code integration."""

from pathlib import Path

import click

from mlflow.claude_code.config import get_tracing_status, setup_environment_config
from mlflow.claude_code.hooks import disable_tracing_hooks, setup_hooks_config


@click.group("autolog")
def commands():
    """Commands for autologging with MLflow."""


@commands.command("claude")
@click.argument("directory", default=".", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "--tracking-uri", "-u", help="MLflow tracking URI (e.g., 'databricks' or 'file://mlruns')"
)
@click.option("--experiment-id", "-e", help="MLflow experiment ID")
@click.option("--experiment-name", "-n", help="MLflow experiment name")
@click.option("--disable", is_flag=True, help="Disable Claude tracing in the specified directory")
@click.option("--status", is_flag=True, help="Show current tracing status")
def claude(
    directory: str,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
    disable: bool,
    status: bool,
) -> None:
    """Set up Claude Code tracing in a directory.

    This command configures Claude Code hooks to automatically trace conversations
    to MLflow. After setup, use the regular 'claude' command and traces will be
    automatically created.

    DIRECTORY: Directory to set up tracing in (default: current directory)

    Examples:

      # Set up tracing in current directory with local storage
      mlflow autolog claude

      # Set up tracing in a specific project directory
      mlflow autolog claude ~/my-project

      # Set up tracing with Databricks
      mlflow autolog claude -u databricks -e 123456789

      # Set up tracing with custom tracking URI
      mlflow autolog claude -u file://./custom-mlruns

      # Disable tracing in current directory
      mlflow autolog claude --disable
    """
    target_dir = Path(directory).resolve()
    claude_dir = target_dir / ".claude"
    settings_file = claude_dir / "settings.json"

    if status:
        _show_status(target_dir, settings_file)
        return

    if disable:
        _handle_disable(settings_file)
        return

    click.echo(f"Configuring Claude tracing in: {target_dir}")

    # Create .claude directory and set up hooks
    claude_dir.mkdir(parents=True, exist_ok=True)
    setup_hooks_config(settings_file)
    click.echo("✅ Claude Code hooks configured")

    # Set up environment variables
    setup_environment_config(settings_file, tracking_uri, experiment_id, experiment_name)

    # Show final status
    _show_setup_status(target_dir, tracking_uri, experiment_id, experiment_name)


def _handle_disable(settings_file: Path) -> None:
    """Handle disable command."""
    if disable_tracing_hooks(settings_file):
        click.echo("✅ Claude tracing disabled")
    else:
        click.echo("❌ No Claude configuration found - tracing was not enabled")


def _show_status(target_dir: Path, settings_file: Path) -> None:
    """Show current tracing status."""
    click.echo(f"📍 Claude tracing status in: {target_dir}")

    status = get_tracing_status(settings_file)

    if not status.enabled:
        click.echo("❌ Claude tracing is not enabled")
        if status.reason:
            click.echo(f"   Reason: {status.reason}")
        return

    click.echo("✅ Claude tracing is ENABLED")
    click.echo(f"📊 Tracking URI: {status.tracking_uri}")

    if status.experiment_id:
        click.echo(f"🔬 Experiment ID: {status.experiment_id}")
    elif status.experiment_name:
        click.echo(f"🔬 Experiment Name: {status.experiment_name}")
    else:
        click.echo("🔬 Experiment: Default (experiment 0)")


def _show_setup_status(
    target_dir: Path,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
) -> None:
    """Show setup completion status."""
    current_dir = Path.cwd().resolve()

    click.echo("\n" + "=" * 50)
    click.echo("🎯 Claude Tracing Setup Complete!")
    click.echo("=" * 50)

    click.echo(f"📁 Directory: {target_dir}")

    # Show tracking configuration
    if tracking_uri:
        click.echo(f"📊 Tracking URI: {tracking_uri}")

    if experiment_id:
        click.echo(f"🔬 Experiment ID: {experiment_id}")
    elif experiment_name:
        click.echo(f"🔬 Experiment Name: {experiment_name}")
    else:
        click.echo("🔬 Experiment: Default (experiment 0)")

    # Show next steps
    click.echo("\n" + "=" * 30)
    click.echo("🚀 Next Steps:")
    click.echo("=" * 30)

    # Only show cd if it's a different directory
    if target_dir != current_dir:
        click.echo(f"cd {target_dir}")

    click.echo("claude -p 'your prompt here'")

    if tracking_uri and tracking_uri.startswith("file://"):
        click.echo("\n💡 View your traces:")
        click.echo(f"   mlflow ui --backend-store-uri {tracking_uri}")
    elif not tracking_uri:
        click.echo("\n💡 View your traces:")
        click.echo("   mlflow ui")
    elif tracking_uri == "databricks":
        click.echo("\n💡 View your traces in your Databricks workspace")

    click.echo("\n🔧 To disable tracing later:")
    click.echo("   mlflow autolog claude --disable")
