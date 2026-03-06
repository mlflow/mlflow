"""MLflow CLI commands for Gemini CLI integration."""

from pathlib import Path

import click

from mlflow.gemini_cli.config import get_tracing_status, setup_environment_config
from mlflow.gemini_cli.hooks import disable_tracing_hooks, setup_hooks_config


@click.command("gemini-cli")
@click.argument("directory", default=".", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "--tracking-uri", "-u", help="MLflow tracking URI (e.g., 'databricks' or 'file://mlruns')"
)
@click.option("--experiment-id", "-e", help="MLflow experiment ID")
@click.option("--experiment-name", "-n", help="MLflow experiment name")
@click.option("--disable", is_flag=True, help="Disable Gemini CLI tracing in the specified directory")
@click.option("--status", is_flag=True, help="Show current tracing status")
def gemini_cli(
    directory: str,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
    disable: bool,
    status: bool,
) -> None:
    """Set up Gemini CLI tracing in a directory.

    This command configures Gemini CLI hooks to automatically trace conversations
    to MLflow. After setup, use the regular 'gemini' command and traces will be
    automatically created.

    DIRECTORY: Directory to set up tracing in (default: current directory)

    Examples:

      # Set up tracing in current directory with local storage
      mlflow autolog gemini-cli

      # Set up tracing in a specific project directory
      mlflow autolog gemini-cli ~/my-project

      # Set up tracing with Databricks
      mlflow autolog gemini-cli -u databricks -e 123456789

      # Set up tracing with custom tracking URI
      mlflow autolog gemini-cli -u file://./custom-mlruns

      # Disable tracing in current directory
      mlflow autolog gemini-cli --disable
    """
    target_dir = Path(directory).resolve()
    gemini_dir = target_dir / ".gemini"
    settings_file = gemini_dir / "settings.json"

    if status:
        _show_status(target_dir, settings_file)
        return

    if disable:
        _handle_disable(settings_file)
        return

    click.echo(f"Configuring Gemini CLI tracing in: {target_dir}")

    # Create .gemini directory and set up hooks
    gemini_dir.mkdir(parents=True, exist_ok=True)
    setup_hooks_config(settings_file)
    click.echo("Gemini CLI hooks configured")

    # Build environment variables for user reference
    env_vars = setup_environment_config(tracking_uri, experiment_id, experiment_name)

    # Show final status
    _show_setup_status(target_dir, tracking_uri, experiment_id, experiment_name, env_vars)


def _handle_disable(settings_file: Path) -> None:
    """Handle disable command."""
    if disable_tracing_hooks(settings_file):
        click.echo("Gemini CLI tracing disabled")
    else:
        click.echo("No Gemini CLI configuration found - tracing was not enabled")


def _show_status(target_dir: Path, settings_file: Path) -> None:
    """Show current tracing status."""
    click.echo(f"Gemini CLI tracing status in: {target_dir}")

    tracing_status = get_tracing_status(settings_file)

    if not tracing_status.enabled:
        click.echo("Gemini CLI tracing is not enabled")
        if tracing_status.reason:
            click.echo(f"   Reason: {tracing_status.reason}")
        return

    click.echo("Gemini CLI tracing is ENABLED")
    if tracing_status.tracking_uri:
        click.echo(f"Tracking URI: {tracing_status.tracking_uri}")

    if tracing_status.experiment_id:
        click.echo(f"Experiment ID: {tracing_status.experiment_id}")
    elif tracing_status.experiment_name:
        click.echo(f"Experiment Name: {tracing_status.experiment_name}")
    else:
        click.echo("Experiment: Default (experiment 0)")


def _show_setup_status(
    target_dir: Path,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
    env_vars: dict[str, str],
) -> None:
    """Show setup completion status."""
    current_dir = Path.cwd().resolve()

    click.echo("\n" + "=" * 50)
    click.echo("Gemini CLI Tracing Setup Complete!")
    click.echo("=" * 50)

    click.echo(f"Directory: {target_dir}")

    # Show tracking configuration
    if tracking_uri:
        click.echo(f"Tracking URI: {tracking_uri}")

    if experiment_id:
        click.echo(f"Experiment ID: {experiment_id}")
    elif experiment_name:
        click.echo(f"Experiment Name: {experiment_name}")
    else:
        click.echo("Experiment: Default (experiment 0)")

    # Show required environment variables
    click.echo("\n" + "=" * 30)
    click.echo("Required Environment Variables:")
    click.echo("=" * 30)
    click.echo("Set these in your shell before running Gemini CLI:")
    for var_name, var_value in env_vars.items():
        click.echo(f"  export {var_name}={var_value}")

    # Show next steps
    click.echo("\n" + "=" * 30)
    click.echo("Next Steps:")
    click.echo("=" * 30)

    # Only show cd if it's a different directory
    if target_dir != current_dir:
        click.echo(f"cd {target_dir}")

    click.echo("gemini 'your prompt here'")

    if tracking_uri and tracking_uri.startswith("file://"):
        click.echo(f"\nView your traces:")
        click.echo(f"   mlflow server --backend-store-uri {tracking_uri}")
    elif not tracking_uri:
        click.echo(f"\nView your traces:")
        click.echo("   mlflow server")
    elif tracking_uri == "databricks":
        click.echo("\nView your traces in your Databricks workspace")

    click.echo("\nTo disable tracing later:")
    click.echo("   mlflow autolog gemini-cli --disable")
