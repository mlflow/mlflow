"""MLflow CLI commands for Opencode integration."""

from pathlib import Path

import click

from mlflow.opencode.config import (
    MLFLOW_PLUGIN_NPM_PACKAGE,
    disable_tracing,
    get_opencode_config_path,
    get_tracing_status,
    setup_hook_config,
)


@click.group("autolog")
def commands():
    """Commands for autologging with MLflow."""


@commands.command("opencode")
@click.argument("directory", default=".", type=click.Path(file_okay=False, dir_okay=True))
@click.option(
    "--tracking-uri",
    "-u",
    help="MLflow tracking URI (e.g., 'sqlite:///mlflow.db' or 'http://localhost:5000')",
)
@click.option("--experiment-id", "-e", help="MLflow experiment ID")
@click.option("--experiment-name", "-n", help="MLflow experiment name")
@click.option("--disable", is_flag=True, help="Disable Opencode tracing in the specified directory")
@click.option("--status", is_flag=True, help="Show current tracing status")
def opencode(
    directory: str,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
    disable: bool,
    status: bool,
) -> None:
    """Set up Opencode tracing in a directory.

    This command configures Opencode to automatically trace conversations
    to MLflow. After setup, use Opencode normally and traces will be
    automatically created.

    DIRECTORY: Directory to set up tracing in (default: current directory)

    Examples:

      # Set up tracing with SQLite backend (recommended)
      mlflow autolog opencode -u sqlite:///mlflow.db

      # Set up tracing in a specific project directory
      mlflow autolog opencode ~/my-project -u sqlite:///mlflow.db

      # Set up tracing with remote MLflow server
      mlflow autolog opencode -u http://localhost:5000

      # Set up tracing with Databricks
      mlflow autolog opencode -u databricks -e 123456789

      # Disable tracing in current directory
      mlflow autolog opencode --disable
    """
    target_dir = Path(directory).resolve()
    config_path = get_opencode_config_path(target_dir)

    if status:
        _show_status(target_dir, config_path)
        return

    if disable:
        _handle_disable(config_path)
        return

    click.echo(f"Configuring Opencode tracing in: {target_dir}")

    # Set up plugin configuration
    plugin_name = setup_hook_config(config_path, tracking_uri, experiment_id, experiment_name)
    click.echo(f"✅ Opencode plugin configured: {plugin_name}")

    # Show final status
    _show_setup_status(target_dir, tracking_uri, experiment_id, experiment_name)


def _handle_disable(config_path: Path) -> None:
    if disable_tracing(config_path):
        click.echo("✅ Opencode tracing disabled")
    else:
        click.echo("❌ No Opencode configuration found - tracing was not enabled")


def _show_status(target_dir: Path, config_path: Path) -> None:
    click.echo(f"📍 Opencode tracing status in: {target_dir}")

    status = get_tracing_status(config_path)

    if not status.enabled:
        click.echo("❌ Opencode tracing is not enabled")
        if status.reason:
            click.echo(f"   Reason: {status.reason}")
        return

    click.echo("✅ Opencode tracing is ENABLED")
    if status.tracking_uri:
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
    click.echo("\n" + "=" * 50)
    click.echo("🎯 Opencode Tracing Setup Complete!")
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

    # Step 1: Install the npm package
    click.echo("\n1️⃣  Install the MLflow plugin (if not already installed):")
    click.echo(f"    bun add {MLFLOW_PLUGIN_NPM_PACKAGE}")
    click.echo(f"    # or: npm install {MLFLOW_PLUGIN_NPM_PACKAGE}")

    # Step 2: Run opencode
    click.echo("\n2️⃣  Run Opencode (from the project directory):")
    click.echo("    opencode 'your prompt here'")

    # View traces
    if tracking_uri and tracking_uri.startswith("sqlite:"):
        click.echo("\n💡 View your traces:")
        click.echo(f"   mlflow server --backend-store-uri {tracking_uri}")
        click.echo("   # Then open http://localhost:5000")
    elif tracking_uri and tracking_uri.startswith("http"):
        click.echo("\n💡 View your traces:")
        click.echo(f"   Open {tracking_uri} in your browser")
    elif tracking_uri == "databricks":
        click.echo("\n💡 View your traces in your Databricks workspace")
    elif not tracking_uri:
        click.echo("\n💡 View your traces:")
        click.echo("   mlflow server")

    click.echo("\n🔧 To disable tracing later:")
    click.echo("   mlflow autolog opencode --disable")
