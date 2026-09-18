"""Click CLI commands for the Kiro CLI autolog integration."""

import sys
from pathlib import Path

import click

from mlflow.autolog import autolog as commands
from mlflow.kiro_cli.config import get_tracing_status, setup_environment_config
from mlflow.kiro_cli.hooks import (
    agent_spawn_hook_handler,
    disable_tracing_hooks,
    post_tool_use_hook_handler,
    pre_tool_use_hook_handler,
    setup_hooks_config,
    stop_hook_handler,
    user_prompt_submit_hook_handler,
)


@commands.group("kiro-cli", invoke_without_command=True)
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
@click.option("--disable", is_flag=True, help="Disable Kiro CLI tracing in the specified directory")
@click.option("--status", is_flag=True, help="Show current tracing status")
@click.pass_context
def kiro_cli(
    ctx: click.Context,
    directory: str,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
    disable: bool,
    status: bool,
) -> None:
    """Set up Kiro CLI tracing in a directory.

    This command configures Kiro CLI hooks in ``.kiro/agents/kiro_default.json``
    so conversations are automatically traced to MLflow. After setup, use the
    regular ``kiro`` command and traces will be created automatically.

    Examples:

      # Set up tracing in current directory with local storage
      mlflow autolog kiro-cli

      # Set up tracing in a specific project directory
      mlflow autolog kiro-cli -d ~/my-project

      # Set up tracing with Databricks
      mlflow autolog kiro-cli -u databricks -e 123456789

      # Set up tracing with a custom tracking URI
      mlflow autolog kiro-cli -u file://./custom-mlruns

      # Disable tracing in current directory
      mlflow autolog kiro-cli --disable
    """
    # Skip setup when a subcommand (e.g., stop-hook) is being invoked
    if ctx.invoked_subcommand is not None:
        return

    target_dir = Path(directory).resolve()
    agent_config_path = target_dir / ".kiro/agents/kiro_default.json"
    settings_path = target_dir / ".kiro/settings.json"

    # --status wins over --disable when both are set (no disk mutation)
    if status:
        _show_status(target_dir, settings_path)
        return

    if disable:
        _handle_disable(agent_config_path, settings_path)
        return

    # -e wins over -n when both are set
    if experiment_id and experiment_name:
        experiment_name = None

    click.echo(f"Configuring Kiro CLI tracing in: {target_dir}")

    # Create .kiro/agents/ directory and set up hooks
    try:
        agent_config_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        click.echo(f"❌ Permission denied creating directory: {agent_config_path.parent}")
        sys.exit(1)

    setup_hooks_config(agent_config_path)
    click.echo("✅ Kiro CLI hooks configured")

    # Set up environment variables
    try:
        setup_environment_config(settings_path, tracking_uri, experiment_id, experiment_name)
    except PermissionError:
        click.echo(f"❌ Permission denied writing to: {settings_path}")
        sys.exit(1)

    # Show final status
    _show_setup_status(target_dir, tracking_uri, experiment_id, experiment_name)


def _handle_disable(agent_config_path: Path, settings_path: Path) -> None:
    """Handle disable command."""
    if disable_tracing_hooks(agent_config_path, settings_path):
        click.echo("✅ Kiro CLI tracing disabled")
    else:
        click.echo("❌ No Kiro CLI configuration found - tracing was not enabled")


def _show_status(target_dir: Path, settings_path: Path) -> None:
    """Show current tracing status."""
    click.echo(f"📍 Kiro CLI tracing status in: {target_dir}")

    status = get_tracing_status(settings_path)

    if not status.enabled:
        click.echo("❌ Kiro CLI tracing is not enabled")
        if status.reason:
            click.echo(f"   Reason: {status.reason}")
        return

    click.echo("✅ Kiro CLI tracing is ENABLED")
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
    click.echo("🎯 Kiro CLI Tracing Setup Complete!")
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

    click.echo('kiro chat "your prompt here"')

    if tracking_uri == "databricks":
        click.echo("\n💡 View your traces in your Databricks workspace")
    elif tracking_uri:
        click.echo("\n💡 View your traces:")
        click.echo(f"   mlflow server --backend-store-uri {tracking_uri}")
    else:
        click.echo("\n💡 View your traces:")
        click.echo("   mlflow server")

    click.echo("\n🔧 To disable tracing later:")
    click.echo("   mlflow autolog kiro-cli --disable")


# ============================================================================
# HIDDEN SUBCOMMANDS (invoked by Kiro CLI hooks, not by users)
# ============================================================================


@kiro_cli.command("stop-hook", hidden=True)
def stop_hook() -> None:
    """Hook handler invoked when a Kiro CLI conversation ends."""
    stop_hook_handler()


@kiro_cli.command("agent-spawn-hook", hidden=True)
def agent_spawn_hook() -> None:
    """Hook handler invoked when a Kiro CLI agent spawns."""
    agent_spawn_hook_handler()


@kiro_cli.command("user-prompt-submit-hook", hidden=True)
def user_prompt_submit_hook() -> None:
    """Hook handler invoked when a user submits a prompt."""
    user_prompt_submit_hook_handler()


@kiro_cli.command("pre-tool-use-hook", hidden=True)
def pre_tool_use_hook() -> None:
    """Hook handler invoked before a tool is used."""
    pre_tool_use_hook_handler()


@kiro_cli.command("post-tool-use-hook", hidden=True)
def post_tool_use_hook() -> None:
    """Hook handler invoked after a tool is used."""
    post_tool_use_hook_handler()
