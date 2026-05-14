"""MLflow CLI commands for Claude Code integration."""

from pathlib import Path

import click

from mlflow.claude_code.config import get_tracing_status, setup_environment_config
from mlflow.claude_code.hooks import stop_hook_handler
from mlflow.claude_code.plugin import (
    disable_tracing_plugin,
    ensure_plugin_installed,
    migrate_legacy_hooks,
)


@click.group("autolog")
def commands():
    """Commands for autologging with MLflow."""


@commands.group("claude", invoke_without_command=True)
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
@click.option("--disable", is_flag=True, help="Disable Claude tracing in the specified directory")
@click.option("--status", is_flag=True, help="Show current tracing status")
@click.option(
    "--mlflow-cmd",
    default=None,
    help=(
        "Deprecated and ignored. Python-based Claude hooks were replaced by the "
        "marketplace plugin runtime."
    ),
)
@click.pass_context
def claude(
    ctx: click.Context,
    directory: str,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
    disable: bool,
    status: bool,
    mlflow_cmd: str | None,
) -> None:
    """Set up Claude Code tracing in a directory.

    This command installs the MLflow Claude plugin into Claude Code and writes
    MLflow configuration into `.claude/settings.json`. After setup, use the
    regular `claude` command and traces will be created by the plugin runtime.

    Examples:

      # Set up tracing in current directory with local storage
      mlflow autolog claude

      # Set up tracing in a specific project directory
      mlflow autolog claude -d ~/my-project

      # Set up tracing with Databricks
      mlflow autolog claude -u databricks -e 123456789

      # Set up tracing with custom tracking URI
      mlflow autolog claude -u file://./custom-mlruns

      # Disable tracing in current directory
      mlflow autolog claude --disable
    """
    # Skip setup when a subcommand (e.g., stop-hook) is being invoked
    if ctx.invoked_subcommand is not None:
        return

    if mlflow_cmd is not None:
        if not mlflow_cmd.strip():
            raise click.BadParameter(
                "must not be empty or whitespace-only", param_hint="'--mlflow-cmd'"
            )
        click.echo("⚠️  --mlflow-cmd is deprecated and ignored.")

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

    # Create .claude directory and install the plugin runtime
    claude_dir.mkdir(parents=True, exist_ok=True)
    try:
        ensure_plugin_installed(target_dir)
        migrate_legacy_hooks(settings_file)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Failed to configure Claude tracing: {exc}") from exc
    click.echo("✅ Claude Code plugin installed")

    # Set up environment variables consumed by the plugin
    setup_environment_config(settings_file, tracking_uri, experiment_id, experiment_name)

    # Show final status
    _show_setup_status(target_dir, settings_file)


def _handle_disable(settings_file: Path) -> None:
    """Handle disable command."""
    if disable_tracing_plugin(settings_file):
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

    if status.experiment_name:
        click.echo(f"🔬 Experiment Name: {status.experiment_name}")
    if status.experiment_id:
        click.echo(f"🆔 Experiment ID: {status.experiment_id}")
    elif not status.experiment_name:
        click.echo("🔬 Experiment: Default (experiment 0)")


def _show_setup_status(
    target_dir: Path,
    settings_file: Path,
) -> None:
    """Show setup completion status."""
    current_dir = Path.cwd().resolve()
    status = get_tracing_status(settings_file)

    click.echo("\n" + "=" * 50)
    click.echo("🎯 Claude Tracing Setup Complete!")
    click.echo("=" * 50)

    click.echo(f"📁 Directory: {target_dir}")

    # Show tracking configuration
    if status.tracking_uri:
        click.echo(f"📊 Tracking URI: {status.tracking_uri}")

    if status.experiment_name:
        click.echo(f"🔬 Experiment Name: {status.experiment_name}")
    if status.experiment_id:
        click.echo(f"🆔 Experiment ID: {status.experiment_id}")
    elif not status.experiment_name:
        click.echo("🔬 Experiment: Default (experiment 0)")

    # Show next steps
    click.echo("\n" + "=" * 30)
    click.echo("🚀 Next Steps:")
    click.echo("=" * 30)

    # Only show cd if it's a different directory
    if target_dir != current_dir:
        click.echo(f"cd {target_dir}")

    click.echo("claude -p 'your prompt here'")

    if status.tracking_uri and status.tracking_uri.startswith("file://"):
        click.echo("\n💡 View your traces:")
        click.echo(f"   mlflow server --backend-store-uri {status.tracking_uri}")
    elif not status.tracking_uri:
        click.echo("\n💡 View your traces:")
        click.echo("   mlflow server")
    elif status.tracking_uri == "databricks":
        click.echo("\n💡 View your traces in your Databricks workspace")

    click.echo("\n🔧 To disable tracing later:")
    click.echo("   mlflow autolog claude --disable")


@claude.command("stop-hook", hidden=True)
def stop_hook() -> None:
    """Legacy hook shim kept for older Python-hook installations."""
    stop_hook_handler()
