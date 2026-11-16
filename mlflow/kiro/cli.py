"""MLflow CLI commands for the Kiro CLI integration."""

from pathlib import Path

import click

from mlflow.kiro.config import (
    KIRO_ENV_FILE,
    KIRO_HOOKS_DIR,
    disable_tracing_hooks,
    get_tracing_status,
    setup_environment_config,
    setup_hooks_config,
)
from mlflow.kiro.hooks import stop_hook_handler


# Re-use the existing ``autolog`` group created by the claude_code integration
# (they share the same click group: ``mlflow autolog``).
@click.group("autolog")
def commands() -> None:
    """Commands for autologging with MLflow."""


@commands.group("kiro", invoke_without_command=True)
@click.option(
    "--directory",
    "-d",
    default=".",
    type=click.Path(file_okay=False, dir_okay=True),
    help="Project directory to configure (default: current directory)",
)
@click.option("--tracking-uri", "-u", help="MLflow tracking URI (e.g. 'databricks' or 'file://mlruns')")
@click.option("--experiment-id", "-e", help="MLflow experiment ID")
@click.option("--experiment-name", "-n", help="MLflow experiment name")
@click.option("--disable", is_flag=True, help="Remove Kiro tracing from the specified directory")
@click.option("--status", is_flag=True, help="Show current tracing status")
@click.pass_context
def kiro(
    ctx: click.Context,
    directory: str,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
    disable: bool,
    status: bool,
) -> None:
    """Set up Kiro CLI tracing in a project directory.

    This command writes a Kiro Agent Stop hook that automatically captures
    every Kiro session as an MLflow trace.

    \b
    Examples:

      # Set up tracing in the current directory (local storage)
      mlflow autolog kiro

      # Set up tracing in a specific project directory
      mlflow autolog kiro -d ~/my-project

      # Set up tracing with Databricks
      mlflow autolog kiro -u databricks -e 123456789

      # Set up tracing with a custom tracking URI
      mlflow autolog kiro -u sqlite:///mlflow.db -n "Kiro Sessions"

      # Disable tracing
      mlflow autolog kiro --disable

      # Check status
      mlflow autolog kiro --status
    """
    if ctx.invoked_subcommand is not None:
        return

    target_dir = Path(directory).resolve()
    hooks_dir = target_dir / KIRO_HOOKS_DIR
    env_path = target_dir / KIRO_ENV_FILE

    if status:
        _show_status(target_dir, hooks_dir, env_path)
        return

    if disable:
        _handle_disable(hooks_dir, env_path)
        return

    click.echo(f"Configuring Kiro tracing in: {target_dir}")

    # Write hook file
    setup_hooks_config(hooks_dir)
    click.echo("✅ Kiro Agent Stop hook configured")

    # Write env config
    setup_environment_config(env_path, tracking_uri, experiment_id, experiment_name)
    click.echo("✅ MLflow environment configured")

    _show_setup_summary(target_dir, tracking_uri, experiment_id, experiment_name)


# ---------------------------------------------------------------------------
# Sub-handlers
# ---------------------------------------------------------------------------


def _handle_disable(hooks_dir: Path, env_path: Path) -> None:
    if disable_tracing_hooks(hooks_dir, env_path):
        click.echo("✅ Kiro tracing disabled")
    else:
        click.echo("❌ No Kiro tracing configuration found — nothing to disable")


def _show_status(target_dir: Path, hooks_dir: Path, env_path: Path) -> None:
    click.echo(f"📍 Kiro tracing status in: {target_dir}")
    st = get_tracing_status(hooks_dir, env_path)
    if not st.enabled:
        click.echo("❌ Kiro tracing is not enabled")
        if st.reason:
            click.echo(f"   Reason: {st.reason}")
        return

    click.echo("✅ Kiro tracing is ENABLED")
    click.echo(f"📊 Tracking URI: {st.tracking_uri or '(default)'}")
    if st.experiment_id:
        click.echo(f"🔬 Experiment ID: {st.experiment_id}")
    elif st.experiment_name:
        click.echo(f"🔬 Experiment Name: {st.experiment_name}")
    else:
        click.echo("🔬 Experiment: Default (experiment 0)")


def _show_setup_summary(
    target_dir: Path,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
) -> None:
    current_dir = Path.cwd().resolve()

    click.echo("\n" + "=" * 50)
    click.echo("🎯 Kiro Tracing Setup Complete!")
    click.echo("=" * 50)
    click.echo(f"📁 Directory: {target_dir}")

    if tracking_uri:
        click.echo(f"📊 Tracking URI: {tracking_uri}")
    if experiment_id:
        click.echo(f"🔬 Experiment ID: {experiment_id}")
    elif experiment_name:
        click.echo(f"🔬 Experiment Name: {experiment_name}")
    else:
        click.echo("🔬 Experiment: Default (experiment 0)")

    click.echo("\n" + "=" * 30)
    click.echo("🚀 Next Steps:")
    click.echo("=" * 30)

    if target_dir != current_dir:
        click.echo(f"cd {target_dir}")

    click.echo("# Then use Kiro as normal — traces are captured automatically")

    if tracking_uri and tracking_uri.startswith("file://"):
        click.echo("\n💡 View your traces:")
        click.echo(f"   mlflow server --backend-store-uri {tracking_uri}")
    elif not tracking_uri:
        click.echo("\n💡 View your traces:")
        click.echo("   mlflow server")
    elif tracking_uri == "databricks":
        click.echo("\n💡 View your traces in your Databricks workspace")

    click.echo("\n🔧 To disable tracing later:")
    click.echo("   mlflow autolog kiro --disable")


# ---------------------------------------------------------------------------
# Hidden stop-hook subcommand (invoked by Kiro itself)
# ---------------------------------------------------------------------------


@kiro.command("stop-hook", hidden=True)
def stop_hook() -> None:
    """Hook handler invoked when a Kiro agent session ends."""
    stop_hook_handler()
