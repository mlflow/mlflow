"""MLflow CLI commands for Claude Code integration."""

import os
import sys
from pathlib import Path

import click

from mlflow.claude_code.config import get_tracing_status, setup_environment_config
from mlflow.claude_code.hooks import stop_hook_handler
from mlflow.claude_code.plugin import (
    disable_tracing_plugin,
    ensure_plugin_installed,
)
from mlflow.environment_variables import (
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)


def _title(text: str) -> str:
    return click.style(text, fg="magenta", bold=True)


def _ok(text: str) -> str:
    return click.style(text, fg="green", bold=True)


def _warn(text: str) -> str:
    return click.style(text, fg="yellow", bold=True)


def _error(text: str) -> str:
    return click.style(text, fg="red", bold=True)


def _label(text: str) -> str:
    return click.style(text, bold=True)


def _question(text: str) -> str:
    return click.style(text, fg="yellow", bold=True)


def _value(text: str) -> str:
    return click.style(text, fg="cyan")


def _muted(text: str) -> str:
    return click.style(text, dim=True)


_DEFAULT_TRACKING_URI_SENTINEL = "default"


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
    "--non-interactive",
    "-y",
    is_flag=True,
    help="Skip prompts and use flags, environment variables, or defaults.",
)
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
    non_interactive: bool,
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

    if experiment_id and experiment_name:
        raise click.BadParameter("Choose either --experiment-id or --experiment-name, not both.")

    if mlflow_cmd is not None:
        if not mlflow_cmd.strip():
            raise click.BadParameter(
                "must not be empty or whitespace-only", param_hint="'--mlflow-cmd'"
            )
        click.echo(f"{_warn('⚠')} {_muted('--mlflow-cmd is deprecated and ignored.')}")

    target_dir = Path(directory).resolve()
    claude_dir = target_dir / ".claude"
    settings_file = claude_dir / "settings.json"

    if status:
        _show_status(target_dir, settings_file)
        return

    if disable:
        _handle_disable(settings_file)
        return

    _print_setup_intro(tracking_uri, experiment_id, experiment_name, non_interactive)
    tracking_uri, experiment_id, experiment_name = _resolve_setup_inputs(
        tracking_uri,
        experiment_id,
        experiment_name,
        non_interactive,
    )

    click.echo(f"{_title('MLflow Claude Tracing Setup')}")
    click.echo(f"{_label('Project:')} {_value(str(target_dir))}")

    # Create .claude directory and install the plugin runtime
    claude_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"{_label('Installing plugin:')} {_muted('MLflow Claude plugin for Claude Code')}")
    try:
        ensure_plugin_installed(target_dir)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Failed to configure Claude tracing: {exc}") from exc
    click.echo(f"{_ok('✓')} Claude Code plugin installed")

    # Set up environment variables consumed by the plugin
    setup_environment_config(settings_file, tracking_uri, experiment_id, experiment_name)

    # Show final status
    _show_setup_status(target_dir, settings_file)


def _print_setup_intro(
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
    non_interactive: bool,
) -> None:
    if non_interactive or not _is_interactive_shell():
        return

    missing_tracking = not (tracking_uri or os.environ.get(MLFLOW_TRACKING_URI.name))
    missing_experiment = not (
        experiment_id
        or experiment_name
        or os.environ.get(MLFLOW_EXPERIMENT_ID.name)
        or os.environ.get(MLFLOW_EXPERIMENT_NAME.name)
    )
    if not missing_tracking and not missing_experiment:
        return

    click.echo(f"{_title('Interactive Mode')}")
    click.echo(_muted("MLflow Claude tracing setup is running in interactive mode."))
    click.echo(
        _muted(
            "If you want non-interactive setup, provide values with CLI options or set "
            "MLFLOW_TRACKING_URI and MLFLOW_EXPERIMENT_ID in your environment."
        )
    )
    click.echo("")


def _resolve_setup_inputs(
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
    non_interactive: bool,
) -> tuple[str | None, str | None, str | None]:
    resolved_tracking_uri = tracking_uri or os.environ.get(MLFLOW_TRACKING_URI.name)
    resolved_experiment_id = experiment_id or os.environ.get(MLFLOW_EXPERIMENT_ID.name)
    resolved_experiment_name = experiment_name or os.environ.get(MLFLOW_EXPERIMENT_NAME.name)

    if non_interactive or not _is_interactive_shell():
        return resolved_tracking_uri, resolved_experiment_id, resolved_experiment_name

    if not resolved_tracking_uri:
        import mlflow

        actual_default_tracking_uri = mlflow.get_tracking_uri()
        resolved_tracking_uri = click.prompt(
            _question("MLflow tracking URI"),
            default=_DEFAULT_TRACKING_URI_SENTINEL,
            show_default=True,
        ).strip()
        if resolved_tracking_uri == _DEFAULT_TRACKING_URI_SENTINEL:
            resolved_tracking_uri = actual_default_tracking_uri

    if not resolved_experiment_id and not resolved_experiment_name:
        resolved_experiment_id = click.prompt(
            _question("MLflow experiment ID"),
            default="0",
            show_default=True,
        ).strip()

    return resolved_tracking_uri, resolved_experiment_id, resolved_experiment_name


def _is_interactive_shell() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _handle_disable(settings_file: Path) -> None:
    """Handle disable command."""
    if disable_tracing_plugin(settings_file):
        click.echo(f"{_ok('✓')} Claude tracing disabled")
    else:
        click.echo(f"{_error('✗')} No Claude configuration found - tracing was not enabled")


def _show_status(target_dir: Path, settings_file: Path) -> None:
    """Show current tracing status."""
    click.echo(f"{_title('MLflow Claude Tracing Status')}")
    click.echo(f"{_label('Project:')} {_value(str(target_dir))}")

    status = get_tracing_status(settings_file)

    if not status.enabled:
        click.echo(f"{_error('✗')} Claude tracing is not enabled")
        if status.reason:
            click.echo(f"  {_label('Reason:')} {_muted(status.reason)}")
        return

    click.echo(f"{_ok('✓')} Claude tracing is enabled")
    click.echo(f"{_label('Tracking URI:')} {_value(str(status.tracking_uri))}")

    if status.experiment_name:
        click.echo(f"{_label('Experiment name:')} {_value(status.experiment_name)}")
    if status.experiment_id:
        click.echo(f"{_label('Experiment ID:')} {_value(status.experiment_id)}")
    elif not status.experiment_name:
        click.echo(f"{_label('Experiment:')} {_muted('Default (experiment 0)')}")


def _show_setup_status(
    target_dir: Path,
    settings_file: Path,
) -> None:
    """Show setup completion status."""
    current_dir = Path.cwd().resolve()
    status = get_tracing_status(settings_file)

    click.echo("")
    click.echo(_title("Setup Complete"))
    click.echo(f"{_label('Project:')} {_value(str(target_dir))}")

    # Show tracking configuration
    if status.tracking_uri:
        click.echo(f"{_label('Tracking URI:')} {_value(status.tracking_uri)}")

    if status.experiment_name:
        click.echo(f"{_label('Experiment name:')} {_value(status.experiment_name)}")
    if status.experiment_id:
        click.echo(f"{_label('Experiment ID:')} {_value(status.experiment_id)}")
    elif not status.experiment_name:
        click.echo(f"{_label('Experiment:')} {_muted('Default (experiment 0)')}")

    # Show next steps
    click.echo("")
    click.echo(_title("Next Steps"))

    # Only show cd if it's a different directory
    if target_dir != current_dir:
        click.echo(f"  {_muted('Work from:')} {_value(str(target_dir))}")

    click.echo(f"  {_muted('1.')} Use Claude Code as usual in this directory.")
    click.echo(
        f"  {_muted('2.')} Visit the MLflow UI after a Claude conversation ends to inspect traces."
    )

    click.echo("")
    click.echo(_title("Disable Later"))
    click.echo(f"  {_value('mlflow autolog claude --disable')}")


@claude.command("stop-hook", hidden=True)
def stop_hook() -> None:
    """Legacy hook shim kept for older Python-hook installations."""
    stop_hook_handler()
