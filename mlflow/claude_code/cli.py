"""MLflow CLI commands for Claude Code integration."""

import json
import shutil
import subprocess
from pathlib import Path

import click

from mlflow.claude_code.config import get_tracing_status, setup_environment_config
from mlflow.claude_code.hooks import disable_tracing_hooks, setup_hooks_config

# Default config file location
CLAUDE_CONFIG_FILE = Path.home() / ".mlflow" / "claude-config.json"

# Available models
MODELS = [
    ("default", "Let Claude decide the best model"),
    ("claude-sonnet-4-20250514", "Claude Sonnet 4 - Fast and capable"),
    ("claude-opus-4-20250514", "Claude Opus 4 - Most capable"),
]


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
        click.echo(f"   mlflow server --backend-store-uri {tracking_uri}")
    elif not tracking_uri:
        click.echo("\n💡 View your traces:")
        click.echo("   mlflow server")
    elif tracking_uri == "databricks":
        click.echo("\n💡 View your traces in your Databricks workspace")

    click.echo("\n🔧 To disable tracing later:")
    click.echo("   mlflow autolog claude --disable")


# =============================================================================
# New "claude" command group for Ask Claude integration
# =============================================================================


@click.group("claude")
def claude_commands():
    """Commands for Claude Code integration with MLflow UI.

    Use 'mlflow claude init' to set up the Ask Claude feature in MLflow UI.
    """


@claude_commands.command("init")
def init():
    """Initialize Claude Code integration for MLflow UI.

    This interactive command sets up the Ask Claude feature that allows you
    to analyze MLflow traces directly from the UI using Claude Code.

    The command will:
    1. Check that Claude CLI is installed and authenticated
    2. Ask for your project path (for Claude to read source files)
    3. Ask which model to use (default recommended)
    4. Install MLflow-specific Claude skills
    5. Save configuration to ~/.mlflow/claude-config.json

    Example:
        mlflow claude init
    """
    click.echo("=" * 50)
    click.echo("🚀 MLflow Claude Integration Setup")
    click.echo("=" * 50)
    click.echo()

    # Step 1: Check Claude CLI
    if not _check_claude_cli():
        return

    # Step 2: Ask for project path
    project_path = _prompt_project_path()

    # Step 3: Ask for model
    model = _prompt_model()

    # Step 4: Install skills
    _install_skills()

    # Step 5: Save configuration
    _save_config(project_path, model)

    # Show success message
    _show_init_success(project_path, model)


def _check_claude_cli() -> bool:
    """Check if Claude CLI is installed and authenticated."""
    click.echo("Checking Claude CLI...")

    # Check if claude command exists
    claude_path = shutil.which("claude")
    if not claude_path:
        click.echo("❌ Claude CLI not found.")
        click.echo()
        click.echo("Please install Claude CLI first:")
        click.echo("   npm install -g @anthropic-ai/claude-code")
        click.echo()
        click.echo("Then authenticate:")
        click.echo("   claude login")
        return False

    click.echo(f"✅ Claude CLI found: {claude_path}")

    # Check version
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = result.stdout.strip() or result.stderr.strip()
        click.echo(f"   Version: {version}")
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        click.echo("   (Could not determine version)")

    # Check authentication
    try:
        result = subprocess.run(
            ["claude", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            click.echo("✅ Claude CLI authenticated")
        else:
            click.echo("⚠️  Claude CLI may not be authenticated")
            click.echo("   Run 'claude login' if you encounter issues")
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
        click.echo("⚠️  Could not check authentication status")

    click.echo()
    return True


def _prompt_project_path() -> str:
    """Prompt user for project path."""
    click.echo("📁 Project Path Configuration")
    click.echo("-" * 30)
    click.echo("Claude can read your source code to help debug errors.")
    click.echo("Enter the path to your project directory.")
    click.echo()

    default_path = str(Path.cwd())
    project_path = click.prompt(
        "Project path",
        default=default_path,
        type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    )

    click.echo()
    return project_path


def _prompt_model() -> str:
    """Prompt user for model selection."""
    click.echo("🤖 Model Selection")
    click.echo("-" * 30)
    click.echo("Which Claude model should be used for analysis?")
    click.echo()

    for i, (model_id, description) in enumerate(MODELS, 1):
        marker = "→" if i == 1 else " "
        click.echo(f"  {marker} {i}. {model_id}")
        click.echo(f"      {description}")

    click.echo()
    choice = click.prompt(
        "Select model (1-3)",
        default="1",
        type=click.Choice([str(i) for i in range(1, len(MODELS) + 1)]),
    )

    model = MODELS[int(choice) - 1][0]
    click.echo()
    return model


def _install_skills() -> None:
    """Install MLflow-specific Claude skills."""
    click.echo("📚 Installing MLflow Skills")
    click.echo("-" * 30)

    # Get the skills directory from this package
    skills_source = Path(__file__).parent / "skills"
    skills_dest = Path.home() / ".claude" / "commands"

    # Create destination directory
    skills_dest.mkdir(parents=True, exist_ok=True)

    # Copy skill files
    skill_files = [
        "mlflow-trace-analyzer.md",
        "mlflow-error-debugger.md",
    ]

    for skill_file in skill_files:
        src = skills_source / skill_file
        dst = skills_dest / skill_file

        if src.exists():
            shutil.copy2(src, dst)
            click.echo(f"✅ Installed: {skill_file}")
        else:
            click.echo(f"⚠️  Skill file not found: {skill_file}")

    click.echo()


def _save_config(project_path: str, model: str) -> None:
    """Save configuration to file."""
    click.echo("💾 Saving Configuration")
    click.echo("-" * 30)

    config = {
        "projectPath": project_path,
        "model": model,
    }

    # Create directory if needed
    CLAUDE_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(CLAUDE_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

    click.echo(f"✅ Configuration saved to: {CLAUDE_CONFIG_FILE}")
    click.echo()


def _show_init_success(project_path: str, model: str) -> None:
    """Show success message and next steps."""
    click.echo("=" * 50)
    click.echo("🎉 Setup Complete!")
    click.echo("=" * 50)
    click.echo()
    click.echo("Configuration:")
    click.echo(f"  📁 Project: {project_path}")
    click.echo(f"  🤖 Model: {model}")
    click.echo()
    click.echo("Next steps:")
    click.echo("  1. Start MLflow server:")
    click.echo("     mlflow server --host 0.0.0.0 --port 5000")
    click.echo()
    click.echo("  2. Open MLflow UI and navigate to a trace")
    click.echo()
    click.echo("  3. Click 'Ask Claude' to analyze traces")
    click.echo()
    click.echo("To reconfigure, run: mlflow claude init")
