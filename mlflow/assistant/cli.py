"""MLflow CLI commands for Assistant integration."""

import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import click

import mlflow
from mlflow.assistant.providers import AssistantProvider, list_providers

# Tag key for storing project path on experiments
ASSISTANT_PROJECT_PATH_TAG = "mlflow.assistant.projectPath"


@click.group("assistant")
def commands():
    """Commands for MLflow Assistant setup and configuration.

    Use 'mlflow assistant init' to set up the AI assistant feature in MLflow UI.
    """


@commands.command("init")
def init():
    """Initialize MLflow Assistant for the UI.

    This interactive command sets up the AI assistant feature that allows you
    to analyze MLflow traces directly from the UI.

    The command will:
    1. Ask which provider to use (Claude Code for now)
    2. Check provider availability
    3. Optionally connect an experiment with code repository
    4. Ask which model to use
    5. Install provider-specific skills
    6. Save configuration

    Example:
        mlflow assistant init
    """
    click.echo()
    click.secho("╔══════════════════════════════════════════╗", fg="cyan")
    click.secho("║       *    .  *       .   *              ║", fg="cyan")
    click.secho("║   .    *  MLflow Assistant Setup   *  .  ║", fg="cyan", bold=True)
    click.secho("║      *    .       *   .      *           ║", fg="cyan")
    click.secho("╚══════════════════════════════════════════╝", fg="cyan")
    click.echo()

    # Step 1: Select provider
    provider = _prompt_provider()
    if provider is None:
        return

    # Step 2: Check provider availability
    if not _check_provider(provider):
        return

    # Step 3: Optionally connect experiment with code repository
    if not _prompt_experiment_connection():
        return

    # Step 4: Ask for model
    model = _prompt_model()

    # Step 5: Install skills
    _install_skills(provider)

    # Step 6: Save configuration
    _save_config(provider, model)

    # Show success message
    _show_init_success(provider, model)


def _prompt_provider() -> AssistantProvider | None:
    """Prompt user to select a provider."""
    providers = list_providers()

    click.secho("Select AI Provider", fg="cyan", bold=True)
    click.secho("-" * 30, fg="cyan")
    click.echo()

    for i, provider in enumerate(providers, 1):
        if i == 1:
            marker = click.style(" [recommended]", fg="green")
        else:
            marker = ""
        click.echo(f"  {i}. {provider.display_name}{marker}")
        click.secho(f"     {provider.description}", dim=True)

    click.echo()
    click.secho("  More providers coming soon...", dim=True)
    click.echo()

    choice = click.prompt(
        "Select provider",
        default="1",
        type=click.Choice([str(i) for i in range(1, len(providers) + 1)]),
        show_choices=False,
    )

    provider = providers[int(choice) - 1]
    click.echo()
    return provider


def _check_provider(provider: AssistantProvider) -> bool:
    """Check if the selected provider is available."""
    click.secho("Checking Provider Availability", fg="cyan", bold=True)
    click.secho("-" * 30, fg="cyan")

    try:
        provider.check_connection(echo=click.echo)
        click.echo()
        return True
    except RuntimeError as e:
        click.secho(str(e), fg="red")
        click.echo()
        return False


def _is_localhost(url: str) -> bool:
    """Check if a URL points to localhost."""
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    return hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0")


def _prompt_experiment_connection() -> bool:
    """Prompt user to optionally connect an experiment with code repository."""
    click.secho("Experiment & Code Context ", fg="cyan", bold=True, nl=False)
    click.secho("[Recommended]", fg="green", bold=True)
    click.secho("-" * 30, fg="cyan")
    click.echo()
    click.echo("You can connect an experiment with a code repository to give")
    click.echo("the assistant context about your source code for better analysis.")
    click.secho("(You can also set this up later in the MLflow UI.)", dim=True)
    click.echo()

    connect = click.confirm(
        "Do you want to connect an experiment with a code repository?",
        default=True,
    )

    if not connect:
        click.echo()
        return True

    click.echo()

    # Ask for tracking URI
    default_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    tracking_uri = click.prompt("Tracking URI", default=default_uri)

    # Check if it's a remote URL
    parsed = urlparse(tracking_uri)
    if parsed.scheme in ("http", "https") and not _is_localhost(tracking_uri):
        click.echo()
        click.secho(
            "Error: MLflow Assistant is only supported for locally hosted MLflow servers.",
            fg="red",
        )
        click.secho(
            f"The URL '{tracking_uri}' appears to be a remote server.",
            fg="red",
        )
        click.echo()
        return False

    mlflow.set_tracking_uri(tracking_uri)
    click.echo()

    # Ask for experiment ID or name
    click.echo("Enter the experiment ID or name to connect:")
    click.secho("  - Experiment ID: numeric ID (e.g., 123456789)", dim=True)
    click.secho("  - Experiment name: string name (e.g., 'my-experiment')", dim=True)
    click.echo()

    experiment_input = click.prompt("Experiment ID or name", default="")

    if not experiment_input:
        click.secho("No experiment specified, skipping.", fg="yellow")
        click.echo()
        return True

    # Get the experiment
    try:
        # Try to parse as integer (experiment ID)
        try:
            int(experiment_input)
            experiment = mlflow.get_experiment(experiment_input)
        except ValueError:
            experiment = mlflow.get_experiment_by_name(experiment_input)

        if experiment is None:
            click.secho(f"Experiment '{experiment_input}' not found, skipping.", fg="yellow")
            click.echo()
            return True

        experiment_id = experiment.experiment_id
    except Exception as e:
        click.secho(f"Error getting experiment: {e}", fg="red")
        click.echo()
        return True

    # Ask for project path
    click.echo()
    click.echo("Enter the path to your project directory.")
    click.secho("The assistant will read source files from this directory.", dim=True)
    click.echo()

    default_path = str(Path.cwd())
    project_path = click.prompt(
        "Project path",
        default=default_path,
        type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
    )

    # Set the project path as an experiment tag
    try:
        mlflow.MlflowClient().set_experiment_tag(
            experiment_id, ASSISTANT_PROJECT_PATH_TAG, project_path
        )
        click.secho(f"Project path saved to experiment '{experiment_input}'", fg="green")
    except Exception as e:
        click.secho(f"Error saving project path: {e}", fg="red")

    click.echo()
    return True


def _prompt_model() -> str:
    """Prompt user for model selection."""
    click.secho("Model Selection", fg="cyan", bold=True)
    click.secho("-" * 30, fg="cyan")
    click.echo()
    click.echo("Choose a model for analysis:")
    click.secho("  - Press Enter to use the default model (recommended)", dim=True)
    click.secho("  - Or type a specific model name (e.g., claude-sonnet-4-20250514)", dim=True)
    click.echo()

    model = click.prompt("Model", default="default")
    click.echo()
    return model


def _install_skills(provider: AssistantProvider) -> None:
    """Install provider-specific skills."""
    click.secho("Installing Skills", fg="cyan", bold=True)
    click.secho("-" * 30, fg="cyan")

    if provider.name == "claude_code":
        _install_claude_skills()
    else:
        click.secho("No skills to install for this provider.", dim=True)

    click.echo()


def _install_claude_skills() -> None:
    """Install MLflow-specific Claude skills."""
    # Get the skills directory from this package
    skills_source = Path(__file__).parent / "skills"
    skills_dest = Path.home() / ".claude" / "commands"

    if not skills_source.exists():
        click.secho("Skills directory not found (skipped)", fg="yellow")
        return

    # Create destination directory
    skills_dest.mkdir(parents=True, exist_ok=True)

    # Find all markdown files in the skills directory
    skill_files = list(skills_source.glob("*.md"))

    if not skill_files:
        click.secho("No skill files found (skipped)", fg="yellow")
        return

    for src in skill_files:
        dst = skills_dest / src.name
        shutil.copy2(src, dst)
        click.secho(f"Installed: {src.name}", fg="green")


def _save_config(
    provider: AssistantProvider,
    model: str,
) -> None:
    """Save configuration to file."""
    click.secho("Saving Configuration", fg="cyan", bold=True)
    click.secho("-" * 30, fg="cyan")

    config = {
        "provider": provider.name,
        "model": model,
    }

    # Use provider's config path
    config_file = provider.config_path
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    click.secho(f"Configuration saved to: {config_file}", fg="green")
    click.echo()


def _show_init_success(
    provider: AssistantProvider,
    model: str,
) -> None:
    """Show success message and next steps."""
    click.secho("  ~ * ~ * ~ * ~ * ~ * ~ * ~ * ~", fg="green")
    click.secho("        Setup Complete!        ", fg="green", bold=True)
    click.secho("  ~ * ~ * ~ * ~ * ~ * ~ * ~ * ~", fg="green")
    click.echo()
    click.secho("Configuration:", bold=True)
    click.echo(f"  Provider: {provider.display_name}")
    click.echo(f"  Model: {model}")
    click.echo()
    click.secho("Next steps:", bold=True)
    click.echo("  1. Start MLflow server:")
    click.secho("     mlflow server --port 5000", fg="cyan")
    click.echo()
    click.echo("  2. Open MLflow UI and navigate to a trace")
    click.echo()
    click.echo("  3. Click 'Ask Assistant'")
    click.echo()
    click.secho("To reconfigure, run: ", nl=False)
    click.secho("mlflow assistant init", fg="cyan")
