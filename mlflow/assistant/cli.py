"""MLflow CLI commands for Assistant integration."""

import shutil
from pathlib import Path

import click

from mlflow.assistant.config import AssistantConfig, ProjectConfig
from mlflow.assistant.providers import AssistantProvider, list_providers


@click.command("assistant")
@click.option(
    "--configure",
    is_flag=True,
    help="Configure or reconfigure the assistant settings",
)
def commands(configure: bool):
    """MLflow Assistant - AI-powered trace analysis.

    Run 'mlflow assistant --configure' to set up the assistant.
    """
    if configure:
        _run_configuration()
    else:
        # Check if already configured
        config = AssistantConfig.load()
        if not config.providers:
            click.secho(
                "Assistant is not configured. Please run: mlflow assistant --configure",
                fg="yellow",
            )
        else:
            click.secho(
                "Assistant launch is not yet implemented. To use Assistant, run `mlflow assistant "
                "--configure` to setup, then launch the MLflow UI manually.",
                fg="yellow",
            )


def _run_configuration():
    """Configure MLflow Assistant for the UI.

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
        mlflow assistant --configure
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
    if not _prompt_experiment_path():
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
        marker = click.style(" [recommended]", fg="green") if i == 1 else ""
        click.echo(f"  {i}. {provider.display_name}{marker}")
        click.secho(f"     {provider.description}", dim=True)

    click.echo()
    click.secho("  More providers coming soon...", dim=True)
    click.echo()

    default_provider = providers[0]
    choice = click.prompt(
        f"Select provider [1: {default_provider.display_name}]",
        default="1",
        type=click.Choice([str(i) for i in range(1, len(providers) + 1)]),
        show_choices=False,
        show_default=False,
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


def _prompt_experiment_path() -> bool:
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

    # Ask for experiment ID
    click.echo("Enter the experiment ID to connect to the code repository:")
    click.secho("  Example: 123456789", dim=True)
    click.echo()

    experiment_id = click.prompt("Experiment ID", default="")

    if not experiment_id:
        click.secho("No experiment specified, skipping.", fg="yellow")
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

    # Save the project path mapping locally
    try:
        config = AssistantConfig.load()
        config.projects[experiment_id] = ProjectConfig(type="local", location=project_path)
        config.save()
        click.secho(
            f"Project path saved for experiment '{experiment_id}' -> {project_path}",
            fg="green",
        )
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
    skill_source = Path(__file__).parent / "skills"
    skills = [d.name for d in skill_source.glob("*")]
    dst = provider.skill_path
    dst.mkdir(parents=True, exist_ok=True)
    for skill in skills:
        shutil.copytree(skill_source / skill, dst / skill, dirs_exist_ok=True)

    click.secho("Installed skills: ", fg="green")
    for skill in skills:
        click.secho(f"  - {skill}")
    click.echo()


def _save_config(
    provider: AssistantProvider,
    model: str,
) -> None:
    """Save configuration to file."""
    click.secho("Saving Configuration", fg="cyan", bold=True)
    click.secho("-" * 30, fg="cyan")

    config = AssistantConfig.load()
    config.set_provider(provider.name, model)
    config.save()

    click.secho("Configuration saved", fg="green")
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
    click.secho("mlflow assistant --configure", fg="cyan")
