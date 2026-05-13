"""MLflow CLI commands for Assistant integration."""

import shutil
import sys
import threading
import time
from pathlib import Path

import click

from mlflow.assistant.config import AssistantConfig, ProjectConfig, SkillsConfig
from mlflow.assistant.providers import AssistantProvider, list_providers
from mlflow.assistant.providers.base import ProviderNotConfiguredError
from mlflow.assistant.skill_installer import install_skills


class Spinner:
    """Simple spinner animation for long-running operations."""

    def __init__(self, message: str = "Loading"):
        self.message = message
        self.spinning = False
        self.thread = None
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def _spin(self):
        i = 0
        while self.spinning:
            frame = self.frames[i % len(self.frames)]
            sys.stdout.write(f"\r{frame} {self.message}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    def __enter__(self):
        self.spinning = True
        self.thread = threading.Thread(target=self._spin, name="Spinner")
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.spinning = False
        if self.thread:
            self.thread.join()
        sys.stdout.write("\r" + " " * (len(self.message) + 4) + "\r")
        sys.stdout.flush()


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
    5. Ask where to install skills (user-level or project-level)
    6. Install provider-specific skills
    7. Save configuration

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
    project_path = _prompt_experiment_path()

    # Step 4: Ask for model
    model = _prompt_model()

    # Step 5: Ask for skill location
    skills_config = _prompt_skill_location(project_path)

    # Step 6: Install skills
    skill_path = _install_skills(provider, skills_config, project_path)

    # Step 7: Save configuration
    _save_config(provider, model, skills_config)

    # Show success message
    _show_init_success(provider, model, skill_path)


def _prompt_provider() -> AssistantProvider | None:
    """Prompt user to select a provider."""
    providers = list_providers()

    click.secho("Step 1/4: Select AI Provider", fg="cyan", bold=True)
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
        click.style(f"Select provider [1: {default_provider.display_name}]", fg="bright_blue"),
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
    click.secho("Step 2/4: Checking Provider", fg="cyan", bold=True)
    click.secho("-" * 30, fg="cyan")
    click.echo()

    # First check if CLI is installed
    claude_path = shutil.which("claude")
    if not claude_path:
        click.secho(
            "Claude Code CLI is not installed. "
            "Install it with: npm install -g @anthropic-ai/claude-code",
            fg="red",
        )
        click.echo()
        return False

    click.echo(f"Claude CLI found: {claude_path}")

    try:
        spinner_msg = "Checking connection... " + click.style(
            "(this may take a few seconds)", dim=True
        )
        with Spinner(spinner_msg):
            provider.check_connection()
        click.secho("Connection verified", fg="green")
        click.echo()
        return True
    except ProviderNotConfiguredError as e:
        click.secho(str(e), fg="red")
        click.echo()
        return False


def _fetch_recent_experiments(tracking_uri: str, max_results: int = 5) -> list[tuple[str, str]]:
    """Fetch recent experiments from the tracking server.

    Returns:
        List of (experiment_id, experiment_name) tuples.
    """
    import mlflow

    original_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()
        experiments = client.search_experiments(
            max_results=max_results,
            order_by=["last_update_time DESC"],
        )
        return [(exp.experiment_id, exp.name) for exp in experiments]
    except Exception:
        return []
    finally:
        mlflow.set_tracking_uri(original_uri)


def _resolve_experiment_id(tracking_uri: str, name_or_id: str) -> str | None:
    """Resolve experiment name or ID to experiment ID.

    Args:
        tracking_uri: MLflow tracking server URI.
        name_or_id: Experiment name or ID.

    Returns:
        Experiment ID if found, None otherwise.
    """
    import mlflow

    original_uri = mlflow.get_tracking_uri()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()

        # First try to get by ID (if it looks like an ID)
        if name_or_id.isdigit():
            try:
                if exp := client.get_experiment(name_or_id):
                    return exp.experiment_id
            except Exception:
                pass

        # Try to get by name
        if exp := client.get_experiment_by_name(name_or_id):
            return exp.experiment_id

        return None
    except Exception:
        return None
    finally:
        mlflow.set_tracking_uri(original_uri)


def _prompt_experiment_path() -> Path | None:
    """Prompt user to optionally connect an experiment with code repository.

    Returns:
        The project path if configured, None otherwise.
    """
    click.secho("Step 3/5: Experiment & Code Context ", fg="cyan", bold=True, nl=False)
    click.secho("[Optional, Recommended]", fg="green", bold=True)
    click.secho("-" * 30, fg="cyan")
    click.echo()
    click.echo("You can connect an experiment with a code repository to give")
    click.echo("the assistant context about your source code for better analysis.")
    click.secho("(You can also set this up later in the MLflow UI.)", dim=True)
    click.echo()

    connect = click.confirm(
        click.style(
            "Do you want to connect an experiment with a code repository?", fg="bright_blue"
        ),
        default=True,
    )

    if not connect:
        click.echo()
        return None

    click.echo()

    # Ask for tracking URI to fetch experiments
    tracking_uri = click.prompt(
        click.style("Enter the MLflow tracking server URI", fg="bright_blue"),
        default="http://localhost:5000",
    )

    click.echo()
    click.secho("Fetching recent experiments...", dim=True)

    # Fetch recent experiments
    experiments = _fetch_recent_experiments(tracking_uri)

    if not experiments:
        click.secho("Could not fetch experiments from the server.", fg="yellow")
        click.echo("You can set this up later in the MLflow UI.")
        click.echo()
        return None

    click.echo()
    click.echo(click.style("Select an experiment to connect:", fg="bright_blue"))
    click.echo()

    for i, (exp_id, exp_name) in enumerate(experiments, 1):
        click.echo(f"  {i}. {exp_name} (ID: {exp_id})")

    other_option = len(experiments) + 1
    click.echo(f"  {other_option}. Enter experiment name or ID manually")
    click.echo()

    choice = click.prompt(
        click.style("Select experiment", fg="bright_blue"),
        type=click.IntRange(1, other_option),
        default=1,
    )

    if choice == other_option:
        while True:
            click.echo()
            name_or_id = click.prompt(
                click.style("Experiment name or ID", fg="bright_blue"), default=""
            )
            if not name_or_id:
                click.secho("No experiment specified. Please try again.", fg="yellow")
                continue

            experiment_id = _resolve_experiment_id(tracking_uri, name_or_id)
            if experiment_id:
                # Use the input as display name (could be name or ID)
                experiment_name = name_or_id
                break

            click.secho(
                f"Experiment '{name_or_id}' not found. Please try again.",
                fg="red",
            )
    else:
        experiment_id, experiment_name = experiments[choice - 1]

    click.secho(
        f"Experiment '{experiment_name}' selected",
        fg="green",
    )
    click.echo()

    # Ask for project path
    default_path = str(Path.cwd())
    while True:
        raw_path = click.prompt(
            click.style("Enter the path to your project directory:", fg="bright_blue"),
            default=default_path,
        )
        # Expand ~ and resolve relative paths
        expanded_path = Path(raw_path).expanduser().resolve()
        if expanded_path.is_dir():
            project_path = str(expanded_path)
            break
        click.secho(f"Directory '{raw_path}' does not exist. Please try again.", fg="red")

    # Save the project path mapping locally
    try:
        config = AssistantConfig.load()
        config.projects[experiment_id] = ProjectConfig(type="local", location=project_path)
        config.save()
        click.secho(
            f"Project path {project_path} is saved for experiment '{experiment_name}'",
            fg="green",
        )
    except Exception as e:
        click.secho(f"Error saving project path: {e}", fg="red")

    click.echo()
    return expanded_path


def _prompt_model() -> str:
    """Prompt user for model selection."""
    click.secho("Step 4/5: Model Selection", fg="cyan", bold=True)
    click.secho("-" * 30, fg="cyan")
    click.echo()
    click.echo("Choose a model for analysis:")
    click.secho("  - Press Enter to use the default model (recommended)", dim=True)
    click.secho("  - Or type a specific model name (e.g., claude-sonnet-4-20250514)", dim=True)
    click.echo()

    model = click.prompt(click.style("Model", fg="bright_blue"), default="default")
    click.echo()
    return model


def _prompt_skill_location(project_path: Path | None) -> SkillsConfig:
    """Prompt user for skill installation location.

    Args:
        project_path: The project path from experiment setup, or None if skipped.

    Returns:
        SkillsConfig with the selected location type and optional custom path.
    """
    click.secho("Step 5/5: Skill Installation Location", fg="cyan", bold=True)
    click.secho("-" * 30, fg="cyan")
    click.echo()
    click.echo("Choose where to install MLflow skills for Assistant:")
    click.echo()

    # TODO: Update this when we support other providers
    user_path = Path.home() / ".claude" / "skills"
    click.echo(f"  1. User level ({user_path})")
    click.secho("     Skills available globally across all projects", dim=True)
    click.echo()

    if project_path:
        project_skill_path = project_path / ".claude" / "skills"
        click.echo(f"  2. Project level ({project_skill_path})")
        click.secho("     Skills available only in this project", dim=True)
        click.echo()
        click.echo("  3. Custom location")
        click.secho("     Specify a custom path for skills", dim=True)
        click.echo()
        valid_choices = ["1", "2", "3"]
    else:
        click.echo("  2. Custom location")
        click.secho("     Specify a custom path for skills", dim=True)
        click.echo()
        valid_choices = ["1", "2"]

    choice = click.prompt(
        click.style("Select location [1: User level]", fg="bright_blue"),
        default="1",
        type=click.Choice(valid_choices),
        show_choices=False,
        show_default=False,
    )

    click.echo()

    if choice == "1":
        return SkillsConfig(type="global")
    elif choice == "2" and project_path:
        return SkillsConfig(type="project")
    else:
        # Custom location
        while True:
            raw_path = click.prompt(
                click.style("Enter the custom path for skills", fg="bright_blue"),
                default=str(user_path),
            )
            expanded_path = Path(raw_path).expanduser().resolve()
            # For custom paths, we'll create the directory, so just check parent exists
            if expanded_path.parent.exists() or expanded_path.exists():
                click.echo()
                return SkillsConfig(type="custom", custom_path=str(expanded_path))
            click.secho(
                f"Parent directory '{expanded_path.parent}' does not exist. Please try again.",
                fg="red",
            )


def _install_skills(
    provider: AssistantProvider, skills_config: SkillsConfig, project_path: Path | None
) -> Path:
    """Install skills bundled with MLflow.

    Returns:
        The resolved path where skills were installed.
    """
    match skills_config.type:
        case "global":
            skill_path = provider.resolve_skills_path(Path.home())
        case "project":
            skill_path = provider.resolve_skills_path(project_path)
        case "custom":
            skill_path = Path(skills_config.custom_path).expanduser()
    if installed_skills := install_skills(skill_path):
        for skill in installed_skills:
            click.secho(f"  - {skill}")
    else:
        click.secho("No skills available to install.", fg="yellow")
    click.echo()
    return skill_path


def _save_config(provider: AssistantProvider, model: str, skills_config: SkillsConfig) -> None:
    """Save configuration to file."""
    click.secho("Saving Configuration", fg="cyan", bold=True)
    click.secho("-" * 30, fg="cyan")

    config = AssistantConfig.load()
    config.set_provider(provider.name, model)
    config.providers[provider.name].skills = skills_config
    config.save()

    click.secho("Configuration saved", fg="green")
    click.echo()


def _show_init_success(provider: AssistantProvider, model: str, skill_path: Path) -> None:
    """Show success message and next steps."""
    click.secho("  ~ * ~ * ~ * ~ * ~ * ~ * ~ * ~", fg="green")
    click.secho("        Setup Complete!        ", fg="green", bold=True)
    click.secho("  ~ * ~ * ~ * ~ * ~ * ~ * ~ * ~", fg="green")
    click.echo()
    click.secho("Configuration:", bold=True)
    click.echo(f"  Provider: {provider.display_name}")
    click.echo(f"  Model: {model}")
    click.echo(f"  Skills: {skill_path}")
    click.echo()
    click.secho("Next steps:", bold=True)
    click.echo("  1. Start MLflow server:")
    click.secho("     $ mlflow server", fg="cyan")
    click.echo()
    click.echo("  2. Open MLflow UI and navigate to an experiment")
    click.echo()
    click.echo("  3. Click 'Ask Assistant'")
    click.echo()
    click.secho("To reconfigure, run: ", nl=False)
    click.secho("mlflow assistant --configure", fg="cyan")
