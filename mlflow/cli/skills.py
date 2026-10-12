"""CLI commands for bundled MLflow Assistant skills and the skill registry."""

from pathlib import Path

import click

from mlflow.assistant.skill_installer import BundledSkill, list_bundled_skills
from mlflow.exceptions import MlflowException


def _echo_skill_details(skill: BundledSkill):
    skill_name_styled = click.style(skill.name, fg="cyan", bold=True)
    skill_path_styled = click.style(f" ({skill.path})", fg="cyan")
    click.echo(skill_name_styled + skill_path_styled)
    if skill.description:
        click.echo(f"  {skill.description}")


@click.group("skills")
def commands():
    """Inspect bundled MLflow skills and pull skills from the skill registry."""


@commands.command("list")
def list_command():
    """List the MLflow skills bundled with this installation."""
    skills = list_bundled_skills()
    if not skills:
        click.secho(
            "No MLflow skills found in this installation.\n"
            "If you are working from a source checkout, fetch the skills submodule with:\n"
            "    git submodule update --init --recursive",
            fg="yellow",
        )
        return

    for skill in skills:
        _echo_skill_details(skill)


@commands.command("view")
@click.argument("skill_name", type=str)
def view_command(skill_name: str):
    """View the details of an MLflow skill."""
    skills = list_bundled_skills()
    target_skill = next((s for s in skills if s.name == skill_name), None)
    if not target_skill:
        raise click.ClickException(f"Skill {skill_name} not found.")
    _echo_skill_details(target_skill)


@commands.command("pull")
@click.argument("uri", required=False)
@click.option(
    "--skill-uri",
    "skill_uri",
    help="Skill URI to pull, equivalent to the positional URI argument.",
)
@click.option(
    "--destination",
    "-d",
    type=click.Path(file_okay=False, path_type=Path),
    help=(
        "Directory to write the skill into. It must not exist or must be empty. "
        "Defaults to a directory named after the skill in the current working directory."
    ),
)
def pull_command(uri: str | None, skill_uri: str | None, destination: Path | None):
    """
    Pull a registered skill's content to a local directory.

    URI is skills:/<name>/<version>, skills:/<name>@<alias>, or skills:/<name> for the latest
    version, with an organization written as skills:/@<organization>/<name>. Content is fetched
    from the version's source with your own credentials and verified against its recorded
    digest before anything is written to the destination.
    """
    if uri is not None and skill_uri is not None and uri != skill_uri:
        raise click.UsageError("Pass the skill URI either as an argument or with --skill-uri.")
    if (target_uri := uri or skill_uri) is None:
        raise click.UsageError("Missing the skill URI to pull.")
    # mlflow.genai pulls in the evaluation stack; importing it lazily keeps the rest of the
    # skills CLI fast to start.
    from mlflow.genai.skills import pull

    try:
        path = pull(target_uri, destination=destination)
    except MlflowException as e:
        raise click.ClickException(e.message) from e
    click.echo(f"Pulled {target_uri} to {path}")
