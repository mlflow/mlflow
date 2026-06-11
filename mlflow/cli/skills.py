"""CLI commands for inspecting MLflow Assistant skills."""

import click

from mlflow.assistant.skill_installer import list_bundled_skills


@click.group("skills")
def commands():
    """Inspect the MLflow skills bundled with this installation."""


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
        click.secho(skill.name, fg="cyan", bold=True)
        if skill.description:
            click.echo(f"  {skill.description}")
