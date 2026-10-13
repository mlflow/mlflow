"""CLI commands for inspecting MLflow Assistant skills."""

import click

from mlflow.assistant.skill_installer import BundledSkill, list_bundled_skills


def _echo_skill_details(skill: BundledSkill):
    skill_name_styled = click.style(skill.name, fg="cyan", bold=True)
    skill_path_styled = click.style(f" ({skill.path})", fg="cyan")
    click.echo(skill_name_styled + skill_path_styled)
    if skill.description:
        click.echo(f"  {skill.description}")


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
