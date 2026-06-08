"""CLI command for installing MLflow Claude Code skills."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import click


@click.command("claude-setup")
@click.option(
    "--skills-repo",
    default="https://github.com/mlflow/skills",
    show_default=True,
    help="URL of the skills repository to clone.",
)
@click.option(
    "--target-dir",
    default=str(Path.home() / ".claude" / "skills"),
    show_default=True,
    help="Target directory where skills will be installed.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing skills.",
)
def claude_setup(skills_repo: str, target_dir: str, force: bool) -> None:
    """Install MLflow skills for Claude Code.

    Clones the mlflow/skills repository and installs each top-level skill
    directory into the target directory (default: ~/.claude/skills/).

    After installation, restart Claude Code to activate the new skills.

    \b
    Examples:
        mlflow claude-setup
        mlflow claude-setup --force
        mlflow claude-setup --target-dir /path/to/skills
        mlflow claude-setup --skills-repo https://github.com/my-org/my-skills
    """
    target = Path(target_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        click.echo(f"Cloning {skills_repo}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", skills_repo, tmpdir],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode(errors="replace").strip()
            raise click.ClickException(
                f"Failed to clone {skills_repo}.\n{stderr}"
            ) from None

        installed = []
        skipped = []

        for skill_dir in sorted(Path(tmpdir).iterdir()):
            if not skill_dir.is_dir() or skill_dir.name.startswith("."):
                continue

            dest = target / skill_dir.name
            if dest.exists() and not force:
                skipped.append(skill_dir.name)
                continue

            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(str(skill_dir), str(dest))
            installed.append(skill_dir.name)
            click.echo(f"  \u2713 {skill_dir.name}")

        if skipped:
            click.echo(f"\nSkipped (already installed): {', '.join(skipped)}")
            click.echo("Use --force to overwrite.")

        if installed:
            click.echo(
                f"\nInstalled {len(installed)} MLflow skill(s) to {target}"
            )
        else:
            click.echo(f"\nNo new skills installed to {target}")

        click.echo("Restart Claude Code to activate.")
