"""
Manage skill installation

Skills are maintained in the mlflow/assistant/skills subtree in the MLflow repository,
which points to the https://github.com/mlflow/skills repository.
"""

import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from mlflow.ai_commands.ai_command_utils import parse_frontmatter

SKILL_MANIFEST_FILE = "SKILL.md"
SKILLS_PACKAGE = "mlflow.assistant.skills"


@dataclass
class BundledSkill:
    name: str
    description: str
    path: Path


def _find_skill_directories(path: Path) -> list[Path]:
    return [item.parent for item in path.rglob(SKILL_MANIFEST_FILE)]


def list_bundled_skills() -> list[BundledSkill]:
    """List the MLflow skills bundled with this installation.

    Skills live in the ``mlflow.assistant.skills`` package.

    Returns:
        Skills sorted by name. Empty when the package is not importable or the
        submodule is not checked out (e.g. a development clone without
        ``git submodule update --init``).
    """
    try:
        skills_pkg = resources.files(SKILLS_PACKAGE)
    except ModuleNotFoundError:
        return []
    skills = []
    for item in skills_pkg.iterdir():
        if not item.is_dir():
            continue
        skill_manifest = item.joinpath(SKILL_MANIFEST_FILE)
        if not skill_manifest.is_file():
            continue
        with resources.as_file(skill_manifest) as manifest_path:
            metadata, _ = parse_frontmatter(manifest_path.read_text(encoding="utf-8"))
            skills.append(
                BundledSkill(
                    name=metadata.get("name") or manifest_path.parent.name,
                    description=metadata.get("description") or "",
                    path=manifest_path.parent,
                )
            )
    return sorted(skills, key=lambda skill: skill.name)


def install_skills(destination_path: Path) -> list[str]:
    """
    Install MLflow skills to the specified destination path (e.g., ~/.claude/skills).

    Args:
        destination_path: The path where skills should be installed.

    Returns:
        A list of installed skill names.
    """
    destination_dir = destination_path.expanduser()
    skills_pkg = resources.files(SKILLS_PACKAGE)
    installed_skills = []

    for item in skills_pkg.iterdir():
        if not item.is_dir():
            continue
        skill_manifest = item.joinpath(SKILL_MANIFEST_FILE)
        if not skill_manifest.is_file():
            continue

        # Use resources.as_file() on the manifest to get a real path
        with resources.as_file(skill_manifest) as manifest_path:
            skill_dir = manifest_path.parent
            target_dir = destination_dir / skill_dir.name
            destination_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(skill_dir, target_dir, dirs_exist_ok=True)
            installed_skills.append(skill_dir.name)

    return sorted(installed_skills)


def list_installed_skills(destination_path: Path) -> list[str]:
    """
    List installed skills in the specified destination path.

    Args:
        destination_path: The path where skills are installed.

    Returns:
        A list of installed skill names.
    """
    if not destination_path.exists():
        return []
    return sorted(d.name for d in _find_skill_directories(destination_path))
