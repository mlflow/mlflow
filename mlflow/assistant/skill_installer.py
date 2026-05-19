"""
Manage skill installation

Skills are maintained in the mlflow/assistant/skills subtree in the MLflow repository,
which points to the https://github.com/mlflow/skills repository.
"""

import shutil
from importlib import resources
from pathlib import Path

SKILL_MANIFEST_FILE = "SKILL.md"


def _find_skill_directories(path: Path) -> list[Path]:
    return [item.parent for item in path.rglob(SKILL_MANIFEST_FILE)]


def install_skills(destination_path: Path) -> list[str]:
    """
    Install MLflow skills to the specified destination path (e.g., ~/.claude/skills).

    Args:
        destination_path: The path where skills should be installed.

    Returns:
        A list of installed skill names.
    """
    destination_dir = destination_path.expanduser()
    skills_pkg = resources.files("mlflow.assistant.skills")
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
