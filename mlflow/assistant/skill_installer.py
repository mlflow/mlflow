"""
Manage skill installation

Skills are maintained in the mlflow/assistant/skills subtree in the MLflow repository,
which points to the https://github.com/mlflow/skills repository.
"""

import shutil
from importlib.resources import files
from pathlib import Path

SKILL_MANIFEST_FILE = "SKILL.md"


def install_skills(destination_path: Path) -> list[str]:
    """
    Install skills bundled with MLflow as a git subtree to the specified destination path.

    Args:
        destination_path: The path where skills should be installed.

    Returns:
        A list of installed skill names.
    """
    destination_dir = destination_path.expanduser()

    skill_dirs = _get_skills_to_install()
    if not skill_dirs:
        return []

    destination_dir.mkdir(parents=True, exist_ok=True)
    installed_skills = []
    for skill_dir in skill_dirs:
        target_dir = destination_dir / skill_dir.name
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


def _get_skills_to_install() -> list[Path]:
    skills_path = _get_subtree_skills_path()
    return _find_skill_directories(skills_path)


def _get_subtree_skills_path() -> Path:
    # Load skills from the mlflow.assistant.skills subtree
    return files("mlflow.assistant.skills")._paths[0]


def _find_skill_directories(path: Path) -> list[Path]:
    return [item.parent for item in path.rglob(SKILL_MANIFEST_FILE)]
