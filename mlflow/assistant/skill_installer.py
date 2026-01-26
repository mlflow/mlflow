"""
Manage skill installation

Skills are maintained in the mlflow/assistant/skills subtree in the MLflow repository,
which points to the https://github.com/mlflow/skills repository.
"""

import shutil
from importlib.resources import files
from pathlib import Path

SKILL_MANIFEST_FILE = "SKILL.md"


def _get_skills_source_path() -> Path | None:
    """Get the filesystem path to the bundled skills package.

    Returns:
        Path to the skills directory, or None if not available.
    """
    skills_pkg = files("mlflow.assistant.skills")

    # For directory-based packages (editable installs, source), _paths gives us the path
    if hasattr(skills_pkg, "_paths") and skills_pkg._paths:
        return skills_pkg._paths[0]

    # For MultiplexedPath (namespace packages), try _path attribute
    if hasattr(skills_pkg, "_path"):
        return Path(skills_pkg._path)

    return None


def install_skills(destination_path: Path) -> list[str]:
    """
    Install MLflow skills to the specified destination path (e.g., ~/.claude/skills).

    Args:
        destination_path: The path where skills should be installed.

    Returns:
        A list of installed skill names.
    """
    destination_dir = destination_path.expanduser()

    skills_path = _get_skills_source_path()
    if skills_path is None:
        return []

    skill_dirs = _find_skill_directories(skills_path)
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


def _find_skill_directories(path: Path) -> list[Path]:
    return [item.parent for item in path.rglob(SKILL_MANIFEST_FILE)]
