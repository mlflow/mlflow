"""
Skills installer for Claude Code.

Installs skills bundled with MLflow to the specified skills directory.
Skills are maintained in the mlflow/skills repository and included via git subtree.
"""

import shutil
from importlib.resources import files
from pathlib import Path

SKILL_MANIFEST_FILE = "SKILL.md"


def get_skills_directory(skills_location: str) -> Path:
    return Path(skills_location).expanduser()


def _get_bundled_skills_path() -> Path:
    return Path(str(files("mlflow.assistant.skills")))


def _find_skill_directories(path: Path) -> list[Path]:
    return [item.parent for item in path.rglob(SKILL_MANIFEST_FILE)]


def list_installed_skills(skills_location: str) -> list[str]:
    skills_dir = get_skills_directory(skills_location)
    if not skills_dir.exists():
        return []
    return sorted(d.name for d in _find_skill_directories(skills_dir))


def list_bundled_skills() -> list[str]:
    bundled_path = _get_bundled_skills_path()
    return sorted(d.name for d in _find_skill_directories(bundled_path))


def install_skills(skills_location: str) -> list[str]:
    skills_dir = get_skills_directory(skills_location)
    bundled_path = _get_bundled_skills_path()

    # Find all directories containing SKILL.md in bundled skills
    skill_dirs = _find_skill_directories(bundled_path)

    if not skill_dirs:
        return []

    # Create skills directory if it doesn't exist
    skills_dir.mkdir(parents=True, exist_ok=True)

    installed_skills = []
    for skill_dir in skill_dirs:
        target_dir = skills_dir / skill_dir.name
        shutil.copytree(skill_dir, target_dir, dirs_exist_ok=True)
        installed_skills.append(skill_dir.name)

    return sorted(installed_skills)
