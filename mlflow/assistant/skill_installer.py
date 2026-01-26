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
    import logging
    destination_dir = destination_path.expanduser()
    logging.error(f"Destination directory: {destination_dir}")
    skills_pkg = resources.files("mlflow.assistant.skills")
    logging.error(f"Skills package: {skills_pkg}")
    installed_skills = []

    for item in skills_pkg.iterdir():
        logging.error(f"Item: {item}")
        if not item.is_dir():
            logging.error(f"Item is not a directory: {item}")
            continue
        skill_manifest = item.joinpath(SKILL_MANIFEST_FILE)
        logging.error(f"Skill manifest: {skill_manifest}")
        if not skill_manifest.is_file():
            logging.error(f"Skill manifest is not a file: {skill_manifest}")
            continue

        # Use resources.as_file() on the manifest to get a real path
        with resources.as_file(skill_manifest) as manifest_path:
            logging.error(f"Manifest path: {manifest_path}")
            skill_dir = manifest_path.parent
            logging.error(f"Skill directory: {skill_dir}")
            target_dir = destination_dir / skill_dir.name
            logging.error(f"Target directory: {target_dir}")
            destination_dir.mkdir(parents=True, exist_ok=True)
            logging.error(f"Destination directory: {destination_dir}")
            shutil.copytree(skill_dir, target_dir, dirs_exist_ok=True)
            logging.error(f"Copied skill directory: {skill_dir} to {target_dir}")
            installed_skills.append(skill_dir.name)

    logging.error(f"Installed skills: {installed_skills}")
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
