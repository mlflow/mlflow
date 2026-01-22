"""
Skills installer for Claude Code.

Installs skills from https://github.com/mlflow/skills to the specified
skills directory.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

SKILLS_REPO_URL = "https://github.com/mlflow/skills.git"
SKILL_MANIFEST_FILE = "SKILL.md"


class GitNotInstalledError(Exception):
    pass


class CloneFailedError(Exception):
    pass


def check_git_available() -> bool:
    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            check=True,
            timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_skills_directory(skills_location: str) -> Path:
    return Path(skills_location).expanduser()


def list_installed_skills(skills_location: str) -> list[str]:
    skills_dir = get_skills_directory(skills_location)
    if not skills_dir.exists():
        return []

    skills = [
        item.name
        for item in skills_dir.iterdir()
        if item.is_dir() and (item / SKILL_MANIFEST_FILE).exists()
    ]
    return sorted(skills)


def install_skills(skills_location: str) -> list[str]:
    if not check_git_available():
        raise GitNotInstalledError("Git is not installed or not available in PATH")

    skills_dir = get_skills_directory(skills_location)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        repo_path = tmp_path / "skills"

        # Clone the repository with depth 1 for efficiency
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", SKILLS_REPO_URL, repo_path],
                capture_output=True,
                check=True,
                timeout=60,
            )
        except subprocess.CalledProcessError as e:
            raise CloneFailedError(f"Failed to clone skills repository: {e.stderr.decode()}")
        except subprocess.TimeoutExpired:
            raise CloneFailedError("Timed out cloning skills repository")

        # Find all directories containing SKILL.md
        skill_dirs = _find_skill_directories(repo_path)

        if not skill_dirs:
            return []

        # Create skills directory if it doesn't exist
        skills_dir.mkdir(parents=True, exist_ok=True)

        installed_skills = []
        for skill_dir in skill_dirs:
            skill_name = skill_dir.name
            target_dir = skills_dir / skill_name

            # Remove existing skill directory if it exists (overwrite)
            if target_dir.exists():
                shutil.rmtree(target_dir)

            # Copy skill directory to target
            shutil.copytree(skill_dir, target_dir)
            installed_skills.append(skill_name)

        return sorted(installed_skills)


def _find_skill_directories(repo_path: Path) -> list[Path]:
    return [item.parent for item in repo_path.rglob(SKILL_MANIFEST_FILE)]
