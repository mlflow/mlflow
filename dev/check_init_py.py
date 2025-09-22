"""
Pre-commit hook to check for missing __init__.py files in mlflow package directories.

This script ensures that all directories under the mlflow package that contain Python files
also have an __init__.py file. This prevents setuptools from excluding these directories
during package build.

Usage:
    python dev/check_init_py.py

Requirements:
- If mlflow/foo/bar.py exists, mlflow/foo/__init__.py must exist.
- Ignore directories that do not contain any Python files (e.g., mlflow/server/js).
- Use 'git ls-files' to ensure only tracked files/directories are considered.
"""

import subprocess
import sys
from pathlib import Path


def get_tracked_python_files() -> list[Path]:
    """Get all tracked Python files under the mlflow directory using git ls-files."""
    try:
        result = subprocess.check_output(
            ["git", "ls-files", "mlflow/**/*.py"],
            text=True,
        )
        return [Path(f) for f in result.strip().split("\n") if f]
    except subprocess.CalledProcessError as e:
        print(f"Error running git ls-files: {e}", file=sys.stderr)
        sys.exit(1)


def get_python_directories(python_files: list[Path]) -> set[str]:
    """Extract all directories that contain Python files."""
    directories = set()
    for file_path in python_files:
        parent_dir = str(file_path.parent)
        # Include all directories, including the root mlflow directory
        directories.add(parent_dir)
    return directories


def check_missing_init_files(python_directories: set[str]) -> list[str]:
    """Check which directories are missing __init__.py files."""
    missing_init = []
    for directory in sorted(python_directories):
        init_file = Path(directory) / "__init__.py"
        if not init_file.exists():
            missing_init.append(directory)
    return missing_init


def main() -> int:
    """Main function that performs the check and returns exit code."""
    python_files = get_tracked_python_files()

    if not python_files:
        # No Python files found, nothing to check
        return 0

    python_directories = get_python_directories(python_files)
    missing_init_files = check_missing_init_files(python_directories)

    if missing_init_files:
        print("Error: The following directories contain Python files but lack __init__.py:")
        for directory in missing_init_files:
            print(f"  {directory}/")
        print("\nThis can cause setuptools to exclude these directories during package build.")
        print("Please add __init__.py files to the directories listed above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
