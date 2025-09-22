"""
Pre-commit hook to check for missing `__init__.py` files in mlflow package directories.

This script ensures that all directories under the mlflow package that contain Python files
also have an `__init__.py` file. This prevents `setuptools` from excluding these directories
during package build.

Usage:
    uv run dev/check_init_py.py

Requirements:
- If `mlflow/foo/bar.py` exists, `mlflow/foo/__init__.py` must exist.
- Ignore directories that do not contain any Python files (e.g., `mlflow/server/js`).
"""

import subprocess
import sys
from pathlib import Path


def get_tracked_python_files() -> list[Path]:
    try:
        result = subprocess.check_output(
            ["git", "ls-files", "mlflow/**/*.py"],
            text=True,
        )
        return [Path(f) for f in result.splitlines() if f]
    except subprocess.CalledProcessError as e:
        print(f"Error running git ls-files: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    python_files = get_tracked_python_files()
    if not python_files:
        return 0

    python_dirs = {f.parent for f in python_files}
    missing_init_files = [d for d in python_dirs if not (d / "__init__.py").exists()]
    if missing_init_files:
        print("Error: The following directories contain Python files but lack __init__.py:")
        for d in sorted(missing_init_files):
            print(f"  {d.as_posix()}/")
        print("Please add __init__.py files to the directories listed above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
