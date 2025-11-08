"""
Pre-commit hook to check for missing `__init__.py` files in mlflow and tests directories.

This script ensures that all directories under the mlflow package and tests directory that contain
Python files also have an `__init__.py` file. This prevents `setuptools` from excluding these
directories during package build and ensures test modules are properly structured.

Usage:
    uv run dev/check_init_py.py

Requirements:
- If `mlflow/foo/bar.py` exists, `mlflow/foo/__init__.py` must exist.
- If `tests/foo/test_bar.py` exists, `tests/foo/__init__.py` must exist.
- Only test files (starting with `test_`) in the tests directory are checked.
- All parent directories of Python files are checked recursively for `__init__.py`.
- Ignore directories that do not contain any Python files (e.g., `mlflow/server/js`).
"""

import subprocess
import sys
from pathlib import Path


def get_tracked_python_files() -> list[Path]:
    try:
        result = subprocess.check_output(
            ["git", "ls-files", "mlflow/**/*.py", "tests/**/*.py"],
            text=True,
        )
        paths = (Path(f) for f in result.splitlines() if f)
        return [p for p in paths if not p.is_relative_to("tests") or p.name.startswith("test_")]
    except subprocess.CalledProcessError as e:
        print(f"Error running git ls-files: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    python_files = get_tracked_python_files()
    if not python_files:
        return 0

    python_dirs = {p for f in python_files for p in f.parents if p != Path(".")}
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
