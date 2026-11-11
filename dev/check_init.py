"""
Pre-commit hook to check for missing `__init__.py` files in mlflow, tests, and skill scripts.

This script ensures that all directories under the mlflow package and tests directory that contain
Python files also have an `__init__.py` file. This prevents `setuptools` from excluding these
directories during package build and ensures test modules are properly structured. It also checks
that all directories containing Claude skill scripts, up to and including the skill directory, have
an `__init__.py` file so mypy can distinguish same-named modules in different skills.

Usage:
    uv run dev/check_init.py

Requirements:
- If `mlflow/foo/bar.py` exists, `mlflow/foo/__init__.py` must exist.
- If `tests/foo/test_bar.py` exists, `tests/foo/__init__.py` must exist.
- Only test files (starting with `test_`) in the tests directory are checked.
- All parent directories of Python files are checked recursively for `__init__.py`.
- For `.claude/skills/<skill>/scripts/**/*.py`, directories from the script's directory through
  `.claude/skills/<skill>/` must have an `__init__.py` file. `.claude/`, `.claude/skills/`,
  `.claude/skills/src/`, and `.claude/skills/tests/` are not checked.
- Ignore directories that do not contain any Python files (e.g., `mlflow/server/js`).
"""

import subprocess
import sys
from pathlib import Path


def get_tracked_python_files() -> list[Path]:
    try:
        result = subprocess.check_output(
            ["git", "ls-files", "mlflow/**/*.py", "tests/**/*.py", ".claude/skills/**/*.py"],
            text=True,
        )
        paths = (Path(f) for f in result.splitlines() if f)
        return [
            p
            for p in paths
            if (not p.is_relative_to("tests") or p.name.startswith("test_"))
            and (
                not p.is_relative_to(".claude/skills")
                or (
                    len(p.parts) >= 5
                    and p.parts[2] not in {"src", "tests"}
                    and p.parts[3] == "scripts"
                )
            )
        ]
    except subprocess.CalledProcessError as e:
        print(f"Error running git ls-files: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> int:
    python_files = get_tracked_python_files()
    if not python_files:
        return 0

    python_dirs = {
        p
        for f in python_files
        if not f.is_relative_to(".claude/skills")
        for p in f.parents
        if p != Path(".")
    }
    for f in python_files:
        if f.is_relative_to(".claude/skills"):
            skill_dir = Path(*f.parts[:3])
            python_dirs.update(p for p in f.parents if p.is_relative_to(skill_dir))

    if missing_init_files := [d for d in python_dirs if not (d / "__init__.py").exists()]:
        print("Error: The following directories contain Python files but lack __init__.py:")
        for d in sorted(missing_init_files):
            print(f"  {d.as_posix()}/")
        print("Please add __init__.py files to the directories listed above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
