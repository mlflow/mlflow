#!/usr/bin/env python3
"""Pre-commit hook to enforce __init__.py files in all test directories."""

import sys
from pathlib import Path


def main() -> int:
    """Check that all directories under tests/ have __init__.py files."""
    repo_root = Path(__file__).parent.parent
    tests_dir = repo_root / "tests"

    if not tests_dir.exists():
        print(f"Error: tests directory not found at {tests_dir}")
        return 1

    missing_init_files = []

    # Find all directories under tests/
    for directory in tests_dir.rglob("*"):
        if (
            directory.is_dir()
            and not directory.name.startswith(".")
            and directory.name != "__pycache__"
        ):
            init_file = directory / "__init__.py"
            if not init_file.exists():
                # Convert to relative path for cleaner output
                rel_path = directory.relative_to(repo_root)
                missing_init_files.append(rel_path)

    if missing_init_files:
        print("Error: The following test directories are missing __init__.py files:")
        for path in sorted(missing_init_files):
            print(f"  {path}")
        print()
        print("To fix this, create empty __init__.py files in these directories:")
        for path in sorted(missing_init_files):
            print(f"  touch {path}/__init__.py")
        return 1

    print("All test directories have __init__.py files ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
