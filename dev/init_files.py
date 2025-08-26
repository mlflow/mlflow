"""Pre-commit hook to enforce __init__.py files in all test directories."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Check that all directories under tests/ have __init__.py files."""
    repo_root = Path(__file__).parent.parent
    tests_dir = repo_root / "tests"

    if not tests_dir.exists():
        print(f"Error: tests directory not found at {tests_dir}")
        return 1

    # Get all git-tracked files in the tests directory
    result = subprocess.check_output(
        ["git", "ls-files", "tests/"],
        cwd=repo_root,
        text=True,
    )
    tracked_files = result.strip().splitlines()

    # Extract unique directories from tracked files
    tracked_directories: set[Path] = set()
    for file_path in tracked_files:
        if file_path:  # Skip empty strings
            path = Path(file_path)
            # Add all parent directories of this file
            tracked_directories.update(p for p in path.parents if p.is_relative_to(tests_dir))

    # Check each tracked directory for __init__.py
    missing_init_files: list[Path] = []
    for directory in tracked_directories:
        full_dir_path = repo_root / directory
        if full_dir_path.is_dir():
            init_file = full_dir_path / "__init__.py"
            if not init_file.exists():
                missing_init_files.append(directory)

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
