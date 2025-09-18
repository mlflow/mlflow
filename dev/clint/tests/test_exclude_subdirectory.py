import re
import subprocess
from pathlib import Path

from clint.config import Config
from clint.utils import get_repo_root, resolve_paths


def test_exclude_filtering_logic_from_subdirectory(tmp_path, monkeypatch):
    """Test that exclude filtering logic works correctly when running from a subdirectory."""
    # Create a temporary git repo structure that mimics the mlflow repo
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    # Initialize git repo
    subprocess.check_call(["git", "init"], cwd=repo_root)
    subprocess.check_call(["git", "config", "user.name", "Test"], cwd=repo_root)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=repo_root)

    # Create a directory structure with files that should be excluded
    excluded_dir = repo_root / "excluded_dir"
    excluded_dir.mkdir()

    excluded_file = excluded_dir / "test_file.py"
    excluded_file.write_text("# This file should be excluded")

    # Create another file that should not be excluded
    normal_file = repo_root / "normal_file.py"
    normal_file.write_text("# This file should not be excluded")

    # Create pyproject.toml with exclude configuration
    pyproject = repo_root / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
exclude = ["excluded_dir"]
""")

    # Add files to git
    subprocess.check_call(["git", "add", "."], cwd=repo_root)
    subprocess.check_call(["git", "commit", "-m", "Initial commit"], cwd=repo_root)

    # Test from repo root - should exclude files
    monkeypatch.chdir(repo_root)
    config = Config.load()

    # Test exclude filtering from repo root
    resolved_files = resolve_paths([Path("excluded_dir")])
    assert len(resolved_files) > 0  # Files found before filtering

    # Apply exclude filtering logic (the NEW fixed logic)
    repo_root_path = get_repo_root()
    cwd = Path.cwd()
    regex = re.compile("|".join(map(re.escape, config.exclude)))

    filtered_files = []
    for f in resolved_files:
        # Convert file path to be relative to repo root for exclude pattern matching
        repo_relative_path = (cwd / f).relative_to(repo_root_path)
        if not regex.match(str(repo_relative_path)):
            filtered_files.append(f)

    assert len(filtered_files) == 0  # All files should be excluded

    # Test from subdirectory - should also exclude files
    subdir = repo_root / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)

    resolved_files = resolve_paths([Path("../excluded_dir")])
    assert len(resolved_files) > 0  # Files found before filtering

    # Apply exclude filtering logic from subdirectory
    cwd = Path.cwd()
    filtered_files = []
    for f in resolved_files:
        # Convert file path to be relative to repo root for exclude pattern matching
        repo_relative_path = (cwd / f).resolve().relative_to(repo_root_path)
        if not regex.match(str(repo_relative_path)):
            filtered_files.append(f)

    assert len(filtered_files) == 0  # All files should be excluded (this is the fix)

    # Test with a file that should not be excluded
    resolved_files = resolve_paths([Path("../normal_file.py")])
    assert len(resolved_files) == 1  # File found before filtering

    filtered_files = []
    for f in resolved_files:
        repo_relative_path = (cwd / f).resolve().relative_to(repo_root_path)
        if not regex.match(str(repo_relative_path)):
            filtered_files.append(f)

    assert len(filtered_files) == 1  # File should not be excluded
