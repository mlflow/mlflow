from pathlib import Path

import pytest
from clint.config import Config


@pytest.fixture(scope="module")
def temp_git_repo(tmp_path_factory):
    """Create a temporary git repository for testing."""
    tmp_path = tmp_path_factory.mktemp("git_repo")

    # Initialize a git repository
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)

    return tmp_path


def test_config_validate_exclude_paths_success(
    temp_git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_file = temp_git_repo / "test_file.py"
    test_file.touch()
    test_dir = temp_git_repo / "test_dir"
    test_dir.mkdir()

    pyproject = temp_git_repo / "pyproject.toml"
    pyproject.write_text(f"""
[tool.clint]
exclude = [
    "{test_file.name}",
    "{test_dir.name}"
]
""")

    # Mock get_repo_root to return the temp_git_repo
    monkeypatch.setattr("clint.config.get_repo_root", lambda: temp_git_repo)

    config = Config.load()
    assert len(config.exclude) == 2
    assert test_file.name in config.exclude
    assert test_dir.name in config.exclude


def test_config_validate_exclude_paths_failure(
    temp_git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pyproject = temp_git_repo / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
exclude = [
    "non_existing_file.py",
    "non_existing_dir"
]
""")

    # Mock get_repo_root to return the temp_git_repo
    monkeypatch.setattr("clint.config.get_repo_root", lambda: temp_git_repo)

    with pytest.raises(ValueError, match="Non-existing paths found in exclude field") as exc_info:
        Config.load()

    error_msg = str(exc_info.value)
    assert "non_existing_file.py" in error_msg
    assert "non_existing_dir" in error_msg


def test_config_validate_exclude_paths_mixed(
    temp_git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing_file = temp_git_repo / "existing_file.py"
    existing_file.touch()

    pyproject = temp_git_repo / "pyproject.toml"
    pyproject.write_text(f"""
[tool.clint]
exclude = [
    "{existing_file.name}",
    "non_existing_file.py"
]
""")

    # Mock get_repo_root to return the temp_git_repo
    monkeypatch.setattr("clint.config.get_repo_root", lambda: temp_git_repo)

    with pytest.raises(ValueError, match="Non-existing paths found in exclude field") as exc_info:
        Config.load()

    error_msg = str(exc_info.value)
    assert "non_existing_file.py" in error_msg
    # Check that only non_existing_file.py is in the error list, not existing_file.py
    assert "['non_existing_file.py']" in error_msg


def test_config_empty_exclude_list(temp_git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pyproject = temp_git_repo / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
exclude = []
""")

    # Mock get_repo_root to return the temp_git_repo
    monkeypatch.setattr("clint.config.get_repo_root", lambda: temp_git_repo)

    config = Config.load()
    assert config.exclude == []


def test_config_no_exclude_field(temp_git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pyproject = temp_git_repo / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
select = ["do-not-disable"]
""")

    # Mock get_repo_root to return the temp_git_repo
    monkeypatch.setattr("clint.config.get_repo_root", lambda: temp_git_repo)

    config = Config.load()
    assert config.exclude == []


def test_config_loads_from_repo_root(temp_git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that config is loaded from repository root regardless of current working directory."""
    # Create a subdirectory within the git repo
    subdir = temp_git_repo / "subdir"
    subdir.mkdir()

    # Create pyproject.toml in repo root
    pyproject = temp_git_repo / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
select = ["do-not-disable"]
exclude = ["excluded_path"]
""")

    # Create the excluded path so validation passes
    excluded_path = temp_git_repo / "excluded_path"
    excluded_path.mkdir()

    # Mock get_repo_root to return the temp_git_repo
    monkeypatch.setattr("clint.config.get_repo_root", lambda: temp_git_repo)

    # Change to subdirectory - config should still be loaded from repo root
    monkeypatch.chdir(subdir)
    config = Config.load()

    # Verify config was loaded correctly from repo root, not current directory
    assert "excluded_path" in config.exclude
    assert "do-not-disable" in config.select
