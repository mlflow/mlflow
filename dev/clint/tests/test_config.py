from pathlib import Path

import pytest
from clint.config import Config


def test_config_validate_exclude_paths_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_file = tmp_path / "test_file.py"
    test_file.touch()
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f"""
[tool.clint]
exclude = [
    "{test_file}",
    "{test_dir}"
]
""")

    # Mock get_repo_root to return the tmp_path for this test
    monkeypatch.setattr("clint.config.get_repo_root", lambda: tmp_path)

    config = Config.load()
    assert len(config.exclude) == 2
    assert str(test_file) in config.exclude
    assert str(test_dir) in config.exclude


def test_config_validate_exclude_paths_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
exclude = [
    "non_existing_file.py",
    "non_existing_dir"
]
""")

    # Mock get_repo_root to return the tmp_path for this test
    monkeypatch.setattr("clint.config.get_repo_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="Non-existing paths found in exclude field") as exc_info:
        Config.load()

    error_msg = str(exc_info.value)
    assert "non_existing_file.py" in error_msg
    assert "non_existing_dir" in error_msg


def test_config_validate_exclude_paths_mixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing_file = tmp_path / "existing_file.py"
    existing_file.touch()

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f"""
[tool.clint]
exclude = [
    "{existing_file}",
    "non_existing_file.py"
]
""")

    # Mock get_repo_root to return the tmp_path for this test
    monkeypatch.setattr("clint.config.get_repo_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="Non-existing paths found in exclude field") as exc_info:
        Config.load()

    error_msg = str(exc_info.value)
    assert "non_existing_file.py" in error_msg
    assert str(existing_file) not in error_msg


def test_config_empty_exclude_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
exclude = []
""")

    # Mock get_repo_root to return the tmp_path for this test
    monkeypatch.setattr("clint.config.get_repo_root", lambda: tmp_path)

    config = Config.load()
    assert config.exclude == []


def test_config_no_exclude_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
select = ["do-not-disable"]
""")

    # Mock get_repo_root to return the tmp_path for this test
    monkeypatch.setattr("clint.config.get_repo_root", lambda: tmp_path)

    config = Config.load()
    assert config.exclude == []


def test_config_loads_from_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that config is loaded from repository root regardless of current working directory."""
    # Create a mock repository structure
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subdir = repo_root / "subdir"
    subdir.mkdir()

    # Create pyproject.toml in repo root
    pyproject = repo_root / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
select = ["do-not-disable"]
exclude = ["excluded_path"]
""")

    # Create the excluded path so validation passes
    excluded_path = repo_root / "excluded_path"
    excluded_path.mkdir()

    # Mock get_repo_root to return the repo_root
    monkeypatch.setattr("clint.config.get_repo_root", lambda: repo_root)

    # Change to subdirectory - config should still be loaded from repo root
    original_cwd = tmp_path.cwd()
    try:
        monkeypatch.chdir(subdir)
        config = Config.load()

        # Verify config was loaded correctly from repo root, not current directory
        assert "excluded_path" in config.exclude
        assert "do-not-disable" in config.select

    finally:
        monkeypatch.chdir(original_cwd)
