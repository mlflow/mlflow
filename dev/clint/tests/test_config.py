from pathlib import Path
from unittest.mock import patch

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

    with patch("clint.config.get_repo_root", return_value=tmp_path):
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

    with patch("clint.config.get_repo_root", return_value=tmp_path):
        with pytest.raises(
            ValueError, match="Non-existing paths found in exclude field"
        ) as exc_info:
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

    with patch("clint.config.get_repo_root", return_value=tmp_path):
        with pytest.raises(
            ValueError, match="Non-existing paths found in exclude field"
        ) as exc_info:
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

    with patch("clint.config.get_repo_root", return_value=tmp_path):
        config = Config.load()
    assert config.exclude == []


def test_config_no_exclude_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
select = ["do-not-disable"]
""")

    with patch("clint.config.get_repo_root", return_value=tmp_path):
        config = Config.load()
    assert config.exclude == []


def test_config_loads_from_repo_root_regardless_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that config is loaded from repo root, not current working directory."""
    # Setup repo root with pyproject.toml
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    pyproject = repo_root / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
select = ["do-not-disable"]
""")

    # Create a subdirectory within the repo
    subdir = repo_root / "subdir"
    subdir.mkdir()

    # Change working directory to the subdirectory
    monkeypatch.chdir(subdir)

    # Config should still load from repo root, not the subdirectory
    with patch("clint.config.get_repo_root", return_value=repo_root):
        config = Config.load()

    # Verify the config was loaded
    assert "do-not-disable" in config.select
