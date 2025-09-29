import subprocess
from pathlib import Path
from typing import Generator

import pytest
from clint.config import Config
from clint.utils import get_repo_root


@pytest.fixture(autouse=True)
def clear_repo_root_cache() -> Generator[None, None, None]:
    """Clear the get_repo_root cache before each test to avoid cross-test contamination."""
    get_repo_root.cache_clear()
    yield
    get_repo_root.cache_clear()


@pytest.fixture
def tmp_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temporary git repository for testing."""
    subprocess.check_call(
        ["git", "init"], cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    subprocess.check_call(["git", "config", "user.name", "Test User"], cwd=tmp_path)
    subprocess.check_call(["git", "config", "user.email", "test@example.com"], cwd=tmp_path)
    monkeypatch.chdir(tmp_path)

    return tmp_path


def test_config_validate_exclude_paths_success(tmp_git_repo: Path) -> None:
    test_file = tmp_git_repo / "test_file.py"
    test_file.touch()
    test_dir = tmp_git_repo / "test_dir"
    test_dir.mkdir()

    pyproject = tmp_git_repo / "pyproject.toml"
    pyproject.write_text(f"""
[tool.clint]
exclude = [
    "{test_file.name}",
    "{test_dir.name}"
]
""")

    config = Config.load()
    assert len(config.exclude) == 2
    assert test_file.name in config.exclude
    assert test_dir.name in config.exclude


def test_config_validate_exclude_paths_failure(tmp_git_repo: Path) -> None:
    pyproject = tmp_git_repo / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
exclude = [
    "non_existing_file.py",
    "non_existing_dir"
]
""")

    with pytest.raises(ValueError, match="Non-existing paths found in exclude field") as exc_info:
        Config.load()

    error_msg = str(exc_info.value)
    assert "non_existing_file.py" in error_msg
    assert "non_existing_dir" in error_msg


def test_config_validate_exclude_paths_mixed(tmp_git_repo: Path) -> None:
    existing_file = tmp_git_repo / "existing_file.py"
    existing_file.touch()

    pyproject = tmp_git_repo / "pyproject.toml"
    pyproject.write_text(f"""
[tool.clint]
exclude = [
    "{existing_file.name}",
    "non_existing_file.py"
]
""")

    with pytest.raises(ValueError, match="Non-existing paths found in exclude field") as exc_info:
        Config.load()

    error_msg = str(exc_info.value)
    assert "non_existing_file.py" in error_msg
    assert "['non_existing_file.py']" in error_msg


def test_config_empty_exclude_list(tmp_git_repo: Path) -> None:
    pyproject = tmp_git_repo / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
exclude = []
""")

    config = Config.load()
    assert config.exclude == []


def test_config_no_exclude_field(tmp_git_repo: Path) -> None:
    pyproject = tmp_git_repo / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
select = ["do-not-disable"]
""")

    config = Config.load()
    assert config.exclude == []


def test_config_loads_from_repo_root(tmp_git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that config is loaded from repository root regardless of current working directory."""
    subdir = tmp_git_repo / "subdir"
    subdir.mkdir()
    pyproject = tmp_git_repo / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
select = ["do-not-disable"]
exclude = ["excluded_path"]
""")
    excluded_path = tmp_git_repo / "excluded_path"
    excluded_path.mkdir()
    monkeypatch.chdir(subdir)
    config = Config.load()
    assert "excluded_path" in config.exclude
    assert "do-not-disable" in config.select
