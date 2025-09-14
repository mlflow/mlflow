from pathlib import Path

import pytest
from clint.config import Config


def test_config_validate_exclude_paths_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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

    monkeypatch.chdir(tmp_path)
    config = Config.load()
    assert len(config.exclude) == 2
    assert str(test_file) in config.exclude
    assert str(test_dir) in config.exclude


def test_config_validate_exclude_paths_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
exclude = [
    "non_existing_file.py",
    "non_existing_dir"
]
""")

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Non-existing paths found in exclude field") as exc_info:
        Config.load()

    error_msg = str(exc_info.value)
    assert "non_existing_file.py" in error_msg
    assert "non_existing_dir" in error_msg


def test_config_validate_exclude_paths_mixed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Non-existing paths found in exclude field") as exc_info:
        Config.load()

    error_msg = str(exc_info.value)
    assert "non_existing_file.py" in error_msg
    assert str(existing_file) not in error_msg


def test_config_empty_exclude_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
exclude = []
""")

    monkeypatch.chdir(tmp_path)
    config = Config.load()
    assert config.exclude == []


def test_config_no_exclude_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
[tool.clint]
select = ["do-not-disable"]
""")

    monkeypatch.chdir(tmp_path)
    config = Config.load()
    assert config.exclude == []
