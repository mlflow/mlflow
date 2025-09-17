from pathlib import Path

import pytest
from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.pytest_mark_repeat import PytestMarkRepeat


def test_pytest_mark_repeat(
    index_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    tmp_file = tmp_path / "test_pytest_mark_repeat.py"
    code = """
import pytest

@pytest.mark.repeat(10)
def test_flaky_function():
    ...
"""
    tmp_file.write_text(code)
    config = Config(select={PytestMarkRepeat.name})
    violations = lint_file(tmp_file.relative_to(tmp_path), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, PytestMarkRepeat) for v in violations)
    assert violations[0].loc == Location(3, 1)
