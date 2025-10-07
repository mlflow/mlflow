from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.pytest_mark_repeat import PytestMarkRepeat


def test_pytest_mark_repeat(index_path: Path) -> None:
    code = """
import pytest

@pytest.mark.repeat(10)
def test_flaky_function():
    ...
"""
    config = Config(select={PytestMarkRepeat.name})
    violations = lint_file(Path("test_pytest_mark_repeat.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, PytestMarkRepeat) for v in violations)
    assert violations[0].loc == Location(3, 1)
