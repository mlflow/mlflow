from pathlib import Path

import pytest
from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.os_environ_set_in_test import OsEnvironSetInTest


def test_os_environ_set_in_test(
    index_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    tmp_file = tmp_path / "test_file.py"
    code = """
import os

# Bad
def test_func():
    os.environ["MY_VAR"] = "value"

# Good
def non_test_func():
    os.environ["MY_VAR"] = "value"
"""
    tmp_file.write_text(code)
    config = Config(select={OsEnvironSetInTest.name})
    violations = lint_file(tmp_file.relative_to(tmp_path), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, OsEnvironSetInTest) for v in violations)
    assert violations[0].loc == Location(5, 4)
