from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.os_environ_delete_in_test import OsEnvironDeleteInTest


def test_os_environ_delete_in_test(index_path: Path) -> None:
    code = """
import os

def test_something():
    # Bad
    del os.environ["MY_VAR"]

    # Good
    # monkeypatch.delenv("MY_VAR")
"""
    config = Config(select={OsEnvironDeleteInTest.name})
    violations = lint_file(Path("test_env.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, OsEnvironDeleteInTest) for v in violations)
    assert violations[0].loc == Location(5, 4)
