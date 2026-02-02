from pathlib import Path

from clint.config import Config
from clint.linter import Position, Range, lint_file
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
    assert violations[0].range == Range(Position(5, 4))


def test_os_environ_pop_in_test(index_path: Path) -> None:
    code = """
import os

def test_something():
    # Bad
    os.environ.pop("MY_VAR")

    # Good
    # monkeypatch.delenv("MY_VAR")
"""
    config = Config(select={OsEnvironDeleteInTest.name})
    violations = lint_file(Path("test_env.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, OsEnvironDeleteInTest) for v in violations)
    assert violations[0].range == Range(Position(5, 4))


def test_os_environ_pop_with_default_in_test(index_path: Path) -> None:
    code = """
import os

def test_something():
    # Bad - with default value
    os.environ.pop("MY_VAR", None)

    # Good
    # monkeypatch.delenv("MY_VAR", raising=False)
"""
    config = Config(select={OsEnvironDeleteInTest.name})
    violations = lint_file(Path("test_env.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, OsEnvironDeleteInTest) for v in violations)
    assert violations[0].range == Range(Position(5, 4))


def test_os_environ_multiple_violations(index_path: Path) -> None:
    code = """
import os

def test_something():
    # Bad - del
    del os.environ["VAR1"]

    # Bad - pop
    os.environ.pop("VAR2")

    # Bad - pop with default
    os.environ.pop("VAR3", None)
"""
    config = Config(select={OsEnvironDeleteInTest.name})
    violations = lint_file(Path("test_env.py"), code, config, index_path)
    assert len(violations) == 3
    assert all(isinstance(v.rule, OsEnvironDeleteInTest) for v in violations)
    assert violations[0].range == Range(Position(5, 4))
    assert violations[1].range == Range(Position(8, 4))
    assert violations[2].range == Range(Position(11, 4))


def test_os_environ_pop_not_in_test(index_path: Path) -> None:
    code = """
import os

def some_function():
    # This is OK - not in a test file
    os.environ.pop("MY_VAR")
"""
    config = Config(select={OsEnvironDeleteInTest.name})
    violations = lint_file(Path("utils.py"), code, config, index_path)
    assert len(violations) == 0
