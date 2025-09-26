from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.os_chdir_in_test import OsChdirInTest


def test_os_chdir_in_test(index_path: Path) -> None:
    code = """
import os

# Bad
def test_func():
    os.chdir("/tmp")

# Good
def non_test_func():
    os.chdir("/tmp")
"""
    config = Config(select={OsChdirInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, OsChdirInTest) for v in violations)
    assert violations[0].loc == Location(5, 4)


def test_os_chdir_in_test_with_from_import(index_path: Path) -> None:
    code = """
from os import chdir

# Bad
def test_func():
    chdir("/tmp")

# Good
def non_test_func():
    chdir("/tmp")
"""
    config = Config(select={OsChdirInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, OsChdirInTest) for v in violations)
    assert violations[0].loc == Location(5, 4)


def test_os_chdir_in_test_no_violation_outside_test(index_path: Path) -> None:
    code = """
import os

def normal_function():
    os.chdir("/tmp")
"""
    config = Config(select={OsChdirInTest.name})
    violations = lint_file(Path("non_test_file.py"), code, config, index_path)
    assert len(violations) == 0


def test_os_chdir_in_test_with_alias(index_path: Path) -> None:
    code = """
import os as operating_system

# Bad - should still catch aliased import
def test_func():
    operating_system.chdir("/tmp")
"""
    config = Config(select={OsChdirInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 1
    assert all(isinstance(v.rule, OsChdirInTest) for v in violations)
    assert violations[0].loc == Location(5, 4)


def test_os_chdir_in_test_nested_functions_not_caught(index_path: Path) -> None:
    """
    Nested functions are not considered to be "in test" - this matches
    the behavior of other test-specific rules like os.environ.
    """
    code = """
import os

def test_outer():
    def inner_function():
        os.chdir("/tmp")  # Not caught since inner_function is not a test function
    inner_function()
"""
    config = Config(select={OsChdirInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 0


def test_os_chdir_not_os_module(index_path: Path) -> None:
    code = """
class FakeOs:
    @staticmethod
    def chdir(path):
        pass

fake_os = FakeOs()

def test_func():
    fake_os.chdir("/tmp")  # Should not trigger since it's not os.chdir
"""
    config = Config(select={OsChdirInTest.name})
    violations = lint_file(Path("test_file.py"), code, config, index_path)
    assert len(violations) == 0
