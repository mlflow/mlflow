from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.test_name_typo import TestNameTypo


def test_test_name_typo(index_path: Path) -> None:
    code = """import pytest

# Bad - starts with 'test' but missing underscore
def testSomething():
    assert True

# Bad - another one without underscore
def testAnother():
    assert True

# Good - properly named test
def test_valid_function():
    assert True

# Good - not a test function
def helper_function():
    return 42

# Good - starts with something else
def tset_something():
    pass
"""
    config = Config(select={TestNameTypo.name})
    violations = lint_file(Path("test_something.py"), code, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, TestNameTypo) for v in violations)
    assert violations[0].loc == Location(3, 0)
    assert violations[1].loc == Location(7, 0)
