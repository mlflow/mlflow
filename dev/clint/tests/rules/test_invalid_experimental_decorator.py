from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.invalid_experimental_decorator import InvalidExperimentalDecorator


def test_invalid_experimental_decorator(index_path: Path) -> None:
    code = """
from mlflow.utils.annotations import experimental

# Bad - no arguments
@experimental
def bad_function1():
    pass

# Bad - no version argument
@experimental()
def bad_function2():
    pass

# Bad - invalid version format
@experimental(version="invalid")
def bad_function3():
    pass

# Bad - pre-release version
@experimental(version="1.0.0rc1")
def bad_function4():
    pass

# Bad - non-string version
@experimental(version=123)
def bad_function5():
    pass

# Good - valid semantic version
@experimental(version="1.2.3")
def good_function1():
    pass

# Good - valid semantic version with multiple parts
@experimental(version="2.0.0")
def good_function2():
    pass
"""
    config = Config(select={InvalidExperimentalDecorator.name})
    violations = lint_file(Path("test.py"), code, config, index_path)
    assert len(violations) == 5
    assert all(isinstance(v.rule, InvalidExperimentalDecorator) for v in violations)
    assert violations[0].loc == Location(4, 1)  # @experimental without args
    assert violations[1].loc == Location(9, 1)  # @experimental() without version
    assert violations[2].loc == Location(14, 1)  # invalid version format
    assert violations[3].loc == Location(19, 1)  # pre-release version
    assert violations[4].loc == Location(24, 1)  # non-string version
