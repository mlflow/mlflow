from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.no_class_based_tests import NoClassBasedTests


def test_no_class_based_tests(index_path: Path) -> None:
    code = """import pytest

# Bad - class-based test with test methods
class TestSomething:
    def test_feature_a(self):
        assert True

    def test_feature_b(self):
        assert True

    def helper_method(self):
        return 42

# Bad - another class-based test
class TestAnotherThing:
    def test_something(self):
        pass

# Good - class without test methods (utility class)
class HelperClass:
    def helper_function(self):
        return 42

    def setup_something(self):
        pass

    def test_something(self):
        pass

# Good - function-based test
def test_valid_function():
    assert True

# Good - regular function
def helper_function():
    return 42
"""
    config = Config(select={NoClassBasedTests.name})
    violations = lint_file(Path("test_something.py"), code, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, NoClassBasedTests) for v in violations)
    assert violations[0].loc == Location(3, 0)  # TestSomething class
    assert violations[1].loc == Location(14, 0)  # TestAnotherThing class


def test_no_class_based_tests_non_test_file(index_path: Path) -> None:
    """Test that the rule doesn't apply to non-test files"""
    code = """import pytest

# This should not be flagged because it's not in a test file
class TestSomething:
    def test_feature_a(self):
        assert True
"""
    config = Config(select={NoClassBasedTests.name})
    violations = lint_file(Path("regular_file.py"), code, config, index_path)
    assert len(violations) == 0
