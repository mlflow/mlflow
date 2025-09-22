from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.redundant_test_docstring import RedundantTestDocstring


def test_no_docstring_in_test_functions(index_path: Path) -> None:
    code = '''import pytest

# Good - multi-line docstring is allowed
def test_feature_a():
    """
    This test verifies that feature A works correctly.
    It has multiple lines of documentation.
    """
    assert True

# Bad - single-line docstring
def test_feature_b():
    """Test feature B."""
    assert True

# Good - multi-line async test function with docstring
async def test_async_feature():
    """
    This async test verifies async functionality.
    With additional context provided here.
    """
    assert True

# Good - test function without docstring
def test_feature_c():
    assert True

# Good - helper function with docstring (not a test)
def helper_function():
    """
    This is a helper function, docstrings are allowed.
    """
    return 42

# Good - regular function with single-line docstring (not a test)
def setup_data():
    """Setup test data."""
    return []

# Bad - another single-line docstring
def test_single_line():
    """This is a single line docstring."""
    assert True
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("test_something.py"), code, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, RedundantTestDocstring) for v in violations)
    assert violations[0].loc == Location(11, 0)  # test_feature_b
    assert violations[1].loc == Location(40, 0)  # test_single_line


def test_no_docstring_in_test_classes(index_path: Path) -> None:
    code = '''import pytest

# Good - test class with multi-line docstring
class TestFeature:
    """
    Tests for the Feature module.
    Includes comprehensive test coverage.
    """
    def test_method(self):
        assert True

# Bad - test class with single-line docstring
class TestAnotherFeature:
    """Test another feature."""
    pass

# Good - test class without docstring
class TestNoDocstring:
    def test_method(self):
        assert True

# Good - non-test class with docstring (allowed)
class HelperClass:
    """
    This is a helper class, docstrings are allowed.
    """
    def helper_method(self):
        return 42
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("test_classes.py"), code, config, index_path)
    assert len(violations) == 1
    assert violations[0].loc == Location(12, 0)  # TestAnotherFeature class
    assert isinstance(violations[0].rule, RedundantTestDocstring)
    assert violations[0].rule.has_class_docstring is True


def test_no_docstring_in_tests_non_test_file(index_path: Path) -> None:
    code = '''# Not a test file - all docstrings allowed
def test_something():
    """This docstring should be allowed since it's not a test file."""
    assert True

class TestFeature:
    """This docstring should also be allowed."""
    pass
'''

    config = Config(select={RedundantTestDocstring.name})
    # Non-test file
    violations = lint_file(Path("regular_module.py"), code, config, index_path)
    assert len(violations) == 0


def test_conftest_exclusion(index_path: Path) -> None:
    code = '''# conftest.py is excluded from this rule
import pytest

@pytest.fixture
def test_fixture():
    """This fixture provides test data."""
    return {"data": "value"}

def test_helper():
    """Helper function for tests."""
    return 42

class TestBase:
    """Base test class."""
    pass
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("conftest.py"), code, config, index_path)
    assert len(violations) == 0


def test_supports_test_suffix_files(index_path: Path) -> None:
    code = '''import pytest

def test_feature():
    """Test feature."""
    assert True

class TestClass:
    """Test class."""
    pass
'''

    config = Config(select={RedundantTestDocstring.name})

    # Test file with _test.py suffix
    violations = lint_file(Path("module_test.py"), code, config, index_path)
    assert len(violations) == 2  # Both function and class should be flagged


def test_nb_comments_are_allowed(index_path: Path) -> None:
    code = '''def test_complex_edge_case():
    # NB: This test uses a workaround for issue #12345
    # because the standard approach fails due to timing.
    special_setup()
    result = perform_operation()
    assert result == expected

def test_another_case():
    # Note: Regular comments are fine too
    # NB: But we especially encourage NB comments for critical info
    assert True

def test_with_docstring():
    """This single-line docstring should be flagged."""
    # NB: Even with NB comments, single-line docstrings are flagged
    assert True
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("test_nb_comments.py"), code, config, index_path)
    assert len(violations) == 1
    assert violations[0].loc.lineno == 12  # Only test_with_docstring should be flagged


def test_edge_cases(index_path: Path) -> None:
    code = '''def test_empty_string_docstring():
    ""
    assert True

def test_whitespace_only():
    """   """
    assert True

def test_nested_function():
    """Test nested."""
    def inner():
        """Inner function can have docstrings."""
        pass
    assert True

class TestNested:
    """Test nested class."""
    class InnerClass:
        """Inner classes can have docstrings."""
        pass
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("test_edge.py"), code, config, index_path)
    # Note: both "" and """   """ become empty strings and are flagged as single-line
    assert len(violations) == 4
    assert violations[0].loc.lineno == 0  # test_empty_string_docstring
    assert violations[1].loc.lineno == 4  # test_whitespace_only
    assert violations[2].loc.lineno == 8  # test_nested_function
    assert violations[3].loc.lineno == 15  # TestNested class


def test_multiline_docstrings_are_allowed(index_path: Path) -> None:
    code = '''def test_with_multiline():
    """
    This is a multi-line docstring.
    It provides substantial documentation.
    """
    assert True

def test_with_multiline_compact():
    """This docstring spans
    multiple lines."""
    assert True

class TestWithMultilineDoc:
    """
    This test class has a multi-line docstring.
    With additional context here.
    """
    pass

class TestCompactMultiline:
    """This class docstring
    spans multiple lines."""
    pass
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("test_multiline.py"), code, config, index_path)
    assert len(violations) == 0  # All multiline docstrings are allowed


def test_error_message_content(index_path: Path) -> None:
    code = '''def test_with_single_line():
    """This is a single-line docstring."""
    pass

def test_with_multiline():
    """
    This is a multi-line docstring.
    It is allowed.
    """
    pass

class TestClass:
    """This test class has a single-line docstring."""
    pass
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("test_messages.py"), code, config, index_path)
    assert len(violations) == 2

    # Check error messages
    func_violation = violations[0]
    assert "test_with_single_line" in func_violation.rule.message
    assert "single-line docstring" in func_violation.rule.message
    assert "low-value" in func_violation.rule.message

    class_violation = violations[1]
    assert "TestClass" in class_violation.rule.message
    assert "Test class" in class_violation.rule.message
    assert "single-line docstring" in class_violation.rule.message
