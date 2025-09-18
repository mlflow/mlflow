from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.no_docstring_in_tests import NoDocstringInTests


def test_no_docstring_in_test_functions(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test_something.py"
    tmp_file.write_text(
        '''import pytest

# Bad - test function with docstring
def test_feature_a():
    """
    This test verifies that feature A works correctly.
    """
    assert True

# Bad - another test function with docstring
def test_feature_b():
    """Test feature B."""
    assert True

# Bad - async test function with docstring
async def test_async_feature():
    """
    This async test verifies async functionality.
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

# Good - regular function with docstring
def setup_data():
    """Setup test data."""
    return []
'''
    )

    config = Config(select={NoDocstringInTests.name})
    violations = lint_file(tmp_file, tmp_file.read_text(), config, index_path)
    assert len(violations) == 3
    assert all(isinstance(v.rule, NoDocstringInTests) for v in violations)
    assert violations[0].loc == Location(3, 0)  # test_feature_a
    assert violations[1].loc == Location(10, 0)  # test_feature_b
    assert violations[2].loc == Location(15, 0)  # test_async_feature


def test_no_docstring_in_test_classes(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test_classes.py"
    tmp_file.write_text(
        '''import pytest

# Bad - test class with docstring
class TestFeature:
    """
    Tests for the Feature module.
    """
    def test_method(self):
        assert True

# Bad - another test class with docstring
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
    )

    config = Config(select={NoDocstringInTests.name})
    violations = lint_file(tmp_file, tmp_file.read_text(), config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, NoDocstringInTests) for v in violations)
    assert violations[0].loc == Location(3, 0)  # TestFeature class
    assert violations[1].loc == Location(11, 0)  # TestAnotherFeature class


def test_no_docstring_in_tests_non_test_file(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "regular_file.py"
    tmp_file.write_text(
        '''def test_something():
    """
    This function is in a non-test file, so docstrings are allowed.
    """
    return True

class TestClass:
    """
    This class is in a non-test file, so docstrings are allowed.
    """
    pass
'''
    )

    config = Config(select={NoDocstringInTests.name})
    violations = lint_file(tmp_file, tmp_file.read_text(), config, index_path)
    assert len(violations) == 0


def test_conftest_exclusion(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "conftest.py"
    tmp_file.write_text(
        '''import pytest

@pytest.fixture
def test_fixture():
    """
    Fixtures in conftest.py often need docstrings for documentation.
    """
    return "fixture_value"

def test_helper():
    """
    Helper functions in conftest.py may also have docstrings.
    """
    return 42
'''
    )

    config = Config(select={NoDocstringInTests.name})
    violations = lint_file(tmp_file, tmp_file.read_text(), config, index_path)
    assert len(violations) == 0


def test_supports_test_suffix_files(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "module_test.py"
    tmp_file.write_text(
        '''def test_something():
    """
    This test has a docstring that should be detected.
    """
    assert True

def test_another():
    # No docstring - good
    assert True
'''
    )

    config = Config(select={NoDocstringInTests.name})
    violations = lint_file(tmp_file, tmp_file.read_text(), config, index_path)
    assert len(violations) == 1
    assert isinstance(violations[0].rule, NoDocstringInTests)
    assert violations[0].loc == Location(0, 0)  # test_something


def test_nb_comments_are_allowed(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test_with_comments.py"
    tmp_file.write_text(
        """import pytest

def test_complex_scenario():
    # NB: This test uses a specific workaround for issue #12345
    # The standard approach doesn't work because of X, Y, Z reasons
    # so we need to do this unusual setup
    special_setup = create_special_setup()

    # Regular comment - this is just explaining the test flow
    result = perform_action(special_setup)

    # NB: We expect 42 here, not 41, because of the special rounding
    # behavior documented in RFC-789. This is intentional and correct.
    assert result == 42

def test_another_case():
    # This test doesn't have a docstring, which is good
    # NB: Using mock.patch here because real API calls would be too slow
    with mock.patch('external.api') as mock_api:
        mock_api.return_value = "expected"
        assert process() == "expected"
"""
    )

    config = Config(select={NoDocstringInTests.name})
    violations = lint_file(tmp_file, tmp_file.read_text(), config, index_path)
    # Should have no violations - comments are fine, docstrings are not
    assert len(violations) == 0


def test_edge_cases(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test_edge_cases.py"
    tmp_file.write_text(
        '''import pytest

# Edge case 1: Empty test function with docstring (still bad)
def test_empty_with_docstring():
    """This empty test should still be flagged."""
    pass

# Edge case 2: Test function with only docstring (bad)
def test_only_docstring():
    """This test only has a docstring."""

# Edge case 3: Nested test class
class OuterClass:
    class TestNested:
        """Nested test class with docstring - should be flagged."""
        def test_nested_method(self):
            """Nested test method with docstring - should be flagged."""
            pass

# Edge case 4: Test with multiline string (not a docstring)
def test_multiline_string():
    multiline = """
    This is not a docstring, it's assigned to a variable.
    """
    assert multiline is not None

# Edge case 5: Test with f-string docstring-like content
def test_formatted_string():
    # This is fine - no docstring
    expected = f"""Expected output:
    - Item 1
    - Item 2"""
    assert expected
'''
    )

    config = Config(select={NoDocstringInTests.name})
    violations = lint_file(tmp_file, tmp_file.read_text(), config, index_path)
    # Should flag: test_empty_with_docstring, test_only_docstring,
    # TestNested class, test_nested_method
    assert len(violations) == 4
    assert violations[0].loc.lineno == 3  # test_empty_with_docstring
    assert violations[1].loc.lineno == 8  # test_only_docstring
    assert violations[2].loc.lineno == 13  # TestNested class
    assert violations[3].loc.lineno == 15  # test_nested_method


def test_parametrized_tests(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test_parametrized.py"
    tmp_file.write_text(
        '''import pytest

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double_with_docstring(input, expected):
    """
    Test that doubling works correctly.
    This docstring should be flagged.
    """
    assert input * 2 == expected

@pytest.mark.parametrize("value", [1, 2, 3])
def test_without_docstring(value):
    # Good - no docstring, just a comment
    assert value > 0
'''
    )

    config = Config(select={NoDocstringInTests.name})
    violations = lint_file(tmp_file, tmp_file.read_text(), config, index_path)
    assert len(violations) == 1
    assert violations[0].loc.lineno == 7  # test_double_with_docstring


def test_error_message_content(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test_messages.py"
    tmp_file.write_text(
        '''def test_with_docstring():
    """This is a docstring that should not be here."""
    pass

class TestClass:
    """This test class has a docstring."""
    pass
'''
    )

    config = Config(select={NoDocstringInTests.name})
    violations = lint_file(tmp_file, tmp_file.read_text(), config, index_path)
    assert len(violations) == 2

    # Check error messages
    func_violation = violations[0]
    assert "test_with_docstring" in func_violation.rule.message
    assert "should not have a docstring" in func_violation.rule.message
    assert "self-documenting" in func_violation.rule.message

    class_violation = violations[1]
    assert "TestClass" in class_violation.rule.message
    assert "Test class" in class_violation.rule.message
