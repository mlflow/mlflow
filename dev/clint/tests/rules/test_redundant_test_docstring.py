from pathlib import Path

from clint.config import Config
from clint.linter import Location, lint_file
from clint.rules.redundant_test_docstring import RedundantTestDocstring


def test_no_docstring_in_test_functions(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test_something.py"
    tmp_file.write_text(
        '''import pytest

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
    )

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path(tmp_file.name), tmp_file.read_text(), config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, RedundantTestDocstring) for v in violations)
    assert violations[0].loc == Location(11, 0)  # test_feature_b
    assert violations[1].loc == Location(40, 0)  # test_single_line


def test_no_docstring_in_test_classes(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test_classes.py"
    tmp_file.write_text(
        '''import pytest

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
    )

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path(tmp_file.name), tmp_file.read_text(), config, index_path)
    assert len(violations) == 1
    assert isinstance(violations[0].rule, RedundantTestDocstring)
    assert violations[0].loc == Location(12, 0)  # TestAnotherFeature class


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

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path(tmp_file.name), tmp_file.read_text(), config, index_path)
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

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path(tmp_file.name), tmp_file.read_text(), config, index_path)
    assert len(violations) == 0


def test_supports_test_suffix_files(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "module_test.py"
    tmp_file.write_text(
        '''def test_something():
    """
    This test has a multi-line docstring.
    This is allowed now.
    """
    assert True

def test_another():
    # No docstring - good
    assert True

def test_with_single_line():
    """Single line docstring should be flagged."""
    assert True
'''
    )

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path(tmp_file.name), tmp_file.read_text(), config, index_path)
    assert len(violations) == 1
    assert isinstance(violations[0].rule, RedundantTestDocstring)
    assert violations[0].loc == Location(11, 0)  # test_with_single_line


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

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path(tmp_file.name), tmp_file.read_text(), config, index_path)
    # Should have no violations - comments are fine, docstrings are not
    assert len(violations) == 0


def test_edge_cases(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test_edge_cases.py"
    tmp_file.write_text(
        '''import pytest

# Edge case 1: Empty test function with single-line docstring (bad)
def test_empty_with_docstring():
    """This empty test should still be flagged."""
    pass

# Edge case 2: Test function with only single-line docstring (bad)
def test_only_docstring():
    """This test only has a docstring."""

# Edge case 3: Nested test class
class OuterClass:
    class TestNested:
        """Nested test class with single-line docstring - should be flagged."""
        def test_nested_method(self):
            """
            Nested test method with multi-line docstring.
            This is allowed now.
            """
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

# Edge case 6: Test with multi-line docstring (good)
def test_with_multiline():
    """
    This is a multi-line docstring.
    It provides valuable information about the test.
    """
    assert True
'''
    )

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path(tmp_file.name), tmp_file.read_text(), config, index_path)
    # Should flag: test_empty_with_docstring, test_only_docstring, TestNested class
    assert len(violations) == 3
    assert violations[0].loc.lineno == 3  # test_empty_with_docstring
    assert violations[1].loc.lineno == 8  # test_only_docstring
    assert violations[2].loc.lineno == 13  # TestNested class


def test_parametrized_tests(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test_parametrized.py"
    tmp_file.write_text(
        '''import pytest

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_double_with_multiline_docstring(input, expected):
    """
    Test that doubling works correctly.
    This multi-line docstring is allowed.
    """
    assert input * 2 == expected

@pytest.mark.parametrize("value", [1, 2, 3])
def test_with_single_line_docstring(value):
    """Single line docstring should be flagged."""
    assert value > 0

@pytest.mark.parametrize("value", [1, 2, 3])
def test_without_docstring(value):
    # Good - no docstring, just a comment
    assert value > 0
'''
    )

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path(tmp_file.name), tmp_file.read_text(), config, index_path)
    assert len(violations) == 1
    assert violations[0].loc.lineno == 15  # test_with_single_line_docstring


def test_multiline_docstrings_are_allowed(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test_multiline.py"
    tmp_file.write_text(
        '''import pytest

def test_with_detailed_documentation():
    """
    This test verifies complex behavior of the system.

    It includes multiple paragraphs of documentation that explain
    the test setup, the expected behavior, and any important
    considerations for future maintainers.

    This kind of detailed documentation is valuable and should be allowed.
    """
    assert True

def test_with_params_documentation():
    """
    Test the parameter handling functionality.

    This test checks:
    - Parameter validation
    - Error handling for invalid inputs
    - Edge cases with boundary values
    """
    assert True

class TestComplexFeature:
    """
    Test suite for the complex feature module.

    This class contains comprehensive tests for all aspects
    of the complex feature, including integration tests and
    edge case scenarios.
    """

    def test_method_with_docs(self):
        """
        Verify the core method behavior.

        This method tests the primary functionality with
        various input combinations.
        """
        assert True

    def test_method_with_single_line(self):
        """This should be flagged as single-line."""
        assert True
'''
    )

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path(tmp_file.name), tmp_file.read_text(), config, index_path)
    # Should only flag the single-line docstring in test_method_with_single_line
    assert len(violations) == 1
    assert isinstance(violations[0].rule, RedundantTestDocstring)
    assert violations[0].rule.function_name == "test_method_with_single_line"


def test_error_message_content(index_path: Path, tmp_path: Path) -> None:
    tmp_file = tmp_path / "test_messages.py"
    tmp_file.write_text(
        '''def test_with_single_line():
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
    )

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path(tmp_file.name), tmp_file.read_text(), config, index_path)
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
