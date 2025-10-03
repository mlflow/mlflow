from pathlib import Path

from clint.config import Config
from clint.linter import lint_file
from clint.rules.redundant_test_docstring import RedundantTestDocstring


def test_redundant_docstrings_are_flagged(index_path: Path) -> None:
    code = '''
def test_feature_a():
    """
    This test verifies that feature A works correctly.
    It has multiple lines of documentation.
    """
    assert True

def test_feature_behavior():
    """Test feature."""
    assert True

def test_c():
    """Test the complex interaction between modules."""
    assert True

def test_validation_logic():
    """Test validation."""
    assert True

def test_feature_d():
    assert True
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("test_something.py"), code, config, index_path)
    assert len(violations) == 2
    assert all(isinstance(v.rule, RedundantTestDocstring) for v in violations)


def test_docstring_word_overlap(index_path: Path) -> None:
    code = '''
def test_very_long_function_name():
    """Short."""
    assert True

def test_short():
    """This is a much longer docstring than the function name."""
    assert True

def test_data_validation():
    """Test data validation"""
    assert True

def test_multi():
    """Line 1
    Line 2"""
    assert True

def test_foo_bar_baz():
    """Test qux."""
    assert True
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("test_length.py"), code, config, index_path)
    assert len(violations) == 1


def test_class_docstrings_follow_same_rules(index_path: Path) -> None:
    code = '''
class TestFeature:
    """
    Tests for the Feature module.
    Includes comprehensive test coverage.
    """
    def test_method(self):
        assert True

class TestFeatureImplementation:
    """Test feature."""
    pass

class TestShort:
    """This is a longer docstring than the class name TestShort."""
    pass
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("test_classes.py"), code, config, index_path)
    assert len(violations) == 1


def test_non_test_files_are_ignored(index_path: Path) -> None:
    code = '''
def test_something():
    """Short."""
    assert True

class TestFeature:
    """Test."""
    pass
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("regular_module.py"), code, config, index_path)
    assert len(violations) == 0


def test_supports_test_suffix_files(index_path: Path) -> None:
    code = '''
def test_feature_implementation():
    """Test feature."""
    assert True

class TestClassImplementation:
    """Test class."""
    pass
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("module_test.py"), code, config, index_path)
    assert len(violations) == 2


def test_multiline_docstrings_are_always_allowed(index_path: Path) -> None:
    code = '''def test_with_multiline():
    """
    Multi-line.
    """
    assert True

def test_with_multiline_compact():
    """Line 1
    Line 2"""
    assert True

class TestWithMultilineDoc:
    """
    Multi
    Line
    """
    pass

class TestCompactMultiline:
    """Line1
    Line2"""
    pass
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("test_multiline.py"), code, config, index_path)
    assert len(violations) == 0


def test_error_message_content(index_path: Path) -> None:
    code = '''def test_data_processing_validation():
    """Test data processing."""
    pass

class TestDataProcessingValidation:
    """Test data processing."""
    pass
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("test_messages.py"), code, config, index_path)
    assert len(violations) == 2

    func_violation = violations[0]
    assert "test_data_processing_validation" in func_violation.rule.message
    assert "redundant docstring" in func_violation.rule.message
    assert "don't add value" in func_violation.rule.message

    class_violation = violations[1]
    assert "TestDataProcessingValidation" in class_violation.rule.message
    assert "Test class" in class_violation.rule.message
    assert "Consider removing it or expanding it" in class_violation.rule.message


def test_module_single_line_docstrings_are_flagged(index_path: Path) -> None:
    code = '''"""This is a test module."""
def test_something():
    assert True
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("test_module.py"), code, config, index_path)
    assert len(violations) == 1
    assert isinstance(violations[0].rule, RedundantTestDocstring)
    assert violations[0].rule.is_module_docstring
    assert "single-line docstring" in violations[0].rule.message


def test_module_multiline_docstrings_are_allowed(index_path: Path) -> None:
    code = '''"""
This is a test module.
It has multiple lines.
"""
def test_something():
    assert True
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("test_module.py"), code, config, index_path)
    assert len(violations) == 0


def test_module_without_docstring_is_not_flagged(index_path: Path) -> None:
    code = """def test_something():
    assert True
"""

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("test_module.py"), code, config, index_path)
    assert len(violations) == 0


def test_non_test_module_docstrings_are_ignored(index_path: Path) -> None:
    code = '''"""This is a regular module."""
def some_function():
    pass
'''

    config = Config(select={RedundantTestDocstring.name})
    violations = lint_file(Path("regular_module.py"), code, config, index_path)
    assert len(violations) == 0
