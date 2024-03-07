import re

import pytest

from mlflow.utils.annotations import _get_min_indent_of_docstring, deprecated, keyword_only


class MyClass:
    @deprecated()
    def method(self):
        """
        Returns 0
        """
        return 0


@deprecated()
def function():
    """
    Returns 1
    """
    return 1


@keyword_only
@deprecated(since="0.0.0")
def deprecated_and_keyword_only_first():
    """Description

    Args:
        x: x

    Returns:
        y
    """
    return 1


@deprecated(since="0.0.0")
@keyword_only
def deprecated_and_keyword_only_second():
    """
    Description

    Args:
        x: x

    Returns:
        y
    """
    return 1


def test_deprecated_method():
    msg = "``tests.utils.test_annotations.MyClass.method`` is deprecated"
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        assert MyClass().method() == 0
    assert msg in MyClass.method.__doc__


def test_deprecated_function():
    msg = "``tests.utils.test_annotations.function`` is deprecated"
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        assert function() == 1
    assert msg in function.__doc__


def test_empty_docstring():
    docstring = ""
    expected_indent = ""
    assert _get_min_indent_of_docstring(docstring) == expected_indent


def test_single_line_docstring():
    docstring = """Single line with indent."""
    expected_indent = ""
    assert _get_min_indent_of_docstring(docstring) == expected_indent


def test_multi_line_docstring_first_line():
    first_line_docstring = """Description

    Args:
        x: x

    Returns:
        y
    """
    expected_indent = "    "
    assert _get_min_indent_of_docstring(first_line_docstring) == expected_indent


def test_multi_line_docstring_second_line():
    second_line_docstring = """
    Description

    Args:
        x: x

    Returns:
        y
    """
    expected_indent = "    "
    assert _get_min_indent_of_docstring(second_line_docstring) == expected_indent


def test_deprecated_and_keyword_first():
    docstring = deprecated_and_keyword_only_first.__doc__
    assert docstring.rstrip() == (
        """    .. note:: This method requires all argument be specified by keyword.
    .. Warning:: ``tests.utils.test_annotations.deprecated_and_keyword_only_first`` is deprecated since 0.0.0. This method will be removed in a future release.
Description

    Args:
        x: x

    Returns:
        y"""  # noqa: E501
    )


def test_deprecated_and_keyword_second():
    docstring = deprecated_and_keyword_only_second.__doc__
    assert docstring.rstrip() == (
        """    .. Warning:: ``tests.utils.test_annotations.deprecated_and_keyword_only_second`` is deprecated since 0.0.0. This method will be removed in a future release.
    .. note:: This method requires all argument be specified by keyword.

    Description

    Args:
        x: x

    Returns:
        y"""  # noqa: E501
    )
