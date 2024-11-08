import re
from dataclasses import dataclass, fields

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
def deprecated_and_keyword_only_first(x):
    """Description

    Args:
        x: x

    Returns:
        y
    """
    return 1


@deprecated(since="0.0.0")
@keyword_only
def deprecated_and_keyword_only_second(x):
    """
    Description

    Args:
        x: x

    Returns:
        y
    """
    return 1


@deprecated()
class DeprecatedClass:
    """
    A deprecated class.
    """

    def __init__(self):
        pass

    def greet(self):
        """
        Greets the user.
        """
        return "Hello"


@deprecated(since="1.0.0")
@dataclass
class DeprecatedDataClass:
    """
    A deprecated dataclass.
    """

    x: int
    y: int

    def add(self):
        return self.x + self.y


@deprecated(since="1.0.0")
@dataclass
class AnotherDeprecatedDataClass:
    a: int
    b: int

    def add(self):
        return self.a + self.b


@deprecated()
@dataclass
class AnotherDeprecatedDataClassOrder:
    """
    A deprecated dataclass with decorators in different order.
    """

    m: int
    n: int


def test_deprecated_method():
    msg = "``tests.utils.test_annotations.MyClass.method`` is deprecated"
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        assert MyClass().method() == 0
    docstring = MyClass.method.__doc__
    assert docstring is not None
    assert msg in docstring


def test_deprecated_function():
    msg = "``tests.utils.test_annotations.function`` is deprecated"
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        assert function() == 1
    docstring = function.__doc__
    assert docstring is not None
    assert msg in docstring


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
    assert docstring is not None
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
    assert docstring is not None
    assert docstring.rstrip() == (
        """    .. Warning:: ``tests.utils.test_annotations.deprecated_and_keyword_only_second`` is deprecated since 0.0.0. This method will be removed in a future release.
    .. note:: This method requires all argument be specified by keyword.

    Description

    Args:
        x: x

    Returns:
        y"""  # noqa: E501
    )


def test_deprecated_class():
    msg = "``tests.utils.test_annotations.DeprecatedClass`` is deprecated"
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        DeprecatedClass()
    docstring = DeprecatedClass.__doc__
    assert docstring is not None
    assert msg in docstring


def test_deprecated_class_method():
    msg = "``tests.utils.test_annotations.DeprecatedClass`` is deprecated"
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        instance = DeprecatedClass()
    assert instance.greet() == "Hello"
    docstring = DeprecatedClass.__doc__
    assert docstring is not None
    assert msg in docstring


def test_deprecated_dataclass():
    msg = "``tests.utils.test_annotations.DeprecatedDataClass`` is deprecated since 1.0.0"
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        DeprecatedDataClass(x=10, y=20)
    docstring = DeprecatedDataClass.__doc__
    assert docstring is not None
    assert msg in docstring


def test_deprecated_dataclass_fields():
    msg = "``tests.utils.test_annotations.DeprecatedDataClass`` is deprecated since 1.0.0"
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        instance = DeprecatedDataClass(x=5, y=15)
    assert instance.x == 5
    assert instance.y == 15
    docstring = DeprecatedDataClass.__doc__
    assert docstring is not None
    assert msg in docstring


def test_deprecated_dataclass_method():
    msg = "``tests.utils.test_annotations.AnotherDeprecatedDataClass`` is deprecated since 1.0.0"
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        instance = AnotherDeprecatedDataClass(a=3, b=4)
    assert instance.add() == 7
    docstring = AnotherDeprecatedDataClass.__doc__
    assert docstring is not None
    assert msg in docstring


def test_deprecated_dataclass_different_order():
    msg = "``tests.utils.test_annotations.AnotherDeprecatedDataClassOrder`` is deprecated"
    with pytest.warns(FutureWarning, match=re.escape(msg)):
        AnotherDeprecatedDataClassOrder(m=7, n=8)
    docstring = AnotherDeprecatedDataClassOrder.__doc__
    assert docstring is not None
    assert msg in docstring


def test_deprecated_dataclass_dunder_methods():
    instance = DeprecatedDataClass(x=1, y=2)

    assert instance.x == 1
    assert instance.y == 2

    expected_repr = "DeprecatedDataClass(x=1, y=2)"
    assert repr(instance) == expected_repr

    instance2 = DeprecatedDataClass(x=1, y=2)
    instance3 = DeprecatedDataClass(x=2, y=3)
    assert instance == instance2
    assert instance != instance3


def test_deprecated_dataclass_preserves_fields():
    instance = DeprecatedDataClass(x=100, y=200)
    field_names = {f.name for f in fields(DeprecatedDataClass)}
    assert field_names == {"x", "y"}
    assert instance.x == 100
    assert instance.y == 200


def test_deprecated_dataclass_preserves_methods():
    instance = DeprecatedDataClass(x=10, y=20)
    assert instance.add() == 30


def test_deprecated_dataclass_preserves_class_attributes():
    assert DeprecatedDataClass.__module__ == "tests.utils.test_annotations"
    assert DeprecatedDataClass.__qualname__ == "DeprecatedDataClass"


def test_deprecated_dataclass_dunder_methods_not_mutated():
    instance = DeprecatedDataClass(x=5, y=10)
    assert instance.x == 5
    assert instance.y == 10

    expected_repr = "DeprecatedDataClass(x=5, y=10)"
    assert repr(instance) == expected_repr

    same_instance = DeprecatedDataClass(x=5, y=10)
    different_instance = DeprecatedDataClass(x=1, y=2)
    assert instance == same_instance
    assert instance != different_instance

    assert instance.add() == 15

    allowed_attrs = {"x", "y", "add"}
    attrs = {attr for attr in dir(instance) if not attr.startswith("__")}
    assert attrs == allowed_attrs


def test_deprecated_dataclass_special_methods_integrity():
    instance = DeprecatedDataClass(x=42, y=84)

    assert instance.x == 42
    assert instance.y == 84

    expected_repr = "DeprecatedDataClass(x=42, y=84)"
    assert repr(instance) == expected_repr

    same_instance = DeprecatedDataClass(x=42, y=84)
    different_instance = DeprecatedDataClass(x=1, y=2)
    assert instance == same_instance
    assert instance != different_instance

    assert instance.add() == 126

    allowed_attrs = {"x", "y", "add"}
    attrs = {attr for attr in dir(instance) if not attr.startswith("__")}
    assert attrs == allowed_attrs
