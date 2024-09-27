import warnings

from mlflow.utils.docstring_utils import (
    ParamDocs,
    _indent,
    docstring_version_compatibility_warning,
    format_docstring,
)


def test_indent_empty():
    a, b = "", " " * 4
    assert _indent(a, b) == a


def test_indent_single_line():
    a, b = "x", " " * 4
    assert _indent(a, b) == a


def test_indent_multi_line():
    a = """x\nx
    x\nx
    x"""
    b = " " * 4
    assert _indent(a, b) == "x\n    x\n        x\n    x\n        x"


def test_param_docs_format():
    pd = ParamDocs({"x": "{{ x }}", "y": "{{ y }}", "z": "{{ x }}, {{ y }}"})
    formatted = pd.format(x="a", y="b")
    assert isinstance(formatted, ParamDocs)
    assert formatted == {"x": "a", "y": "b", "z": "a, b"}


def test_param_docs_format_no_changes():
    @format_docstring(
        {
            "multi_line": """Single line
Another line\n    Another indented line""",
            "single_line": "hi",
        }
    )
    def f():
        """asdf

        Args:
            p1:
                asdf
            p2: asdf
            p3:
                asdf
            p4:
                asdf
        """

    expected = """asdf

        Args:
            p1:
                asdf
            p2: asdf
            p3:
                asdf
            p4:
                asdf
        """

    assert f.__doc__ == expected
    assert f.__name__ == "f"


def test_param_docs_format_google():
    @format_docstring(
        {
            "multi_line": """Single line
Another line\n    Another indented line""",
            "single_line": "hi",
        }
    )
    # fmt: off
    def f():
        """asdf

        Args:
            p1:
                asdf
            p2: {{ multi_line }}
            p3:
                {{ single_line }}
            p4:
                {{ multi_line }}
        """

    expected = """asdf

        Args:
            p1:
                asdf
            p2: Single line
                Another line
                    Another indented line
            p3:
                hi
            p4:
                Single line
                Another line
                    Another indented line
        """
    # fmt: on

    assert f.__doc__ == expected
    assert f.__name__ == "f"


def test_param_docs_format_not_google():
    @format_docstring(
        {
            "multi_line": """Single line
Another line\n    Another indented line""",
            "single_line": "hi",
        }
    )
    # fmt: off
    def f():
        """
        asdf

        Args
            p1: asdf
            p2: {{ multi_line }}
            p3: {{ single_line }}
            p4:
                {{ multi_line }}
        """

    expected = """
        asdf

        Args
            p1: asdf
            p2: Single line
                Another line
                    Another indented line
            p3: hi
            p4:
                Single line
                Another line
                    Another indented line
        """
    # fmt: on

    assert f.__doc__ == expected
    assert f.__name__ == "f"


def test_docstring_version_compatibility_warning():
    @docstring_version_compatibility_warning("sklearn")
    def func():
        pass

    @docstring_version_compatibility_warning("sklearn")
    def another_func():
        func()

    with warnings.catch_warnings(record=True) as w:
        func()

    # Exclude irrelevant warnings
    warns = [x for x in w if "MLflow Models integration is known to be compatible" in str(x)]
    assert len(warns) == 0

    with warnings.catch_warnings(record=True) as w:
        another_func()

    warns = [x for x in w if "MLflow Models integration is known to be compatible" in str(x)]
    assert len(warns) == 0
