import pytest

from mlflow.utils.docstring_utils import _get_minimum_indentation, _format_param_docs


def test_get_minimum_indentation():
    assert (
        _get_minimum_indentation(
            """
    # 4 spaces
      # 6 spaces
        # 8 spaces
"""
        )
        == " " * 4
    )

    assert (
        _get_minimum_indentation(
            """
# no indent
"""
        )
        == ""
    )

    assert _get_minimum_indentation("") == ""


def test_format_param_docs():
    # pylint: disable=W
    @_format_param_docs({"p": "param doc"})
    def single_param(p):
        """
        :param p:{{ p }}
        """

    expected_doc = """
        :param p:
            param doc
        """

    assert single_param.__doc__ == expected_doc
    assert single_param.__name__ == "single_param"

    @_format_param_docs({"p1": "param1 doc", "p2": "param2 doc"})
    def multiple_params(p1, p2):
        """
        :param p1:{{ p1 }}
        :param p2:{{ p2 }}
        """

    expected_doc = """
        :param p1:
            param1 doc
        :param p2:
            param2 doc
        """

    assert multiple_params.__doc__ == expected_doc
    assert multiple_params.__name__ == "multiple_params"

    with pytest.raises(AssertionError):

        @_format_param_docs({"p": "param doc"})
        def no_placeholder(p):
            """
            :param p: param doc
            """
