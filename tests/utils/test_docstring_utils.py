from mlflow.utils.docstring_utils import ParamDocs, _get_minimum_indentation, format_docstring


def test_param_docs_format():
    pd = ParamDocs({"x": "{{ x }}", "y": "{{ y }}", "z": "{{ x }}, {{ y }}"})
    formatted = pd.format(x="a", y="b")
    assert isinstance(formatted, ParamDocs)
    assert formatted == {"x": "a", "y": "b", "z": "a, b"}


def test_get_minimum_indentation():
    text = """
    # 4 spaces
      # 6 spaces
        # 8 spaces
"""
    assert _get_minimum_indentation(text) == " " * 4

    text = """
# no indent
"""
    assert _get_minimum_indentation(text) == ""
    assert _get_minimum_indentation("") == ""


def test_format_docstring():
    # pylint: disable=W
    @format_docstring({"p": "param doc"})
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

    @format_docstring({"p1": "param1 doc", "p2": "param2 doc"})
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
