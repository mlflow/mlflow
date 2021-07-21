import textwrap
import re
from functools import reduce


def _create_placeholder(key):
    return "{{ " + key + " }}"


def _replace_placeholder(template, key, value):
    placeholder = _create_placeholder(key)
    return template.replace(placeholder, value)


class ParamDocs(dict):
    """
    Represents a set of parameter documents.
    """

    def __repr__(self):
        return f"ParamDocs({super().__repr__()})"

    def format(self, **kwargs):
        """
        Replaces placeholders in param docs.

        :param kwargs: A `dict` in the form of `{"< placeholder name >": "< value >"}`.
        :return: A new `ParamDocs` instance with the formatted param docs.

        Examples
        --------
        >>> p = ParamDocs(p1="{{ doc1 }}", p2="{{ doc2 }}")
        >>> p.format(doc1="foo", doc2="bar")
        ParamDocs({'p1': 'foo', 'p2': 'bar'})
        """
        new_param_docs = {}
        for param_name, param_doc in self.items():
            for key, value in kwargs.items():
                param_doc = _replace_placeholder(param_doc, key, value)
            new_param_docs[param_name] = param_doc

        return ParamDocs(new_param_docs)

    def format_docstring(self, docstring):
        """
        Replaces param doc placeholders in `docstring`.

        :param docstring: Docstring to format.
        :return: Formatted docstring.

        Examples
        --------
        >>> p = ParamDocs(p1="doc1", p2="doc2")
        >>> docstring = '''
        ... :param p1: {{ p1 }}
        ... :param p2: {{ p2 }}
        ... '''.strip()
        >>> print(p.format_docstring(docstring))
        :param p1:
            doc1
        :param p2:
            doc2
        """
        min_indent = _get_minimum_indentation(docstring)
        for param_name, param_doc in self.items():
            param_doc = textwrap.indent(param_doc, min_indent + " " * 4)
            if not param_doc.startswith("\n"):
                param_doc = "\n" + param_doc
            docstring = _replace_placeholder(docstring, param_name, param_doc)

        return docstring


LOG_MODEL_PARAM_DOCS = ParamDocs(
    {
        "pip_requirements": """
Either an iterable of pip requirement strings
(e.g. ``["{{ package_name }}", "-r requirements.txt", "-c constrants.txt"]``) or the string path to
a pip requirements file on the local filesystem (e.g. ``"requirements.txt"``). If provided, this
describes the environment this model should be run in. If ``None``, a default list of requirements
is inferred from the current software environment. Both requirements and constraints are
automatically parsed and written to ``requirements.txt`` and ``constraints.txt`` files,
respectively, and stored as part of the model. Requirements are also written to the ``pip`` section
of the model's conda environment (``conda.yaml``) file.
""",
        "extra_pip_requirements": """
Either an iterable of pip requirement strings
(e.g. ``["{{ package_name }}", "-r requirements.txt", "-c constrants.txt"]``) or the string path to
a pip requirements file on the local filesystem (e.g. ``"requirements.txt"``). If provided, this
describes additional pip requirements that are appended to a default set of pip requirements
generated automatically based on the user's current software environment. Both requirements and
constraints are automatically parsed and written to ``requirements.txt`` and ``constraints.txt``
files, respectively, and stored as part of the model. Requirements are also written to the ``pip``
section of the model's conda environment (``conda.yaml``) file.

.. warning::
    The following arguments can't be specified at the same time:

    - ``conda_env``
    - ``pip_requirements``
    - ``extra_pip_requirements``

:ref:`This example<pip-requirements-example>` demonstrates how to specify pip requirements using
``pip_requirements`` and ``extra_pip_requirements``.
""",
    }
)


_leading_whitespace_re = re.compile("(^[ ]*)(?:[^ \n])", re.MULTILINE)


def _get_minimum_indentation(text):
    """
    Returns the minimum indentation of all non-blank lines in `text`.
    """
    indents = _leading_whitespace_re.findall(text)
    return min(indents, key=len) if indents else ""


def format_docstring(param_docs):
    """
    Returns a decorator that replaces param doc placeholders (e.g. '{{ param_name }}') in the
    docstring of the decorated function.

    :param param_docs: A `ParamDocs` instance or `dict`.
    :return: A decorator to apply the formatting.

    Examples
    --------
    >>> param_docs = {"p1": "doc1", "p2": "doc2"}
    >>> @format_docstring(param_docs)
    ... def func(p1, p2):
    ...     '''
    ...     :param p1: {{ p1 }}
    ...     :param p2: {{ p2 }}
    ...     '''
    >>> import textwrap
    >>> print(textwrap.dedent(func.__doc__).strip())
    :param p1:
        doc1
    :param p2:
        doc2
    """
    param_docs = ParamDocs(param_docs)

    def decorator(func):
        func.__doc__ = param_docs.format_docstring(func.__doc__)
        return func

    return decorator
