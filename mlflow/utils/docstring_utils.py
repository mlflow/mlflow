import textwrap
import re


class ParamDocs(dict):
    def __repr__(self):
        return f"ParamDocs({super().__repr__()})"

    @classmethod
    def _create_placeholder(cls, key):
        return "{{ " + key + " }}"

    @classmethod
    def _fill_placeholder(cls, template, key, value):
        placeholder = ParamDocs._create_placeholder(key)
        assert placeholder in template
        return template.replace(placeholder, value)

    def format(self, **kwargs):
        return ParamDocs(
            {
                param_name: ParamDocs._fill_placeholder(param_doc, key, value)
                for param_name, param_doc in self.items()
                for key, value in kwargs.items()
            }
        )

    def format_docstring(self, docstring):
        min_indent = _get_minimum_indentation(docstring)
        for param_name, param_doc in self.items():
            param_doc = textwrap.indent(param_doc, min_indent + " " * 4)
            if not param_doc.startswith("\n"):
                param_doc = "\n" + param_doc
            docstring = ParamDocs._fill_placeholder(docstring, param_name, param_doc)

        return docstring


LOG_MODEL_PARAM_DOCS = ParamDocs(
    {
        "pip_requirements": """
Either an iterable of pip requirement strings
(e.g. ``["{{ package_name }}", "-r requirements.txt"]``) or the string path to a pip requirements
file on the local filesystem (e.g. ``"requirements.txt"``). If provided, this describes the
environment this model should be run in. If ``None``, a default list of requirements is
inferred from the current software environment. Requirements are automatically parsed and
written to a ``requirements.txt`` file that is stored as part of the model. These
requirements are also written to the ``pip`` section of the model's conda environment
(``conda.yaml``) file.
""",
        "extra_pip_requirements": """
Either an iterable of pip requirement strings
(e.g. ``["{{ package_name }}", "-r requirements.txt"]``) or the string path to a pip requirements
file on the local filesystem (e.g. ``"requirements.txt"``). If provided, this specifies
additional pip requirements that are appended to a default set of pip requirements generated
automatically based on the user's current software environment. Requirements are also
written to the ``pip`` section of the model's conda environment (``conda.yaml``) file.

.. warning::
    The following arguments can't be specified at the same time:

    - ``conda_env``
    - ``pip_requirements``
    - ``extra_pip_requirements``

:ref:`This example<pip-requirements-example>` demonstrates how to specify pip requirements
using ``pip_requirements`` and ``extra_pip_requirements``.
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


def _format_param_docs(param_docs):
    """
    Returns a decorator that replaces param document placeholders (e.g. '{{ param_name }}') in the
    docstring of the decorated function.

    :param param_docs: A dict in the form of `{"param_name": "param_doc"}`.

    Examples
    --------
    >>> param_docs = {"foo": "bar"}
    >>> @_format_param_docs(param_docs)
    ... def func(foo):
    ...     '''
    ...     =====================
    ...     :param foo: {{ foo }}
    ...     =====================
    ...     '''
    >>> print(func.__doc__)

        =====================
        :param foo:
            bar
        =====================
    """
    param_docs = ParamDocs(param_docs)

    def decorator(func):
        func.__doc__ = param_docs.format_docstring(func.__doc__)
        return func

    return decorator
