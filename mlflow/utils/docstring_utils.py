import re
import textwrap
import warnings
from functools import wraps

import importlib_metadata
from packaging.version import Version

from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.autologging_utils.versioning import (
    FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY,
)


def _create_placeholder(key):
    return "{{ " + key + " }}"


def _replace_placeholder(template, key, value):
    placeholder = _create_placeholder(key)
    return template.replace(placeholder, value)


class ParamDocs(dict):
    """
    Represents a set of parameter documents in the docstring.
    """

    def __repr__(self):
        return f"ParamDocs({super().__repr__()})"

    def format(self, **kwargs):
        """
        Formats placeholders in this instance with `kwargs`.

        :param kwargs: A `dict` in the form of `{"< placeholder name >": "< value >"}`.
        :return: A new `ParamDocs` instance with the formatted param docs.

        Examples
        --------
        >>> pd = ParamDocs(p1="{{ doc1 }}", p2="{{ doc2 }}")
        >>> pd.format(doc1="foo", doc2="bar")
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
        Formats placeholders in `docstring`.

        :param docstring: Docstring to format.
        :return: Formatted docstring.

        Examples
        --------
        >>> pd = ParamDocs(p1="doc1", p2="doc2")
        >>> docstring = '''
        ... :param p1: {{ p1 }}
        ... :param p2: {{ p2 }}
        ... '''.strip()
        >>> print(pd.format_docstring(docstring))
        :param p1:
            doc1
        :param p2:
            doc2
        """
        if docstring is None:
            return None

        min_indent = _get_minimum_indentation(docstring)
        for param_name, param_doc in self.items():
            param_doc = textwrap.indent(param_doc, min_indent + " " * 4)
            if not param_doc.startswith("\n"):
                param_doc = "\n" + param_doc
            docstring = _replace_placeholder(docstring, param_name, param_doc)

        return docstring


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


# `{{ ... }}` represents a placeholder.
LOG_MODEL_PARAM_DOCS = ParamDocs(
    {
        "conda_env": """
Either a dictionary representation of a Conda environment or the path to a conda environment yaml
file. If provided, this describes the environment this model should be run in. At minimum, it
should specify the dependencies contained in :func:`get_default_conda_env()`. If ``None``, a conda
environment with pip requirements inferred by :func:`mlflow.models.infer_pip_requirements` is added
to the model. If the requirement inference fails, it falls back to using
:func:`get_default_pip_requirements`. pip requirements from ``conda_env`` are written to a pip
``requirements.txt`` file and the full conda environment is written to ``conda.yaml``.
The following is an *example* dictionary representation of a conda environment::

    {
        "name": "mlflow-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.8.15",
            {
                "pip": [
                    "{{ package_name }}==x.y.z"
                ],
            },
        ],
    }
            """,
        "pip_requirements": """
Either an iterable of pip requirement strings
(e.g. ``["{{ package_name }}", "-r requirements.txt", "-c constraints.txt"]``) or the string path to
a pip requirements file on the local filesystem (e.g. ``"requirements.txt"``). If provided, this
describes the environment this model should be run in. If ``None``, a default list of requirements
is inferred by :func:`mlflow.models.infer_pip_requirements` from the current software environment.
If the requirement inference fails, it falls back to using :func:`get_default_pip_requirements`.
Both requirements and constraints are automatically parsed and written to ``requirements.txt`` and
``constraints.txt`` files, respectively, and stored as part of the model. Requirements are also
written to the ``pip`` section of the model's conda environment (``conda.yaml``) file.
""",
        "extra_pip_requirements": """
Either an iterable of pip requirement strings
(e.g. ``["pandas", "-r requirements.txt", "-c constraints.txt"]``) or the string path to
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
        "signature": """
an instance of the :py:class:`ModelSignature <mlflow.models.ModelSignature>`
class that describes the model's inputs and outputs. If not specified but an
``input_example`` is supplied, a signature will be automatically inferred
based on the supplied input example and model. To disable automatic signature
inference when providing an input example, set ``signature`` to ``False``.
To manually infer a model signature, call
:py:func:`infer_signature() <mlflow.models.infer_signature>` on datasets 
with valid model inputs, such as a training dataset with the target column
omitted, and valid model outputs, like model predictions made on the training
dataset, for example:

.. code-block:: python

    from mlflow.models import infer_signature

    train = df.drop_column("target_label")
    predictions = ...  # compute model predictions
    signature = infer_signature(train, predictions)
""",
        "input_example": """
one or several instances of valid model input. The input example is used
as a hint of what data to feed the model. It will be converted to a Pandas
DataFrame and then serialized to json using the Pandas split-oriented
format, or a numpy array where the example will be serialized to json
by converting it to a list. Bytes are base64-encoded. When the ``signature`` parameter is
``None``, the input example is used to infer a model signature.
""",
    }
)


def get_module_min_and_max_supported_ranges(module_name):
    """
    Extracts the minimum and maximum supported package versions from the provided module name.
    The version information is provided via the yaml-to-python-script generation script in
    dev/update_ml_package_versions.py which writes a python file to the importable namespace of
    mlflow.ml_package_versions

    :param module_name: The string name of the module as it is registered in ml_package_versions.py
    :return: tuple of minimum supported version, maximum supported version as strings.
    """
    versions = _ML_PACKAGE_VERSIONS[module_name]["models"]
    min_version = versions["minimum"]
    max_version = versions["maximum"]
    return min_version, max_version


def docstring_version_compatibility_warning(integration_name):
    """
    Generates a docstring that can be applied as a note stating a version compatibility range for
    a given flavor.

    :param integration_name: The name of the module as stored within ml-package-versions.yml
    :return: The wrapped function with the additional docstring header applied
    """

    def annotated_func(func):
        # NB: if using this decorator, ensure the package name to module name reference is
        # updated with the flavor's `save` and `load` functions being used within the dictionary
        # mlflow.utils.autologging_utils.versioning.FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY
        _, module_key = FLAVOR_TO_MODULE_NAME_AND_VERSION_INFO_KEY[integration_name]
        min_ver, max_ver = get_module_min_and_max_supported_ranges(module_key)
        required_pkg_versions = f"``{min_ver}`` -  ``{max_ver}``"

        notice = (
            f"The '{integration_name}' MLflow Models integration is known to be compatible with "
            f"the following package version ranges: {required_pkg_versions}. "
            f"MLflow Models integrations with {integration_name} may not succeed when used with "
            "package versions outside of this range."
        )

        @wraps(func)
        def version_func(*args, **kwargs):
            installed_version = Version(importlib_metadata.version(module_key))
            if installed_version < Version(min_ver) or installed_version > Version(max_ver):
                warnings.warn(notice, category=FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        version_func.__doc__ = (
            "    .. Note:: " + notice + "\n" * 2 + func.__doc__ if func.__doc__ else notice
        )

        return version_func

    return annotated_func
