import textwrap
import warnings

from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.autologging_utils.versioning import (
    get_min_max_version_and_pip_release,
)


def _create_placeholder(key: str):
    return "{{ " + key + " }}"


def _replace_keys_with_placeholders(d: dict) -> dict:
    return {_create_placeholder(k): v for k, v in d.items()}


def _get_indentation_of_key(line: str, placeholder: str) -> str:
    index = line.find(placeholder)
    return (index * " ") if index != -1 else ""


def _indent(text: str, indent: str) -> str:
    """Indent everything but first line in text."""
    lines = text.splitlines()
    if len(lines) <= 1:
        return text

    else:
        first_line = lines[0]
        subsequent_lines = "\n".join(list(lines[1:]))
        indented_subsequent_lines = textwrap.indent(subsequent_lines, indent)
        return first_line + "\n" + indented_subsequent_lines


def _replace_all(text: str, replacements: dict[str, str]) -> str:
    """
    Replace all instances of replacements.keys() with their corresponding
    values in text. The replacements will be inserted on the same line
    with wrapping to the same level of indentation, for example:

    ```
    Args:
        param_1: {{ key }}
    ```

    will become...

    ```
    Args:
        param_1: replaced_value_at same indentation as prior
                 and if there are more lines they will also
                 have the same indentation.
    ```
    """
    for key, value in replacements.items():
        if key in text:
            indent = _get_indentation_of_key(text, key)
            indented_value = _indent(value, indent)
            text = text.replace(key, indented_value)
    return text


class ParamDocs(dict):
    """
    Represents a set of parameter documents in the docstring.
    """

    def __repr__(self):
        return f"ParamDocs({super().__repr__()})"

    def format(self, **kwargs):
        """
        Formats values to be substituted in via the format_docstring() method.

        Args:
            kwargs: A `dict` in the form of `{"< placeholder name >": "< value >"}`.

        Returns:
            A new `ParamDocs` instance with the formatted param docs.

        .. code-block:: text
            :caption: Example

            >>> pd = ParamDocs(p1="{{ doc1 }}", p2="{{ doc2 }}")
            >>> pd.format(doc1="foo", doc2="bar")
            ParamDocs({'p1': 'foo', 'p2': 'bar'})
        """
        replacements = _replace_keys_with_placeholders(kwargs)
        return ParamDocs({k: _replace_all(v, replacements) for k, v in self.items()})

    def format_docstring(self, docstring: str) -> str:
        """
        Formats placeholders in `docstring`.

        Args:
            docstring: A docstring with placeholders to be replaced.
                If provided with None, will return None.

        .. code-block:: text
            :caption: Example

            >>> pd = ParamDocs(p1="doc1", p2="doc2
            doc2 second line")
            >>> docstring = '''
            ... Args:
            ...     p1: {{ p1 }}
            ...     p2: {{ p2 }}
            ... '''.strip()
            >>> print(pd.format_docstring(docstring))
        """
        if docstring is None:
            return None

        replacements = _replace_keys_with_placeholders(self)
        lines = docstring.splitlines()
        for i, line in enumerate(lines):
            lines[i] = _replace_all(line, replacements)

        return "\n".join(lines)


def format_docstring(param_docs):
    """
    Returns a decorator that replaces param doc placeholders (e.g. '{{ param_name }}') in the
    docstring of the decorated function.

    Args:
        param_docs: A `ParamDocs` instance or `dict`.

    Returns:
        A decorator to apply the formatting.

    .. code-block:: text
        :caption: Example

        >>> param_docs = {"p1": "doc1", "p2": "doc2
        doc2 second line"}
        >>> @format_docstring(param_docs)
        ... def func(p1, p2):
        ...     '''
        ...     Args:
        ...         p1: {{ p1 }}
        ...         p2: {{ p2 }}
        ...     '''
        >>> import textwrap
        >>> print(textwrap.dedent(func.__doc__).strip())

        Args:
            p1: doc1
            p2: doc2
                doc2 second line
    """
    param_docs = ParamDocs(param_docs)

    def decorator(func):
        func.__doc__ = param_docs.format_docstring(func.__doc__)
        return func

    return decorator


# `{{ ... }}` represents a placeholder.
LOG_MODEL_PARAM_DOCS = ParamDocs(
    {
        "conda_env": (
            """Either a dictionary representation of a Conda environment or the path to a conda
environment yaml file. If provided, this describes the environment this model should be run in.
At a minimum, it should specify the dependencies contained in :func:`get_default_conda_env()`.
If ``None``, a conda environment with pip requirements inferred by
:func:`mlflow.models.infer_pip_requirements` is added
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
    }"""
        ),
        "pip_requirements": (
            """Either an iterable of pip requirement strings
(e.g. ``["{{ package_name }}", "-r requirements.txt", "-c constraints.txt"]``) or the string path to
a pip requirements file on the local filesystem (e.g. ``"requirements.txt"``). If provided, this
describes the environment this model should be run in. If ``None``, a default list of requirements
is inferred by :func:`mlflow.models.infer_pip_requirements` from the current software environment.
If the requirement inference fails, it falls back to using :func:`get_default_pip_requirements`.
Both requirements and constraints are automatically parsed and written to ``requirements.txt`` and
``constraints.txt`` files, respectively, and stored as part of the model. Requirements are also
written to the ``pip`` section of the model's conda environment (``conda.yaml``) file."""
        ),
        "extra_pip_requirements": (
            """Either an iterable of pip
requirement strings
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

`This example <https://github.com/mlflow/mlflow/blob/master/examples/pip_requirements/pip_requirements.py>`_ demonstrates how to specify pip requirements using
``pip_requirements`` and ``extra_pip_requirements``."""  # noqa: E501
        ),
        "signature": (
            """an instance of the :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
"""
        ),
        "metadata": (
            "Custom metadata dictionary passed to the model and stored in the MLmodel file."
        ),
        "input_example": (
            """one or several instances of valid model input. The input example is used
as a hint of what data to feed the model. It will be converted to a Pandas
DataFrame and then serialized to json using the Pandas split-oriented
format, or a numpy array where the example will be serialized to json
by converting it to a list. Bytes are base64-encoded. When the ``signature`` parameter is
``None``, the input example is used to infer a model signature.
"""
        ),
        "example_no_conversion": (
            """This parameter is deprecated and will be removed in a future release.
It's no longer used and can be safely removed. Input examples are not converted anymore.
"""
        ),
        "prompt_template": (
            """A string that, if provided, will be used to format the user's input prior
to inference. The string should contain a single placeholder, ``{prompt}``, which will be
replaced with the user's input. For example: ``"Answer the following question. Q: {prompt} A:"``.

Currently, only the following pipeline types are supported:

- `feature-extraction <https://huggingface.co/transformers/main_classes/pipelines.html#transformers.FeatureExtractionPipeline>`_
- `fill-mask <https://huggingface.co/transformers/main_classes/pipelines.html#transformers.FillMaskPipeline>`_
- `summarization <https://huggingface.co/transformers/main_classes/pipelines.html#transformers.SummarizationPipeline>`_
- `text2text-generation <https://huggingface.co/transformers/main_classes/pipelines.html#transformers.Text2TextGenerationPipeline>`_
- `text-generation <https://huggingface.co/transformers/main_classes/pipelines.html#transformers.TextGenerationPipeline>`_
"""
        ),
        "code_paths": (
            """A list of local filesystem paths to Python file dependencies (or directories
containing file dependencies). These files are *prepended* to the system path when the model
is loaded. Files declared as dependencies for a given model should have relative
imports declared from a common root path if multiple files are defined with import dependencies
between them to avoid import errors when loading the model.

For a detailed explanation of ``code_paths`` functionality, recommended usage patterns and
limitations, see the
`code_paths usage guide <https://mlflow.org/docs/latest/model/dependencies.html?highlight=code_paths#saving-extra-code-with-an-mlflow-model>`_.
"""
        ),
        # Only pyfunc flavor supports `infer_code_paths`.
        "code_paths_pyfunc": (
            """A list of local filesystem paths to Python file dependencies (or directories
containing file dependencies). These files are *prepended* to the system path when the model
is loaded. Files declared as dependencies for a given model should have relative
imports declared from a common root path if multiple files are defined with import dependencies
between them to avoid import errors when loading the model.

You can leave ``code_paths`` argument unset but set ``infer_code_paths`` to ``True`` to let MLflow
infer the model code paths. See ``infer_code_paths`` argument doc for details.

For a detailed explanation of ``code_paths`` functionality, recommended usage patterns and
limitations, see the
`code_paths usage guide <https://mlflow.org/docs/latest/model/dependencies.html?highlight=code_paths#saving-extra-code-with-an-mlflow-model>`_.
"""
        ),
        "infer_code_paths": (
            """If set to ``True``, MLflow automatically infers model code paths. The inferred
            code path files only include necessary python module files. Only python code files
            under current working directory are automatically inferable. Default value is
            ``False``.

.. warning::
    Please ensure that the custom python module code does not contain sensitive data such as
    credential token strings, otherwise they might be included in the automatic inferred code
    path files and be logged to MLflow artifact repository.

    If your custom python module depends on non-python files (e.g. a JSON file) with a relative
    path to the module code file path, the non-python files can't be automatically inferred as the
    code path file. To address this issue, you should put all used non-python files outside
    your custom code directory.

    If a python code file is loaded as the python ``__main__`` module, then this code file can't be
    inferred as the code path file. If your model depends on classes / functions defined in
    ``__main__`` module, you should use `cloudpickle` to dump your model instance in order to pickle
    classes / functions in ``__main__``.

.. Note:: Experimental: This parameter may change or be removed in a future release without warning.
"""
        ),
        "save_pretrained": (
            """If set to ``False``, MLflow will not save the Transformer model weight files,
instead only saving the reference to the HuggingFace Hub model repository and its commit hash.
This is useful when you load the pretrained model from HuggingFace Hub and want to log or save
it to MLflow without modifying the model weights. In such case, specifying this flag to
``False`` will save the storage space and reduce time to save the model. Please refer to the
:ref:`Storage-Efficient Model Logging <transformers-save-pretrained-guide>` for more detailed usage.


.. warning::

    If the model is saved with ``save_pretrained`` set to ``False``, the model cannot be
    registered to the MLflow Model Registry. In order to convert the model to the one that
    can be registered, you can use :py:func:`mlflow.transformers.persist_pretrained_model()`
    to download the model weights from the HuggingFace Hub and save it in the existing model
    artifacts. Please refer to :ref:`Transformers flavor documentation <persist-pretrained-guide>`
    for more detailed usage.

    .. code-block:: python

        import mlflow.transformers

        model_uri = "YOUR_MODEL_URI_LOGGED_WITH_SAVE_PRETRAINED_FALSE"
        model = mlflow.transformers.persist_pretrained_model(model_uri)
        mlflow.register_model(model_uri, "model_name")

.. important::

    When you save the `PEFT <https://huggingface.co/docs/peft/en/index>`_ model, MLflow will
    override the `save_pretrained` flag to `False` and only store the PEFT adapter weights. The
    base model weights are not saved but the reference to the HuggingFace repository and
    its commit hash are logged instead.
"""
        ),
    }
)


def get_module_min_and_max_supported_ranges(flavor_name):
    """
    Extracts the minimum and maximum supported package versions from the provided module name.
    The version information is provided via the yaml-to-python-script generation script in
    dev/update_ml_package_versions.py which writes a python file to the importable namespace of
    mlflow.ml_package_versions

    Args:
        flavor_name: The flavor name registered in ml_package_versions.py

    Returns:
        tuple of module name, minimum supported version, maximum supported version as strings.
    """
    if flavor_name == "pyspark.ml":
        # pyspark.ml is a special case of spark flavor
        flavor_name = "spark"

    module_name = _ML_PACKAGE_VERSIONS[flavor_name]["package_info"].get("module_name", flavor_name)
    versions = _ML_PACKAGE_VERSIONS[flavor_name]["models"]
    min_version = versions["minimum"]
    max_version = versions["maximum"]
    return module_name, min_version, max_version


def _do_version_compatibility_warning(msg: str):
    """
    Isolate the warn call to show the warning only once.
    """
    warnings.warn(msg, category=UserWarning, stacklevel=2)


def docstring_version_compatibility_warning(integration_name):
    """
    Generates a docstring that can be applied as a note stating a version compatibility range for
    a given flavor and optionally raises a warning if the installed version is outside of the
    supported range.

    Args:
        integration_name: The name of the module as stored within ml-package-versions.yml

    Returns:
        The wrapped function with the additional docstring header applied
    """

    def annotated_func(func):
        # NB: if using this decorator, ensure the package name to module name reference is
        # updated with the flavor's `save` and `load` functions being used within
        # ml-package-version.yml file.
        min_ver, max_ver, pip_release = get_min_max_version_and_pip_release(
            integration_name, "models"
        )
        notice = (
            f"The '{integration_name}' MLflow Models integration is known to be compatible with "
            f"``{min_ver}`` <= ``{pip_release}`` <= ``{max_ver}``. "
            f"MLflow Models integrations with {integration_name} may not succeed when used with "
            "package versions outside of this range."
        )

        func.__doc__ = (
            "    .. Note:: " + notice + "\n" * 2 + func.__doc__ if func.__doc__ else notice
        )

        return func

    return annotated_func
