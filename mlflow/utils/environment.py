import yaml
import tempfile
import os
import logging


from mlflow.utils import PYTHON_VERSION
from mlflow.utils.requirements_utils import _parse_requirements, _infer_requirements
from packaging.requirements import Requirement, InvalidRequirement

_logger = logging.getLogger(__name__)

_conda_header = """\
name: mlflow-env
channels:
  - conda-forge
"""

_CONDA_ENV_FILE_NAME = "conda.yaml"
_REQUIREMENTS_FILE_NAME = "requirements.txt"
_CONSTRAINTS_FILE_NAME = "constraints.txt"


def _mlflow_conda_env(
    path=None,
    additional_conda_deps=None,
    additional_pip_deps=None,
    additional_conda_channels=None,
    install_mlflow=True,
):
    """
    Creates a Conda environment with the specified package channels and dependencies. If there are
    any pip dependencies, including from the install_mlflow parameter, then pip will be added to
    the conda dependencies. This is done to ensure that the pip inside the conda environment is
    used to install the pip dependencies.

    :param path: Local filesystem path where the conda env file is to be written. If unspecified,
                 the conda env will not be written to the filesystem; it will still be returned
                 in dictionary format.
    :param additional_conda_deps: List of additional conda dependencies passed as strings.
    :param additional_pip_deps: List of additional pip dependencies passed as strings.
    :param additional_conda_channels: List of additional conda channels to search when resolving
                                      packages.
    :return: ``None`` if ``path`` is specified. Otherwise, the a dictionary representation of the
             Conda environment.
    """
    pip_deps = (["mlflow"] if install_mlflow else []) + (
        additional_pip_deps if additional_pip_deps else []
    )
    conda_deps = (additional_conda_deps if additional_conda_deps else []) + (
        ["pip"] if pip_deps else []
    )

    env = yaml.safe_load(_conda_header)
    env["dependencies"] = ["python={}".format(PYTHON_VERSION)]
    if conda_deps is not None:
        env["dependencies"] += conda_deps
    env["dependencies"].append({"pip": pip_deps})
    if additional_conda_channels is not None:
        env["channels"] += additional_conda_channels

    if path is not None:
        with open(path, "w") as out:
            yaml.safe_dump(env, stream=out, default_flow_style=False)
        return None
    else:
        return env


def _mlflow_additional_pip_env(
    pip_deps, path=None,
):
    requirements = "\n".join(pip_deps)
    if path is not None:
        with open(path, "w") as out:
            out.write(requirements)
        return None
    else:
        return requirements


def _is_pip_deps(dep):
    """
    Returns True if `dep` is a dict representing pip dependencies
    """
    return isinstance(dep, dict) and "pip" in dep


def _get_pip_deps(conda_env):
    """
    :return: The pip dependencies from the conda env
    """
    if conda_env is not None:
        for dep in conda_env["dependencies"]:
            if _is_pip_deps(dep):
                return dep["pip"]
    return []


def _overwrite_pip_deps(conda_env, new_pip_deps):
    """
    Overwrites the pip dependencies section in the given conda env dictionary.

    {
        "name": "env",
        "channels": [...],
        "dependencies": [
            ...,
            "pip",
            {"pip": [...]},  <- Overwrite this
        ],
    }
    """
    deps = conda_env.get("dependencies", [])
    new_deps = []
    contains_pip_deps = False
    for dep in deps:
        if _is_pip_deps(dep):
            contains_pip_deps = True
            new_deps.append({"pip": new_pip_deps})
        else:
            new_deps.append(dep)

    if not contains_pip_deps:
        new_deps.append({"pip": new_pip_deps})

    return {**conda_env, "dependencies": new_deps}


def _log_pip_requirements(conda_env, path, requirements_file=_REQUIREMENTS_FILE_NAME):
    pip_deps = _get_pip_deps(conda_env)
    _mlflow_additional_pip_env(pip_deps, path=os.path.join(path, requirements_file))


def _parse_pip_requirements(pip_requirements):
    """
    Parses an iterable of pip requirement strings or a pip requirements file.

    :param pip_requirements: Either an iterable of pip requirement strings
        (e.g. ``["scikit-learn", "-r requirements.txt"]``) or the string path to a pip requirements
        file on the local filesystem (e.g. ``"requirements.txt"``). If ``None``, an empty list will
        be returned.
    :return: A tuple of parsed requirements and constraints.
    """
    if pip_requirements is None:
        return [], []

    def _is_string(x):
        return isinstance(x, str)

    def _is_iterable(x):
        try:
            iter(x)
            return True
        except Exception:
            return False

    if _is_string(pip_requirements):
        requirements = []
        constraints = []
        for req_or_con in _parse_requirements(pip_requirements, is_constraint=False):
            if req_or_con.is_constraint:
                constraints.append(req_or_con.req_str)
            else:
                requirements.append(req_or_con.req_str)

        return requirements, constraints
    elif _is_iterable(pip_requirements) and all(map(_is_string, pip_requirements)):
        try:
            # Create a temporary requirements file in the current working directory
            tmp_req_file = tempfile.NamedTemporaryFile(
                mode="w",
                prefix="mlflow.",
                suffix=".tmp.requirements.txt",
                dir=os.getcwd(),
                # Setting `delete` to True causes a permission-denied error on Windows
                # while trying to read the generated temporary file.
                delete=False,
            )
            tmp_req_file.write("\n".join(pip_requirements))
            tmp_req_file.close()
            return _parse_pip_requirements(tmp_req_file.name)
        finally:
            # Clean up the temporary requirements file
            os.remove(tmp_req_file.name)
    else:
        raise TypeError(
            "`pip_requirements` must be either a string path to a pip requirements file on the "
            "local filesystem or an iterable of pip requirement strings, but got `{}`".format(
                type(pip_requirements)
            )
        )


_INFER_PIP_REQUIREMENTS_FALLBACK_MESSAGE = (
    "Encountered an unexpected error while inferring pip requirements (model URI: %s, flavor: %s)"
)


def infer_pip_requirements(model_uri, flavor, fallback=None):
    """
    Infers the pip requirements of the specified model by creating a subprocess and loading
    the model in it to determine which packages are imported.

    :param model_uri: The URI of the model.
    :param flavor: The flavor name of the model.
    :param fallback: If provided, an unexpected error during the inference procedure is swallowed
                     and the value of ``fallback`` is returned. Otherwise, the error is raised.
    :return: A list of inferred pip requirements (e.g. ``["scikit-learn==0.24.2", ...]``).
    """
    try:
        return _infer_requirements(model_uri, flavor)
    except Exception:
        if fallback is not None:
            _logger.exception(_INFER_PIP_REQUIREMENTS_FALLBACK_MESSAGE, model_uri, flavor)
            return fallback
        raise


def _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements):
    """
    Validates that only one or none of `conda_env`, `pip_requirements`, and
    `extra_pip_requirements` is specified.
    """
    args = [
        conda_env,
        pip_requirements,
        extra_pip_requirements,
    ]
    specified = [arg for arg in args if arg is not None]
    if len(specified) > 1:
        raise ValueError(
            "Only one of `conda_env`, `pip_requirements`, and "
            "`extra_pip_requirements` can be specified"
        )


def _is_mlflow_requirement(requirement_string):
    """
    Returns True if `requirement_string` represents a requirement for mlflow (e.g. 'mlflow==1.2.3').
    """
    try:
        # `Requirement` throws an `InvalidRequirement` exception if `requirement_string` doesn't
        # conform to PEP 508 (https://www.python.org/dev/peps/pep-0508).
        return Requirement(requirement_string).name.lower() == "mlflow"
    except InvalidRequirement:
        # A local file path or URL falls into this branch.

        # TODO: Return True if `requirement_string` represents a project directory for MLflow
        # (e.g. '/path/to/mlflow') or git repository URL (e.g. 'https://github.com/mlflow/mlflow').
        return False


def _contains_mlflow_requirement(requirements):
    """
    Returns True if `requirements` contains a requirement for mlflow (e.g. 'mlflow==1.2.3').
    """
    return any(map(_is_mlflow_requirement, requirements))


def _process_pip_requirements(
    default_pip_requirements, pip_requirements=None, extra_pip_requirements=None
):
    """
    Processes `pip_requirements` and `extra_pip_requirements` passed to `mlflow.*.save_model` or
    `mlflow.*.log_model`, and returns a tuple of (conda_env, pip_requirements, pip_constraints).
    """
    constraints = []
    if pip_requirements is not None:
        pip_reqs, constraints = _parse_pip_requirements(pip_requirements)
    elif extra_pip_requirements is not None:
        extra_pip_requirements, constraints = _parse_pip_requirements(extra_pip_requirements)
        pip_reqs = default_pip_requirements + extra_pip_requirements
    else:
        pip_reqs = default_pip_requirements

    if not _contains_mlflow_requirement(pip_reqs):
        pip_reqs.insert(0, "mlflow")

    if constraints:
        pip_reqs.append(f"-c {_CONSTRAINTS_FILE_NAME}")

    # Set `install_mlflow` to False because `pip_reqs` already contains `mlflow`
    conda_env = _mlflow_conda_env(additional_pip_deps=pip_reqs, install_mlflow=False)
    return conda_env, pip_reqs, constraints


def _process_conda_env(conda_env):
    """
    Processes `conda_env` passed to `mlflow.*.save_model` or `mlflow.*.log_model`, and returns
    a tuple of (conda_env, pip_requirements, pip_constraints).
    """
    if isinstance(conda_env, str):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    elif not isinstance(conda_env, dict):
        raise TypeError(
            "Expected a string path to a conda env yaml file or a `dict` representing a conda env, "
            "but got `{}`".format(type(conda_env).__name__)
        )

    # User-specified `conda_env` may contain requirements/constraints file references
    pip_reqs = _get_pip_deps(conda_env)
    pip_reqs, constraints = _parse_pip_requirements(pip_reqs)

    if not _contains_mlflow_requirement(pip_reqs):
        pip_reqs.insert(0, "mlflow")

    if constraints:
        pip_reqs.append(f"-c {_CONSTRAINTS_FILE_NAME}")

    conda_env = _overwrite_pip_deps(conda_env, pip_reqs)
    return conda_env, pip_reqs, constraints
