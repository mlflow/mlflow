import yaml
import tempfile
import os

from mlflow.utils import PYTHON_VERSION
from mlflow.utils.requirements_utils import _parse_requirements

_conda_header = """\
name: mlflow-env
channels:
  - conda-forge
"""


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


def _get_pip_deps(conda_env):
    """
    :return: The pip dependencies from the conda env
    """
    if conda_env is not None:
        for dep in conda_env["dependencies"]:
            if isinstance(dep, dict) and "pip" in dep:
                return dep["pip"]
    return []


def _log_pip_requirements(conda_env, path, requirements_file="requirements.txt"):
    pip_deps = _get_pip_deps(conda_env)
    _mlflow_additional_pip_env(pip_deps, path=os.path.join(path, requirements_file))


def _parse_pip_requirements(pip_requirements):
    """
    Parses an iterable of pip requirement strings or a pip requirements file.

    :param pip_requirements: Either an iterable of pip requirement strings
        (e.g. ``["scikit-learn", "-r requirements.txt"]``) or the string path to a pip requirements
        file on the local filesystem (e.g. ``"requirements.txt"``). If ``None``, an empty list will
        be returned.
    :return: A list of pip requirement strings.
    """
    if pip_requirements is None:
        return []

    def _is_string(x):
        return isinstance(x, str)

    def _is_iterable(x):
        try:
            iter(x)
            return True
        except Exception:
            return False

    if _is_string(pip_requirements):
        return list(_parse_requirements(pip_requirements))
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
