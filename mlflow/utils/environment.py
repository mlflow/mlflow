import yaml

from mlflow.utils import PYTHON_VERSION
from mlflow.version import VERSION

_conda_header = """\
name: mlflow-env
channels:
  - defaults
"""

DEFAULT_PIP_DEPENDENCIES = ['mlflow' + '==' + VERSION]


def _mlflow_conda_env(path=None, additional_conda_deps=None, additional_pip_deps=None,
                      additional_conda_channels=None):
    """
    Creates a Conda environment with the specified package channels and dependencies.

    :param path: Local filesystem path where the conda env file is to be written. If unspecified,
                 the conda env will not be written to the filesystem; it will still be returned
                 in dictionary format.
    :param additional_conda_deps: List of additional conda dependencies passed as strings.
    :param additional_pip_deps: List of additional pip dependencies passed as strings.
    :param additional_channels: List of additional conda channels to search when resolving packages.
    :return: `None` if `path` is specified. Otherwise, the a dictionary representation of the
             Conda environment.
    """
    env = yaml.load(_conda_header)
    env["dependencies"] = ["python={}".format(PYTHON_VERSION)]
    if additional_conda_deps is not None:
        env["dependencies"] += additional_conda_deps
    if additional_pip_deps is not None:
        env["dependencies"].append({"pip": additional_pip_deps})
    if additional_conda_channels is not None:
        env["channels"] += additional_conda_channels

    if path is not None:
        with open(path, "w") as out:
            yaml.safe_dump(env, stream=out, default_flow_style=False)
        return None
    else:
        return env


def update_conda_env_deps(conda_env, pip_dependencies):
    if conda_env:
        dependencies = conda_env.get('dependencies')
        if dependencies:
            if isinstance(dependencies[-1], dict):
                env_pip_dependencies = dependencies[-1]['pip']
                dependencies[-1]['pip'] = pip_dependencies + env_pip_dependencies
            else:
                dependencies.append({"pip": pip_dependencies})
    return conda_env
