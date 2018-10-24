import yaml

from mlflow.utils import PYTHON_VERSION

_conda_header = """\
name: mlflow-env
channels:
  - defaults
"""


def _mlflow_conda_env(path, additional_conda_deps=None, additional_pip_deps=None,
                      additional_conda_channels=None, python_version=None):
    """
    Create conda environment file. Contains default dependency on current python version.
    :param path: local filesystem path where the conda env file is to be created.
    :param additional_conda_deps: List of additional conda dependencies passed as strings.
    :param additional_pip_deps: List of additional pip dependencies passed as strings.
    :param additional_channels: List of additional conda channels to search when resolving packages.
    :param python_version: The version of Python to use in the conda environment. If unspecified,
                           the environment will use the version of Python in which this command
                           is executed.
    :return: path where the conda environment file has been created.
    """
    env = yaml.load(_conda_header)
    env["dependencies"] = [
        "python={}".format(python_version if python_version is not None else PYTHON_VERSION)
    ]
    if additional_conda_deps is not None:
        env["dependencies"] += additional_conda_deps
    if additional_pip_deps is not None:
        env["dependencies"].append({"pip": additional_pip_deps})
    if additional_conda_channels is not None:
        env["channels"] += additional_conda_channels

    with open(path, "w") as f:
        yaml.safe_dump(env, f, default_flow_style=False)
    return path
