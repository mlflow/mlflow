import yaml

from mlflow.utils import PYTHON_VERSION

_conda_header = """\
name: mlflow-env
channels:
  - defaults
"""


def _mlflow_conda_env(path=None, additional_conda_deps=None, additional_pip_deps=None,
        additional_conda_channels=None):
    """
    Create conda environment file. Contains default dependency on current python version.
    :param path: Local filesystem path where the conda env file is to be created. If unspecified,
                 the conda env will be returned as a string.
    :param additional_conda_deps: List of additional conda dependencies passed as strings.
    :param additional_pip_deps: List of additional pip dependencies passed as strings.
    :param additional_channels: List of additional conda channels to search when resolving packages.
    :return: Either:
                * The path where the conda environment has been created, if ``path`` is specified.
                * The conda environment definition as a string, if ``path`` is not specified.
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
        with open(path, "w") as f:
            yaml.safe_dump(env, stream=f, default_flow_style=False)
        return path
    else:
        return yaml.safe_dump(env, stream=None, default_flow_style=False)


def _get_base_env():
    base_env = yaml.load(_conda_header)
    base_env["dependencies"] = ["python={}".format(PYTHON_VERSION)]
    return base_env
