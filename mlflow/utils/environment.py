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
    Creates a Conda environment with the specified package channels and dependencies.
    Optionally, saves the Conda environment to an output path, if specified.

    :param path: Local filesystem path where the conda env file is to be written. If unspecified,
                 the conda env will not be written to the filesystem; it will still be returned
                 in string format.
    :param additional_conda_deps: List of additional conda dependencies passed as strings.
    :param additional_pip_deps: List of additional pip dependencies passed as strings.
    :param additional_channels: List of additional conda channels to search when resolving packages.
    :return: The text of the conda environment that was created.
    """
    env = yaml.load(_conda_header)
    env["dependencies"] = ["python={}".format(PYTHON_VERSION)]
    if additional_conda_deps is not None:
        env["dependencies"] += additional_conda_deps
    if additional_pip_deps is not None:
        env["dependencies"].append({"pip": additional_pip_deps})
    if additional_conda_channels is not None:
        env["channels"] += additional_conda_channels

    env_text = yaml.safe_dump(env, stream=None, default_flow_style=False)

    if path is not None:
        with open(path, "w") as f:
            f.write(env_text)

    return env_text


def _get_base_env():
    base_env = yaml.load(_conda_header)
    base_env["dependencies"] = ["python={}".format(PYTHON_VERSION)]
    return base_env
