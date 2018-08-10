import os
import shutil

from mlflow.utils import PYTHON_VERSION

_conda_header = """name: mlflow-env
channels:
  - anaconda
  - defaults
dependencies:"""


def _mlflow_conda_env(path, additional_conda_deps=None, additional_pip_deps=None):
    """
    Create conda environment file. Contains default dependency on current python version.
    :param path: local filesystem path where the conda env file is to be created.
    :param additional_conda_deps: List of additional conda dependencies passed as strings.
    :param additional_pip_deps: List of additional pip dependencies passed as strings.
    :return: path where the files has been created
    """
    conda_deps = ["python={}".format(PYTHON_VERSION)]
    if additional_conda_deps:
        conda_deps += additional_conda_deps
    pip_deps = additional_pip_deps
    with open(path, "w") as f:
        f.write(_conda_header)
        prefix = "\n  - "
        f.write(prefix + prefix.join(conda_deps))
        if pip_deps:
            f.write(prefix + "pip:")
            prefix = "\n    - "
            f.write(prefix + prefix.join(pip_deps))
        f.write("\n")
    return path


def add_conda_env(model_path, env_path):
    """
    model_path : The path to the root of the MLFlow model to which to add the conda environment.
    env_path : The path of the conda environment. If `env_path` is `None`, no
               environment will be added.

    :return: New path to the conda environment within the MLFlow model directory,
             or `None` if no conda environment was specified.
    """
    if env_path is None:
        return None

    env_basepath = os.path.basename(os.path.abspath(env_path))
    dest_path = os.path.join(model_path, env_basepath)
    shutil.copyfile(env_path, dest_path)
    return dest_path
