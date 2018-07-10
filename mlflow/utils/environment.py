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
