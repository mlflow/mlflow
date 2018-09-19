import os

from mlflow.projects import _get_conda_bin_executable, _get_conda_env_name, _get_environments
from mlflow.utils import PYTHON_VERSION
from mlflow.utils import process
from mlflow.utils.logging_utils import eprint

_conda_header = """name: mlflow-env
channels:
  - anaconda
  - defaults
dependencies:"""


def _get_mlflow_environments(conda_path=_get_conda_bin_executable("conda")):
    return [x for x in _get_environments(conda_path) if len(x) == 47 and x.startswith("mlflow-")]


def _clear_conda_env_cache(remove_envs=None, force=False):
    conda_path = _get_conda_bin_executable("conda")
    if remove_envs is None:
        remove_envs = []
    for x in remove_envs:
        if not os.path.exists(x):
            raise Exception("File  not found: '{}'".format(os.path.abspath(x)))
    remove_env_names = _get_mlflow_environments(conda_path)
    if remove_envs:
        remove_env_filter = [_get_conda_env_name(x) for x in remove_envs]
        remove_env_names = list(set(remove_env_names).intersection(set(remove_env_filter)))

    if not remove_env_names:
        eprint("No environments found, nothing to do.")
        return
    eprint("Following {} environments will be removed:".format(len(remove_env_names)))
    eprint("\n".join(remove_env_names))

    def proceed():
        if force:
            return True
        import sys
        print("Do you want to proceeed? y/n")
        for line in sys.stdin:
            line = line.strip().upper()
            if line == "Y":
                return True
            if line == "N":
                return False
            print("Do you want to proceeed? y/n")

    if not proceed():
        return
    for x in remove_env_names:
        rtc, stdout, stderr = process.exec_cmd([conda_path,
                                                "env", "remove", "-y", "--name", x])
        if rtc:
            msg = "Removing conda env failed, stdout='\n{stdout}\nstderr='\n{stderr}\n"
            msg.format(stdout=stdout, stderr=stderr)
            raise Exception(msg)
        else:
            eprint("removed cached env '{}'".format(x))


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
