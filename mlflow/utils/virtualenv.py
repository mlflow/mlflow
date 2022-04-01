import os
import logging
import subprocess
import hashlib
import shutil
import uuid
from pathlib import Path

import yaml

from mlflow.exceptions import MlflowException
from mlflow.utils import process
from mlflow.utils.environment import (
    PythonEnv,
    _PYTHON_ENV_FILE_NAME,
    _CONDA_ENV_FILE_NAME,
)
from mlflow.utils.requirements_utils import _get_package_name
from mlflow.version import VERSION


_MLFLOW_ENV_ROOT_ENV_VAR = "MLFLOW_ENV_ROOT"
_IS_UNIX = os.name != "nt"


_logger = logging.getLogger(__name__)


def _get_pip_install_mlflow():
    mlflow_home = os.getenv("MLFLOW_HOME")
    if mlflow_home:  # dev version
        return "pip install -e {} 1>&2".format(mlflow_home)
    else:
        return "pip install mlflow=={} 1>&2".format(VERSION)


def _hash(string):
    return hashlib.sha1(string.encode("utf-8")).hexdigest()


def _get_mlflow_env_root():
    """
    Returns the root directory to store virtualenv environments created by MLflow.
    """
    return os.getenv(_MLFLOW_ENV_ROOT_ENV_VAR, str(Path.home().joinpath(".mlflow", "envs")))


def _get_conda_dependencies(conda_yaml_path, exclude=("python", "pip", "setuptools", "wheel")):
    """
    Extracts conda dependencies from a conda yaml file. Packages in `exclude` will be excluded
    from the result.

    :param conda_yaml_path: Conda yaml file path.
    :param exclude: Packages to be excluded from the result.
    """
    with open(conda_yaml_path) as f:
        conda_yaml = yaml.safe_load(f)
    return [
        d
        for d in conda_yaml.get("dependencies", [])
        if isinstance(d, str) and _get_package_name(d) not in exclude
    ]


def _bash_cmd(*args):
    entry_point = ["bash", "-c"] if _IS_UNIX else ["cmd", "/c"]
    sep = " && " if _IS_UNIX else " & "
    return [*entry_point, sep.join(args)]


def _is_pyenv_available():
    return subprocess.run(["pyenv", "--version"], capture_output=True, check=False).returncode == 0


def _validate_pyenv_is_available():
    if not _is_pyenv_available():
        raise MlflowException(
            "pyenv must be available to use this feature. Follow "
            "https://github.com/pyenv/pyenv#installation to install."
        )


def _is_virtualenv_available():
    return (
        subprocess.run(["virtualenv", "--version"], capture_output=True, check=False).returncode
        == 0
    )


def _validate_virtualenv_is_available():
    if not _is_virtualenv_available():
        raise MlflowException(
            "virtualenv must be available to use this feature. Run `pip install virtualenv` "
            "to install."
        )


def _install_python(version):
    """
    Installs a specified version of python and returns a path to the installed python binary

    :param version: Python version to install.
    :return: Path to the installed python binary.
    """
    _logger.info("Installing python %s", version)
    # pyenv-win doesn't support `--skip-existing` but its behavior is enabled by default
    # https://github.com/pyenv-win/pyenv-win/pull/314
    pyenv_install_options = ("--skip-existing",) if _IS_UNIX else ()
    process.exec_cmd(["pyenv", "install", *pyenv_install_options, version], capture_output=False)

    if _IS_UNIX:
        pyenv_root = process.exec_cmd(["pyenv", "root"], capture_output=True)[1].strip()
        path_to_bin = ("bin", "python")
    else:
        # pyenv-win doesn't provide the `pyenv root` command
        pyenv_root = os.getenv("PYENV_ROOT")
        if pyenv_root is None:
            raise MlflowException("Environment variable 'PYENV_ROOT' must be set")
        path_to_bin = ("python.exe",)
    return Path(pyenv_root).joinpath("versions", version, *path_to_bin)


def _get_python_env(local_model_path):
    """
    Constructs `PythonEnv` from `local_model_path`.
    """
    python_env_file = local_model_path / _PYTHON_ENV_FILE_NAME
    if python_env_file.exists():
        return PythonEnv.from_yaml(python_env_file)
    else:
        _logger.info(
            "This model is missing %s. Attempting to extract required dependency information from "
            "%s",
            _PYTHON_ENV_FILE_NAME,
            _CONDA_ENV_FILE_NAME,
        )
        conda_yaml_path = local_model_path / _CONDA_ENV_FILE_NAME
        conda_deps = _get_conda_dependencies(conda_yaml_path)
        if conda_deps:
            raise MlflowException(
                f"This model's environment cannot be restored because it contains "
                f"conda dependencies ({conda_deps})."
            )
        return PythonEnv.from_conda_yaml(conda_yaml_path)


def _get_or_create_virtualenv(local_model_path, env_id=None):
    """
    Restores the model's environment using pyenv + virtualenv.

    :param local_model_path: Local directory containing the model artifacts.
    :param env_id: Optional string that is added to the contents of the yaml file before
                   calculating the hash. It can be used to distinguish environments that have the
                   same conda dependencies but are supposed to be different based on the context.
                   For example, when serving the model we may install additional dependencies to the
                   environment after the environment has been activated.
    """
    # Read environment information
    local_model_path = Path(local_model_path)
    python_env = _get_python_env(local_model_path)
    python_bin_path = _install_python(python_env.python)

    # Create an environment
    env_suffix = _hash(str(python_env.to_dict()) + (env_id or ""))
    env_name = "mlflow-" + env_suffix
    env_root = Path(_get_mlflow_env_root())
    env_root.mkdir(parents=True, exist_ok=True)
    env_dir = env_root / env_name
    env_exists = env_dir.exists()
    if not env_exists:
        _logger.info("Creating a new environment %s", env_dir)
        subprocess.run(["virtualenv", "--python", python_bin_path, str(env_dir)], check=True)
    else:
        _logger.info("Environment %s already exists", env_dir)

    # Construct a command to activate the environment
    paths = ("bin", "activate") if _IS_UNIX else ("Scripts", "activate.bat")
    activator = env_dir.joinpath(*paths)
    activate_cmd = f"source {activator}" if _IS_UNIX else activator

    # Install dependencies
    if not env_exists:
        try:
            _logger.info("Installing dependencies")
            for deps in filter(None, [python_env.build_dependencies, python_env.dependencies]):
                tmp_req_file = local_model_path / f"requirements.{uuid.uuid4().hex}.txt"
                tmp_req_file.write_text("\n".join(deps))
                # In windows `pip install pip==x.y.z` causes the following error:
                # `[WinError 5] Access is denied: 'C:\path\to\pip.exe`
                # We can avoid this error by using `python -m`.
                cmd = _bash_cmd(activate_cmd, f"python -m pip install -r {tmp_req_file}")
                process.exec_cmd(
                    cmd,
                    # Run `pip install` in the model directory to refer to the requirements
                    # file correctly
                    capture_output=False,
                    cwd=local_model_path,
                )
                tmp_req_file.unlink()
        except:
            _logger.warning(
                "Encountered an unexpected error while installing dependencies. Removing %s",
                env_dir,
            )
            shutil.rmtree(env_dir, ignore_errors=True)
            raise

    return activate_cmd


def _execute_in_virtualenv(activate_cmd, command, install_mlflow, command_env=None):
    """
    Runs a specified command in a virtualenv environment.

    :param activate_cmd: Command to activate the virtualenv environment.
    :param command: Command to run in the virtualenv environment.
    :param install_mlflow: Flag to determine whether to install mlflow in the virtualenv
                           environment.
    :param command_env: Environment variables passed to a process running the command.
    """
    # Run the command
    commands = [activate_cmd]
    if install_mlflow:
        commands.append(_get_pip_install_mlflow())
    return process.exec_cmd(_bash_cmd(*commands, command), capture_output=False, env=command_env)
