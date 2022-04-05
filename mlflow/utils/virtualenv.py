import os
import logging
import shutil
import uuid
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.utils.process import _exec_cmd, _join_commands, _IS_UNIX
from mlflow.utils.environment import (
    PythonEnv,
    _PYTHON_ENV_FILE_NAME,
    _CONDA_ENV_FILE_NAME,
    _get_mlflow_env_name,
    _get_pip_install_mlflow,
)
from mlflow.utils.conda import _get_conda_dependencies


_MLFLOW_ENV_ROOT_ENV_VAR = "MLFLOW_ENV_ROOT"


_logger = logging.getLogger(__name__)


def _get_mlflow_virtualenv_root():
    """
    Returns the root directory to store virtualenv environments created by MLflow.
    """
    return os.getenv(_MLFLOW_ENV_ROOT_ENV_VAR, str(Path.home().joinpath(".mlflow", "envs")))


def _is_pyenv_available():
    """
    Returns True if pyenv is available, otherwise False.
    """
    return shutil.which("pyenv") is not None


def _validate_pyenv_is_available():
    """
    Validates pyenv is available. If not, throws an `MlflowException` with a brief instruction on
    how to install pyenv.
    """
    url = (
        "https://github.com/pyenv/pyenv#installation"
        if _IS_UNIX
        else "https://github.com/pyenv-win/pyenv-win#installation"
    )
    if not _is_pyenv_available():
        raise MlflowException(
            f"Could not find the pyenv binary. See {url} for installation instructions."
        )


def _is_virtualenv_available():
    """
    Returns True if virtualenv is available, otherwise False.
    """
    return shutil.which("virtualenv") is not None


def _validate_virtualenv_is_available():
    """
    Validates virtualenv is available. If not, throws an `MlflowException` with a brief instruction
    on how to install virtualenv.
    """
    if not _is_virtualenv_available():
        raise MlflowException(
            "Could not find the virtualenv binary. Run `pip install virtualenv` to install "
            "virtualenv."
        )


def _install_python(version):
    """
    Installs a specified version of python with pyenv and returns a path to the installed python
    binary.

    :param version: Python version to install.
    :return: Path to the installed python binary.
    """
    _logger.info("Installing python %s", version)
    # pyenv-win doesn't support `--skip-existing` but its behavior is enabled by default
    # https://github.com/pyenv-win/pyenv-win/pull/314
    pyenv_install_options = ("--skip-existing",) if _IS_UNIX else ()
    _exec_cmd(["pyenv", "install", *pyenv_install_options, version], capture_output=False)

    if _IS_UNIX:
        pyenv_root = _exec_cmd(["pyenv", "root"], capture_output=True).stdout.strip()
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
    Constructs `PythonEnv` from the model artifacts stored in `local_model_path`. If
    `python_env.yaml` is available, use it, otherwise extract model dependencies from `conda.yaml`.
    If `conda.yaml` contains conda dependencies except `python`, `pip`, `setuptools`, and, `wheel`,
    an `MlflowException` is thrown because conda dependencies cannot be installed in a virtualenv
    environment.

    :param local_model_path: Local directory containing the model artifacts.
    :return: `PythonEnv` instance.
    """
    python_env_file = local_model_path / _PYTHON_ENV_FILE_NAME
    if python_env_file.exists():
        return PythonEnv.from_yaml(python_env_file)
    else:
        _logger.info(
            "This model is missing %s, which is because it was logged in an older version"
            "of MLflow (< 1.26.0) that does not support restoring a model environment with "
            "virtualenv. Attempting to extract model dependencies from %s instead.",
            _PYTHON_ENV_FILE_NAME,
            _CONDA_ENV_FILE_NAME,
        )
        conda_yaml_path = local_model_path / _CONDA_ENV_FILE_NAME
        conda_deps = _get_conda_dependencies(
            conda_yaml_path, exclude=("python", "pip", "setuptools", "wheel")
        )
        if conda_deps:
            raise MlflowException(
                f"Cannot restore this model's environment with virtualenv because it contains "
                f"conda dependencies: {conda_deps}."
            )
        return PythonEnv.from_conda_yaml(conda_yaml_path)


def _create_virtualenv(local_model_path, python_bin_path, env_dir, python_env):
    # Created a command to activate the environment
    paths = ("bin", "activate") if _IS_UNIX else ("Scripts", "activate.bat")
    activate_cmd = env_dir.joinpath(*paths)
    activate_cmd = f"source {activate_cmd}" if _IS_UNIX else activate_cmd

    if env_dir.exists():
        _logger.info("Environment %s already exists", env_dir)
        return activate_cmd

    _logger.info("Creating a new environment %s", env_dir)
    _exec_cmd(["virtualenv", "--python", python_bin_path, env_dir], capture_output=False)

    _logger.info("Installing dependencies")
    for deps in filter(None, [python_env.build_dependencies, python_env.dependencies]):
        # Use a unique name to avoid conflicting custom requirements files logged by a user
        tmp_req_file = local_model_path / f"requirements.{uuid.uuid4().hex}.txt"
        tmp_req_file.write_text("\n".join(deps))
        # In windows `pip install pip==x.y.z` causes the following error:
        # `[WinError 5] Access is denied: 'C:\path\to\pip.exe`
        # This can be avoided by using `python -m`.
        cmd = _join_commands(activate_cmd, f"python -m pip install -r {tmp_req_file}")
        _exec_cmd(
            cmd,
            capture_output=False,
            # Run `pip install` in the model directory to resolve references in the
            # requirements file correctly
            cwd=local_model_path,
        )
        tmp_req_file.unlink()

    return activate_cmd


def _get_or_create_virtualenv(local_model_path, env_id=None):
    """
    Restores an MLflow model's environment with pyenv and virtualenv and returns a command
    to activate it.

    :param local_model_path: Local directory containing the model artifacts.
    :param env_id: Optional string that is added to the contents of the yaml file before
                   calculating the hash. It can be used to distinguish environments that have the
                   same conda dependencies but are supposed to be different based on the context.
                   For example, when serving the model we may install additional dependencies to the
                   environment after the environment has been activated.
    :return: Command to activate the created virtualenv environment
             (e.g. "source /path/to/bin/activate").
    """
    _validate_pyenv_is_available()
    _validate_virtualenv_is_available()

    # Read environment information
    local_model_path = Path(local_model_path)
    python_env = _get_python_env(local_model_path)

    # Create an environment
    python_bin_path = _install_python(python_env.python)
    env_root = Path(_get_mlflow_virtualenv_root())
    env_root.mkdir(parents=True, exist_ok=True)
    env_name = _get_mlflow_env_name(str(python_env) + (env_id or ""))
    env_dir = env_root / env_name
    try:
        return _create_virtualenv(local_model_path, python_bin_path, env_dir, python_env)
    except:
        _logger.warning("Encountered unexpected error while creating %s", env_dir)
        if env_dir.exists():
            _logger.warning("Attempting to remove %s", env_dir)
            shutil.rmtree(env_dir, ignore_errors=True)
            msg = "Failed to remove %s" if env_dir.exists() else "Successfully removed %s"
            _logger.warning(msg, env_dir)

        raise


def _execute_in_virtualenv(activate_cmd, command, install_mlflow, command_env=None):
    """
    Runs a command in a specified virtualenv environment.

    :param activate_cmd: Command to activate the virtualenv environment.
    :param command: Command to run in the virtualenv environment.
    :param install_mlflow: Flag to determine whether to install mlflow in the virtualenv
                           environment.
    :param command_env: Environment variables passed to a process running the command.
    """
    pre_command = [activate_cmd]
    if install_mlflow:
        pre_command.append(_get_pip_install_mlflow())
    cmd = _join_commands(*pre_command, command)
    _logger.info("Running %s", cmd)
    return _exec_cmd(cmd, capture_output=False, env=command_env)
