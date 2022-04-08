import os
import logging
import shutil
import uuid
import re
import tempfile
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.utils.process import _exec_cmd, _join_commands, _IS_UNIX
from mlflow.utils.environment import (
    _PythonEnv,
    _PYTHON_ENV_FILE_NAME,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _get_mlflow_env_name,
    _get_pip_install_mlflow,
)
from mlflow.utils.requirements_utils import _get_package_name
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


_SEMANTIC_VERSION_REGEX = re.compile(r"^([0-9]+)\.([0-9]+)\.([0-9]+)$")


def _find_latest_installable_python_version(version_prefix):
    """
    Find the latest installable python version that matches the given version prefix
    from the output of `pyenv install --list`. For example, `version_prefix("3.8")` returns '3.8.x'
    where 'x' represents the latest micro version in 3.8.
    """
    lines = _exec_cmd(["pyenv", "install", "--list"], capture_output=True).stdout.splitlines()
    semantic_versions = filter(_SEMANTIC_VERSION_REGEX.match, map(str.strip, lines))
    matched = [v for v in semantic_versions if v.startswith(version_prefix)]
    if not matched:
        raise MlflowException((f"Could not find python version that matches {version_prefix}"))
    return sorted(matched)[-1]


def _install_python(version):
    """
    Installs a specified version of python with pyenv and returns a path to the installed python
    binary.

    :param version: Python version to install.
    :return: Path to the installed python binary.
    """
    version = (
        version
        if _SEMANTIC_VERSION_REGEX.match(version)
        else _find_latest_installable_python_version(version)
    )
    _logger.info("Installing python %s", version)
    # pyenv-win doesn't support `--skip-existing` but its behavior is enabled by default
    # https://github.com/pyenv-win/pyenv-win/pull/314
    pyenv_install_options = ("--skip-existing",) if _IS_UNIX else ()
    _exec_cmd(
        ["pyenv", "install", *pyenv_install_options, version],
        capture_output=False,
        # Windows fails to find pyenv and throws `FileNotFoundError` without `shell=True`
        shell=not _IS_UNIX,
    )

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
    Constructs `_PythonEnv` from the model artifacts stored in `local_model_path`. If
    `python_env.yaml` is available, use it, otherwise extract model dependencies from `conda.yaml`.
    If `conda.yaml` contains conda dependencies except `python`, `pip`, `setuptools`, and, `wheel`,
    an `MlflowException` is thrown because conda dependencies cannot be installed in a virtualenv
    environment.

    :param local_model_path: Local directory containing the model artifacts.
    :return: `_PythonEnv` instance.
    """
    python_env_file = local_model_path / _PYTHON_ENV_FILE_NAME
    requirements_file = local_model_path / _REQUIREMENTS_FILE_NAME
    conda_env_file = local_model_path / _CONDA_ENV_FILE_NAME
    if python_env_file.exists():
        return _PythonEnv.from_yaml(python_env_file)
    else:
        _logger.info(
            "This model is missing %s, which is because it was logged in an older version"
            "of MLflow (< 1.26.0) that does not support restoring a model environment with "
            "virtualenv. Attempting to extract model dependencies from %s and %s instead.",
            _PYTHON_ENV_FILE_NAME,
            _REQUIREMENTS_FILE_NAME,
            _CONDA_ENV_FILE_NAME,
        )
        conda_deps = _get_conda_dependencies(conda_env_file)
        build_packages = ("python", *_PythonEnv.BUILD_PACKAGES)
        conda_deps = [d for d in conda_deps if _get_package_name(d) not in build_packages]
        if conda_deps:
            _logger.warning(
                f"Found this model contains conda dependencies: {conda_deps}. The resulting "
                "virtualenv environment will not install them and may not be able to load the "
                "model."
            )
        if requirements_file.exists():
            deps = _PythonEnv.get_dependencies_from_conda_yaml(conda_env_file)
            return _PythonEnv(
                python=deps["python"],
                build_dependencies=deps["build_dependencies"],
                dependencies=[f"-r {_REQUIREMENTS_FILE_NAME}"],
            )
        else:
            return _PythonEnv.from_conda_yaml(conda_env_file)


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
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temporary requirements file
            tmpdir = Path(tmpdir)
            tmp_req_file = tmpdir.joinpath(f"requirements.{uuid.uuid4().hex}.txt")
            tmp_req_file.write_text("\n".join(deps))
            # Copy `requirements.txt` and `constraints.txt` in the temporary directory.
            # --- Why do we need to copy them? ---
            # Let's say `tmp_req_file` is stored in /tmp/123 and it contains '-r requirements.txt'.
            # pip resolves this reference to '-r /tmp/123/requirements.txt'. If we didn't copy
            # requirements.txt in /tmp/123, pip would complain /tmp/123/requirements.txt is not
            # found. The same thing applies to constraints.txt.
            req_file = local_model_path.joinpath(_REQUIREMENTS_FILE_NAME)
            if req_file.exists():
                shutil.copy(req_file, tmpdir)
            con_file = local_model_path.joinpath(_CONSTRAINTS_FILE_NAME)
            if con_file.exists():
                shutil.copy(con_file, tmpdir)
            # In windows `pip install pip==x.y.z` causes the following error:
            # `[WinError 5] Access is denied: 'C:\path\to\pip.exe`
            # This can be avoided by using `python -m`.
            cmd = _join_commands(activate_cmd, f"python -m pip install -r {tmp_req_file}")
            _exec_cmd(cmd, capture_output=False)

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
