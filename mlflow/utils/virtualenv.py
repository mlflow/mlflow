import logging
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

from packaging.version import Version

import mlflow
from mlflow.environment_variables import MLFLOW_ENV_ROOT
from mlflow.exceptions import MlflowException
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.utils.conda import _PIP_CACHE_DIR
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _get_mlflow_env_name,
    _get_pip_install_mlflow,
    _PythonEnv,
)
from mlflow.utils.file_utils import remove_on_error
from mlflow.utils.os import is_windows
from mlflow.utils.process import _exec_cmd, _join_commands
from mlflow.utils.requirements_utils import _parse_requirements

_logger = logging.getLogger(__name__)


def _get_mlflow_virtualenv_root():
    """
    Returns the root directory to store virtualenv environments created by MLflow.
    """
    return MLFLOW_ENV_ROOT.get()


_DATABRICKS_PYENV_BIN_PATH = "/databricks/.pyenv/bin/pyenv"


def _is_pyenv_available():
    """
    Returns True if pyenv is available, otherwise False.
    """
    return _get_pyenv_bin_path() is not None


def _validate_pyenv_is_available():
    """
    Validates pyenv is available. If not, throws an `MlflowException` with a brief instruction on
    how to install pyenv.
    """
    url = (
        "https://github.com/pyenv/pyenv#installation"
        if not is_windows()
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


def _get_pyenv_bin_path():
    if os.path.exists(_DATABRICKS_PYENV_BIN_PATH):
        return _DATABRICKS_PYENV_BIN_PATH
    return shutil.which("pyenv")


def _find_latest_installable_python_version(version_prefix):
    """
    Find the latest installable python version that matches the given version prefix
    from the output of `pyenv install --list`. For example, `version_prefix("3.8")` returns '3.8.x'
    where 'x' represents the latest micro version in 3.8.
    """
    lines = _exec_cmd(
        [_get_pyenv_bin_path(), "install", "--list"],
        capture_output=True,
        shell=is_windows(),
    ).stdout.splitlines()
    semantic_versions = filter(_SEMANTIC_VERSION_REGEX.match, map(str.strip, lines))
    matched = [v for v in semantic_versions if v.startswith(version_prefix)]
    if not matched:
        raise MlflowException(f"Could not find python version that matches {version_prefix}")
    return sorted(matched, key=Version)[-1]


def _install_python(version, pyenv_root=None, capture_output=False):
    """Installs a specified version of python with pyenv and returns a path to the installed python
    binary.

    Args:
        version: Python version to install.
        pyenv_root: The value of the "PYENV_ROOT" environment variable used when running
            `pyenv install` which installs python in `{PYENV_ROOT}/versions/{version}`.
        capture_output: Set the `capture_output` argument when calling `_exec_cmd`.

    Returns:
        Path to the installed python binary.
    """
    version = (
        version
        if _SEMANTIC_VERSION_REGEX.match(version)
        else _find_latest_installable_python_version(version)
    )
    _logger.info("Installing python %s if it does not exist", version)
    # pyenv-win doesn't support `--skip-existing` but its behavior is enabled by default
    # https://github.com/pyenv-win/pyenv-win/pull/314
    pyenv_install_options = ("--skip-existing",) if not is_windows() else ()
    extra_env = {"PYENV_ROOT": pyenv_root} if pyenv_root else None
    pyenv_bin_path = _get_pyenv_bin_path()
    _exec_cmd(
        [pyenv_bin_path, "install", *pyenv_install_options, version],
        capture_output=capture_output,
        # Windows fails to find pyenv and throws `FileNotFoundError` without `shell=True`
        shell=is_windows(),
        extra_env=extra_env,
    )

    if not is_windows():
        if pyenv_root is None:
            pyenv_root = _exec_cmd([pyenv_bin_path, "root"], capture_output=True).stdout.strip()
        path_to_bin = ("bin", "python")
    else:
        # pyenv-win doesn't provide the `pyenv root` command
        pyenv_root = os.getenv("PYENV_ROOT")
        if pyenv_root is None:
            raise MlflowException("Environment variable 'PYENV_ROOT' must be set")
        path_to_bin = ("python.exe",)
    return Path(pyenv_root).joinpath("versions", version, *path_to_bin)


def _get_conda_env_file(model_config):
    from mlflow.pyfunc import _extract_conda_env

    for flavor, config in model_config.flavors.items():
        if flavor == mlflow.pyfunc.FLAVOR_NAME:
            env = config.get(mlflow.pyfunc.ENV)
            if env:
                return _extract_conda_env(env)
    return _CONDA_ENV_FILE_NAME


def _get_python_env_file(model_config):
    from mlflow.pyfunc import EnvType

    for flavor, config in model_config.flavors.items():
        if flavor == mlflow.pyfunc.FLAVOR_NAME:
            env = config.get(mlflow.pyfunc.ENV)
            if isinstance(env, dict):
                # Models saved in MLflow >= 2.0 use a dictionary for the pyfunc flavor
                # `env` config, where the keys are different environment managers (e.g.
                # conda, virtualenv) and the values are corresponding environment paths
                return env[EnvType.VIRTUALENV]
    return _PYTHON_ENV_FILE_NAME


def _get_python_env(local_model_path):
    """Constructs `_PythonEnv` from the model artifacts stored in `local_model_path`. If
    `python_env.yaml` is available, use it, otherwise extract model dependencies from `conda.yaml`.
    If `conda.yaml` contains conda dependencies except `python`, `pip`, `setuptools`, and, `wheel`,
    an `MlflowException` is thrown because conda dependencies cannot be installed in a virtualenv
    environment.

    Args:
        local_model_path: Local directory containing the model artifacts.

    Returns:
        `_PythonEnv` instance.

    """
    model_config = Model.load(local_model_path / MLMODEL_FILE_NAME)
    python_env_file = local_model_path / _get_python_env_file(model_config)
    conda_env_file = local_model_path / _get_conda_env_file(model_config)
    requirements_file = local_model_path / _REQUIREMENTS_FILE_NAME

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
        if requirements_file.exists():
            deps = _PythonEnv.get_dependencies_from_conda_yaml(conda_env_file)
            return _PythonEnv(
                python=deps["python"],
                build_dependencies=deps["build_dependencies"],
                dependencies=[f"-r {_REQUIREMENTS_FILE_NAME}"],
            )
        else:
            return _PythonEnv.from_conda_yaml(conda_env_file)


def _get_virtualenv_name(python_env, work_dir_path, env_id=None):
    requirements = _parse_requirements(
        python_env.dependencies,
        is_constraint=False,
        base_dir=work_dir_path,
    )
    return _get_mlflow_env_name(
        str(python_env) + "".join(map(str, sorted(requirements))) + (env_id or "")
    )


def _create_virtualenv(
    local_model_path, python_bin_path, env_dir, python_env, extra_env=None, capture_output=False
):
    # Created a command to activate the environment
    paths = ("bin", "activate") if not is_windows() else ("Scripts", "activate.bat")
    activate_cmd = env_dir.joinpath(*paths)
    activate_cmd = f"source {activate_cmd}" if not is_windows() else str(activate_cmd)

    if env_dir.exists():
        _logger.info("Environment %s already exists", env_dir)
        return activate_cmd

    with remove_on_error(
        env_dir,
        onerror=lambda e: _logger.warning(
            "Encountered an unexpected error: %s while creating a virtualenv environment in %s, "
            "removing the environment directory...",
            repr(e),
            env_dir,
        ),
    ):
        _logger.info("Creating a new environment in %s with %s", env_dir, python_bin_path)
        _exec_cmd(
            [sys.executable, "-m", "virtualenv", "--python", python_bin_path, env_dir],
            capture_output=capture_output,
        )

        _logger.info("Installing dependencies")
        for deps in filter(None, [python_env.build_dependencies, python_env.dependencies]):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a temporary requirements file in the model directory to resolve the
                # references in it correctly. To do this, we must first symlink or copy the model
                # directory's contents to a temporary location for compatibility with deployment
                # tools that store models in a read-only mount
                try:
                    for model_item in os.listdir(local_model_path):
                        os.symlink(
                            src=os.path.join(local_model_path, model_item),
                            dst=os.path.join(tmpdir, model_item),
                        )
                except Exception as e:
                    _logger.warning(
                        "Failed to symlink model directory during dependency installation"
                        " Copying instead. Exception: %s",
                        e,
                    )
                    _copy_model_to_writeable_destination(local_model_path, tmpdir)

                tmp_req_file = f"requirements.{uuid.uuid4().hex}.txt"
                Path(tmpdir).joinpath(tmp_req_file).write_text("\n".join(deps))
                cmd = _join_commands(activate_cmd, f"python -m pip install -r {tmp_req_file}")
                _exec_cmd(cmd, capture_output=capture_output, cwd=tmpdir, extra_env=extra_env)

    return activate_cmd


def _copy_model_to_writeable_destination(model_src, dst):
    """
    Copies the specified `model_src` directory, which may be read-only, to the writeable `dst`
    directory.
    """
    os.makedirs(dst, exist_ok=True)
    for model_item in os.listdir(model_src):
        # Copy individual files and subdirectories, rather than using `shutil.copytree()`
        # because `shutil.copytree()` will apply the permissions from the source directory,
        # which may be read-only
        copy_fn = shutil.copytree if os.path.isdir(model_item) else shutil.copy2

        copy_fn(
            src=os.path.join(model_src, model_item),
            dst=os.path.join(dst, model_item),
        )


def _get_virtualenv_extra_env_vars(env_root_dir=None):
    extra_env = {
        # PIP_NO_INPUT=1 makes pip run in non-interactive mode,
        # otherwise pip might prompt "yes or no" and ask stdin input
        "PIP_NO_INPUT": "1",
    }
    if env_root_dir is not None:
        # Note: Both conda pip and virtualenv can use the pip cache directory.
        extra_env["PIP_CACHE_DIR"] = os.path.join(env_root_dir, _PIP_CACHE_DIR)
    return extra_env


_VIRTUALENV_ENVS_DIR = "virtualenv_envs"
_PYENV_ROOT_DIR = "pyenv_root"


def _get_or_create_virtualenv(  # noqa: D417
    local_model_path,
    env_id=None,
    env_root_dir=None,
    capture_output=False,
    pip_requirements_override=None,
):
    """Restores an MLflow model's environment with pyenv and virtualenv and returns a command
    to activate it.

    Args:
        local_model_path: Local directory containing the model artifacts.
        env_id: Optional string that is added to the contents of the yaml file before
            calculating the hash. It can be used to distinguish environments that have the
            same conda dependencies but are supposed to be different based on the context.
            For example, when serving the model we may install additional dependencies to the
            environment after the environment has been activated.
        pip_requirements_override: If specified, install the specified python dependencies to
            the environment (upgrade if already installed).

    Returns:
        Command to activate the created virtualenv environment
        (e.g. "source /path/to/bin/activate").

    """
    _validate_pyenv_is_available()
    _validate_virtualenv_is_available()

    # Read environment information
    local_model_path = Path(local_model_path)
    python_env = _get_python_env(local_model_path)

    extra_env = _get_virtualenv_extra_env_vars(env_root_dir)
    if env_root_dir is not None:
        virtual_envs_root_path = Path(env_root_dir) / _VIRTUALENV_ENVS_DIR
        pyenv_root_path = Path(env_root_dir) / _PYENV_ROOT_DIR
        pyenv_root_path.mkdir(parents=True, exist_ok=True)
        pyenv_root_dir = str(pyenv_root_path)
    else:
        virtual_envs_root_path = Path(_get_mlflow_virtualenv_root())
        pyenv_root_dir = None

    virtual_envs_root_path.mkdir(parents=True, exist_ok=True)
    env_name = _get_virtualenv_name(python_env, local_model_path, env_id)
    env_dir = virtual_envs_root_path / env_name
    if env_dir.exists():
        paths = ("bin", "activate") if not is_windows() else ("Scripts", "activate.bat")
        activate_cmd = env_dir.joinpath(*paths)
        return f"source {activate_cmd}" if not is_windows() else str(activate_cmd)

    # Create an environment
    python_bin_path = _install_python(
        python_env.python, pyenv_root=pyenv_root_dir, capture_output=capture_output
    )
    try:
        activate_cmd = _create_virtualenv(
            local_model_path,
            python_bin_path,
            env_dir,
            python_env,
            extra_env=extra_env,
            capture_output=capture_output,
        )

        # Install additional dependencies specified by `requirements_override`
        if pip_requirements_override:
            _logger.info(
                "Installing additional dependencies specified by "
                f"pip_requirements_override: {pip_requirements_override}"
            )
            cmd = _join_commands(
                activate_cmd,
                f"python -m pip install --quiet -U {' '.join(pip_requirements_override)}",
            )
            _exec_cmd(cmd, capture_output=capture_output, extra_env=extra_env)

        return activate_cmd

    except:
        _logger.warning("Encountered unexpected error while creating %s", env_dir)
        if env_dir.exists():
            _logger.warning("Attempting to remove %s", env_dir)
            shutil.rmtree(env_dir, ignore_errors=True)
            msg = "Failed to remove %s" if env_dir.exists() else "Successfully removed %s"
            _logger.warning(msg, env_dir)
        raise


def _execute_in_virtualenv(
    activate_cmd,
    command,
    install_mlflow,
    command_env=None,
    synchronous=True,
    capture_output=False,
    env_root_dir=None,
    **kwargs,
):
    """Runs a command in a specified virtualenv environment.

    Args:
        activate_cmd: Command to activate the virtualenv environment.
        command: Command to run in the virtualenv environment.
        install_mlflow: Flag to determine whether to install mlflow in the virtualenv
            environment.
        command_env: Environment variables passed to a process running the command.
        synchronous: Set the `synchronous` argument when calling `_exec_cmd`.
        capture_output: Set the `capture_output` argument when calling `_exec_cmd`.
        env_root_dir: See doc of PyFuncBackend constructor argument `env_root_dir`.
        kwargs: Set the `kwargs` argument when calling `_exec_cmd`.

    """
    if command_env is None:
        command_env = os.environ.copy()

    if env_root_dir is not None:
        command_env = {**command_env, **_get_virtualenv_extra_env_vars(env_root_dir)}

    pre_command = [activate_cmd]
    if install_mlflow:
        pre_command.append(_get_pip_install_mlflow())

    cmd = _join_commands(*pre_command, command)
    _logger.info("Running command: %s", " ".join(cmd))
    return _exec_cmd(
        cmd, capture_output=capture_output, env=command_env, synchronous=synchronous, **kwargs
    )
