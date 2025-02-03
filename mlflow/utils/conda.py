import hashlib
import json
import logging
import os

import yaml

from mlflow.environment_variables import MLFLOW_CONDA_CREATE_ENV_CMD, MLFLOW_CONDA_HOME
from mlflow.exceptions import ExecutionException
from mlflow.utils import process
from mlflow.utils.environment import Environment
from mlflow.utils.os import is_windows

_logger = logging.getLogger(__name__)

CONDA_EXE = "CONDA_EXE"


def get_conda_command(conda_env_name):
    #  Checking for newer conda versions
    if not is_windows() and (CONDA_EXE in os.environ or MLFLOW_CONDA_HOME.defined):
        conda_path = get_conda_bin_executable("conda")
        activate_conda_env = [f"source {os.path.dirname(conda_path)}/../etc/profile.d/conda.sh"]
        activate_conda_env += [f"conda activate {conda_env_name} 1>&2"]
    else:
        activate_path = get_conda_bin_executable("activate")
        # in case os name is not 'nt', we are not running on windows. It introduces
        # bash command otherwise.
        if not is_windows():
            return [f"source {activate_path} {conda_env_name} 1>&2"]
        else:
            return [f"conda activate {conda_env_name}"]
    return activate_conda_env


def get_conda_bin_executable(executable_name):
    """
    Return path to the specified executable, assumed to be discoverable within the 'bin'
    subdirectory of a conda installation.

    The conda home directory (expected to contain a 'bin' subdirectory) is configurable via the
    ``mlflow.projects.MLFLOW_CONDA_HOME`` environment variable. If
    ``mlflow.projects.MLFLOW_CONDA_HOME`` is unspecified, this method simply returns the passed-in
    executable name.
    """
    if conda_home := MLFLOW_CONDA_HOME.get():
        return os.path.join(conda_home, f"bin/{executable_name}")
    # Use CONDA_EXE as per https://github.com/conda/conda/issues/7126
    if conda_exe := os.getenv(CONDA_EXE):
        conda_bin_dir = os.path.dirname(conda_exe)
        return os.path.join(conda_bin_dir, executable_name)
    return executable_name


def _get_conda_env_name(conda_env_path, env_id=None, env_root_dir=None):
    if conda_env_path:
        with open(conda_env_path) as f:
            conda_env_contents = f.read()
    else:
        conda_env_contents = ""

    if env_id:
        conda_env_contents += env_id

    env_name = "mlflow-{}".format(
        hashlib.sha1(conda_env_contents.encode("utf-8"), usedforsecurity=False).hexdigest()
    )
    if env_root_dir:
        env_root_dir = os.path.normpath(env_root_dir)
        # Generate env name with format "mlflow-{conda_env_contents_hash}-{env_root_dir_hash}"
        # hashing `conda_env_contents` and `env_root_dir` separately helps debugging
        env_name += "-{}".format(
            hashlib.sha1(env_root_dir.encode("utf-8"), usedforsecurity=False).hexdigest()
        )

    return env_name


def _get_conda_executable_for_create_env():
    """
    Returns the executable that should be used to create environments. This is "conda"
    by default, but it can be set to something else by setting the environment variable

    """

    return get_conda_bin_executable(MLFLOW_CONDA_CREATE_ENV_CMD.get())


def _list_conda_environments(extra_env=None):
    """Return a list of names of conda environments.

    Args:
        extra_env: Extra environment variables for running "conda env list" command.

    """
    prc = process._exec_cmd(
        [get_conda_bin_executable("conda"), "env", "list", "--json"], extra_env=extra_env
    )
    return list(map(os.path.basename, json.loads(prc.stdout).get("envs", [])))


_CONDA_ENVS_DIR = "conda_envs"
_CONDA_CACHE_PKGS_DIR = "conda_cache_pkgs"
_PIP_CACHE_DIR = "pip_cache_pkgs"


def _create_conda_env(
    conda_env_path,
    conda_env_create_path,
    project_env_name,
    conda_extra_env_vars,
    capture_output,
):
    if conda_env_path:
        process._exec_cmd(
            [
                conda_env_create_path,
                "env",
                "create",
                "-n",
                project_env_name,
                "--file",
                conda_env_path,
            ],
            extra_env=conda_extra_env_vars,
            capture_output=capture_output,
        )
    else:
        process._exec_cmd(
            [
                conda_env_create_path,
                "create",
                "--channel",
                "conda-forge",
                "--yes",
                "--override-channels",
                "-n",
                project_env_name,
                "python",
            ],
            extra_env=conda_extra_env_vars,
            capture_output=capture_output,
        )

    return Environment(get_conda_command(project_env_name), conda_extra_env_vars)


def _create_conda_env_retry(
    conda_env_path, conda_env_create_path, project_env_name, conda_extra_env_vars, _capture_output
):
    """
    `conda env create` command can fail due to network issues such as `ConnectionResetError`
    while collecting package metadata. This function retries the command up to 3 times.
    """
    num_attempts = 3
    for attempt in range(num_attempts):
        try:
            return _create_conda_env(
                conda_env_path,
                conda_env_create_path,
                project_env_name,
                conda_extra_env_vars,
                capture_output=True,
            )
        except process.ShellCommandException as e:
            error_str = str(e)
            if (num_attempts - attempt - 1) > 0 and (
                "ConnectionResetError" in error_str or "ChunkedEncodingError" in error_str
            ):
                _logger.warning("Conda env creation failed due to network issue. Retrying...")
                continue
            raise


def _get_conda_extra_env_vars(env_root_dir=None):
    """
    Given the `env_root_dir` (See doc of PyFuncBackend constructor argument `env_root_dir`),
    return a dict of environment variables which are used to config conda to generate envs
    under the expected `env_root_dir`.
    """
    if env_root_dir is None:
        return None

    # Create isolated conda package cache dir "conda_pkgs" under the env_root_dir
    # for each python process.
    # Note: shared conda package cache dir causes race condition issues:
    # See https://github.com/conda/conda/issues/8870
    # See https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#specify-environment-directories-envs-dirs
    # and https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#specify-package-directories-pkgs-dirs

    conda_envs_path = os.path.join(env_root_dir, _CONDA_ENVS_DIR)
    conda_pkgs_path = os.path.join(env_root_dir, _CONDA_CACHE_PKGS_DIR)
    pip_cache_dir = os.path.join(env_root_dir, _PIP_CACHE_DIR)

    os.makedirs(conda_envs_path, exist_ok=True)
    os.makedirs(conda_pkgs_path, exist_ok=True)
    os.makedirs(pip_cache_dir, exist_ok=True)

    return {
        "CONDA_ENVS_PATH": conda_envs_path,
        "CONDA_PKGS_DIRS": conda_pkgs_path,
        "PIP_CACHE_DIR": pip_cache_dir,
        # PIP_NO_INPUT=1 makes pip run in non-interactive mode,
        # otherwise pip might prompt "yes or no" and ask stdin input
        "PIP_NO_INPUT": "1",
    }


def get_or_create_conda_env(
    conda_env_path,
    env_id=None,
    capture_output=False,
    env_root_dir=None,
    pip_requirements_override=None,
    extra_envs=None,
):
    """Given a `Project`, creates a conda environment containing the project's dependencies if such
    a conda environment doesn't already exist. Returns the name of the conda environment.

    Args:
        conda_env_path: Path to a conda yaml file.
        env_id: Optional string that is added to the contents of the yaml file before
            calculating the hash. It can be used to distinguish environments that have the
            same conda dependencies but are supposed to be different based on the context.
            For example, when serving the model we may install additional dependencies to the
            environment after the environment has been activated.
        capture_output: Specify the capture_output argument while executing the
            "conda env create" command.
        env_root_dir: See doc of PyFuncBackend constructor argument `env_root_dir`.
        pip_requirements_override: If specified, install the specified python dependencies to
            the environment (upgrade if already installed).
        extra_envs: If specified, a dictionary of extra environment variables will be passed to the
            model inference environment.

    Returns:
        The name of the conda environment.

    """

    conda_path = get_conda_bin_executable("conda")
    conda_env_create_path = _get_conda_executable_for_create_env()

    try:
        # Checks if Conda executable exists
        process._exec_cmd([conda_path, "--help"], throw_on_error=False, extra_env=extra_envs)
    except OSError:
        raise ExecutionException(
            f"Could not find Conda executable at {conda_path}. "
            "Ensure Conda is installed as per the instructions at "
            "https://conda.io/projects/conda/en/latest/"
            "user-guide/install/index.html. "
            "You can also configure MLflow to look for a specific "
            f"Conda executable by setting the {MLFLOW_CONDA_HOME} environment variable "
            "to the path of the Conda executable"
        )

    try:
        # Checks if executable for environment creation exists
        process._exec_cmd(
            [conda_env_create_path, "--help"], throw_on_error=False, extra_env=extra_envs
        )
    except OSError:
        raise ExecutionException(
            f"You have set the env variable {MLFLOW_CONDA_CREATE_ENV_CMD}, but "
            f"{conda_env_create_path} does not exist or it is not working properly. "
            f"Note that {conda_env_create_path} and the conda executable need to be "
            "in the same conda environment. You can change the search path by"
            f"modifying the env variable {MLFLOW_CONDA_HOME}"
        )

    conda_extra_env_vars = _get_conda_extra_env_vars(env_root_dir)
    if extra_envs:
        conda_extra_env_vars.update(extra_envs)

    # Include the env_root_dir hash in the project_env_name,
    # this is for avoid conda env name conflicts between different CONDA_ENVS_PATH.
    project_env_name = _get_conda_env_name(conda_env_path, env_id=env_id, env_root_dir=env_root_dir)
    if env_root_dir is not None:
        project_env_path = os.path.join(env_root_dir, _CONDA_ENVS_DIR, project_env_name)
    else:
        project_env_path = project_env_name

    if project_env_name in _list_conda_environments(conda_extra_env_vars):
        _logger.info("Conda environment %s already exists.", project_env_path)
        return Environment(get_conda_command(project_env_name), conda_extra_env_vars)

    _logger.info("=== Creating conda environment %s ===", project_env_path)
    try:
        _create_conda_env_func = (
            # Retry conda env creation in a pytest session to avoid flaky test failures
            _create_conda_env_retry if "PYTEST_CURRENT_TEST" in os.environ else _create_conda_env
        )
        conda_env = _create_conda_env_func(
            conda_env_path,
            conda_env_create_path,
            project_env_name,
            conda_extra_env_vars,
            capture_output,
        )

        if pip_requirements_override:
            _logger.info(
                "Installing additional dependencies specified"
                f"by pip_requirements_override: {pip_requirements_override}"
            )
            cmd = [
                conda_path,
                "install",
                "-n",
                project_env_name,
                "--yes",
                *pip_requirements_override,
            ]
            process._exec_cmd(cmd, extra_env=conda_extra_env_vars, capture_output=capture_output)

        return conda_env

    except Exception:
        try:
            if project_env_name in _list_conda_environments(conda_extra_env_vars):
                _logger.warning(
                    "Encountered unexpected error while creating conda environment. Removing %s.",
                    project_env_path,
                )
                process._exec_cmd(
                    [
                        conda_path,
                        "remove",
                        "--yes",
                        "--name",
                        project_env_name,
                        "--all",
                    ],
                    extra_env=conda_extra_env_vars,
                    capture_output=False,
                )
        except Exception as e:
            _logger.warning(
                "Removing conda environment %s failed (error: %s)",
                project_env_path,
                repr(e),
            )
        raise


def _get_conda_dependencies(conda_yaml_path):
    """Extracts conda dependencies from a conda yaml file.

    Args:
        conda_yaml_path: Conda yaml file path.
    """
    with open(conda_yaml_path) as f:
        conda_yaml = yaml.safe_load(f)
        return [d for d in conda_yaml.get("dependencies", []) if isinstance(d, str)]
