import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Literal

import mlflow
from mlflow.environment_variables import _MLFLOW_TESTING, MLFLOW_ENV_ROOT
from mlflow.exceptions import MlflowException
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.utils import env_manager as em
from mlflow.utils.conda import _PIP_CACHE_DIR
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _get_mlflow_env_name,
    _PythonEnv,
)
from mlflow.utils.file_utils import check_tarfile_security, remove_on_error
from mlflow.utils.os import is_windows
from mlflow.utils.process import _exec_cmd, _join_commands
from mlflow.utils.requirements_utils import _parse_requirements

_logger = logging.getLogger(__name__)


def _check_zipfile_security(archive_path: str) -> None:
    """Check zip file for path traversal vulnerabilities (zip slip)."""
    with zipfile.ZipFile(archive_path, "r") as zf:
        for name in zf.namelist():
            # Normalize path and check for absolute paths or directory traversal
            normalized = os.path.normpath(name)
            if normalized.startswith("/") or normalized.startswith(".."):
                raise MlflowException(
                    f"Unsafe path in zip archive: {name}. "
                    "Archive contains absolute or escaped paths."
                )


def _verify_checksum(file_path: Path, checksum_url: str) -> None:
    """Verify file integrity using SHA256 checksum from PBS release."""
    try:
        with urllib.request.urlopen(checksum_url, timeout=30) as resp:
            checksum_content = resp.read().decode()
    except Exception as e:
        _logger.warning("Could not download checksum file: %s", e)
        return

    # Parse checksum file (format: "sha256  filename")
    expected_hash = None
    filename = file_path.name
    for line in checksum_content.strip().split("\n"):
        if filename in line:
            expected_hash = line.split()[0]
            break

    if not expected_hash:
        _logger.warning("Checksum for %s not found in checksum file", filename)
        return

    # Calculate actual hash
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    actual_hash = sha256.hexdigest()

    if actual_hash != expected_hash:
        raise MlflowException(
            f"Checksum verification failed for {filename}. "
            f"Expected {expected_hash}, got {actual_hash}."
        )


def _get_mlflow_virtualenv_root():
    return MLFLOW_ENV_ROOT.get()


# PBS (python-build-standalone) configuration
_PBS_BASE_URL = "https://github.com/astral-sh/python-build-standalone/releases/download"
_PBS_INSTALL_DIR = Path.home() / ".mlflow" / "python"


def _is_virtualenv_available():
    return shutil.which("virtualenv") is not None


def _validate_virtualenv_is_available():
    if not _is_virtualenv_available():
        raise MlflowException(
            "Could not find the virtualenv binary. Run `pip install virtualenv` to install "
            "virtualenv."
        )


def _get_pbs_platform_tag():
    machine = platform.machine().lower()
    system = platform.system().lower()

    arch_map = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
    arch = arch_map.get(machine, machine)

    if system == "darwin":
        return f"{arch}-apple-darwin"
    elif system == "linux":
        return f"{arch}-unknown-linux-gnu"
    elif system == "windows":
        return f"{arch}-pc-windows-msvc"
    raise MlflowException(f"Unsupported platform: {system}-{machine}")


def _get_pbs_releases():
    url = "https://api.github.com/repos/astral-sh/python-build-standalone/releases?per_page=10"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _find_latest_installable_python_version(version_prefix):
    releases = _get_pbs_releases()
    platform_tag = _get_pbs_platform_tag()

    for release in releases:
        for asset in release.get("assets", []):
            name = asset["name"]
            if platform_tag not in name or "install_only" not in name:
                continue
            # Extract version from filename like cpython-3.10.14+20240713-...
            if match := re.search(r"cpython-(\d+\.\d+\.\d+)\+(\d+)", name):
                version = match.group(1)
                if version.startswith(version_prefix):
                    return version, release["tag_name"]

    raise MlflowException(f"Could not find Python version matching {version_prefix}")


def _install_python(version, pyenv_root=None, capture_output=False):
    install_dir = Path(pyenv_root) if pyenv_root else _PBS_INSTALL_DIR

    # Get the minor version prefix (e.g., "3.10" from "3.10.16" or "3.10")
    match version.rsplit(".", 1):
        case [major_minor, patch] if "." in major_minor and patch.isdigit():
            version_prefix = major_minor
        case [major, minor] if major.isdigit() and minor.isdigit():
            version_prefix = version
        case _:
            raise MlflowException(f"Invalid Python version: {version}")

    # Find the latest available version for this minor series from PBS
    version, release_tag = _find_latest_installable_python_version(version_prefix)

    python_dir = install_dir / version
    if is_windows():
        python_bin = python_dir / "python" / "python.exe"
    else:
        python_bin = python_dir / "python" / "bin" / "python3"

    if python_bin.exists():
        _logger.info("Python %s already installed at %s", version, python_bin)
        return python_bin

    _logger.info("Installing Python %s from python-build-standalone", version)

    platform_tag = _get_pbs_platform_tag()
    ext = "tar.gz" if not is_windows() else "zip"
    filename = f"cpython-{version}+{release_tag}-{platform_tag}-install_only.{ext}"
    url = f"{_PBS_BASE_URL}/{release_tag}/{filename}"

    python_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        _logger.info("Downloading %s", url)
        urllib.request.urlretrieve(url, tmp_path)

        # Verify checksum
        checksum_url = f"{_PBS_BASE_URL}/{release_tag}/SHA256SUMS"
        _verify_checksum(tmp_path, checksum_url)

        _logger.info("Extracting to %s", python_dir)
        if ext == "tar.gz":
            check_tarfile_security(tmp_path)
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(python_dir)
        else:
            _check_zipfile_security(tmp_path)
            with zipfile.ZipFile(tmp_path, "r") as z:
                z.extractall(python_dir)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not python_bin.exists():
        raise MlflowException(f"Python binary not found at {python_bin} after installation")

    return python_bin


def _get_conda_env_file(model_config):
    from mlflow.pyfunc import _extract_conda_env

    for flavor, config in model_config.flavors.items():
        if flavor == mlflow.pyfunc.FLAVOR_NAME:
            if env := config.get(mlflow.pyfunc.ENV):
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


def _get_virtualenv_activate_cmd(env_dir: Path) -> str:
    # Created a command to activate the environment
    paths = ("bin", "activate") if not is_windows() else ("Scripts", "activate.bat")
    activate_cmd = env_dir.joinpath(*paths)
    return f"source {activate_cmd}" if not is_windows() else str(activate_cmd)


def _get_uv_env_creation_command(env_dir: str | Path, python_version: str) -> str:
    return ["uv", "venv", str(env_dir), f"--python={python_version}"]


def _create_virtualenv(
    local_model_path: Path,
    python_env: _PythonEnv,
    env_dir: Path,
    python_install_dir: str | None = None,
    env_manager: Literal["virtualenv", "uv"] = em.UV,
    extra_env: dict[str, str] | None = None,
    capture_output: bool = False,
    pip_requirements_override: list[str] | None = None,
):
    if env_manager not in {em.VIRTUALENV, em.UV}:
        raise MlflowException.invalid_parameter_value(
            f"Invalid value for `env_manager`: {env_manager}. "
            f"Must be one of `{em.VIRTUALENV}, {em.UV}`"
        )

    activate_cmd = _get_virtualenv_activate_cmd(env_dir)
    if env_dir.exists():
        _logger.info(f"Environment {env_dir} already exists")
        return activate_cmd

    env_creation_extra_env = {}
    if env_manager == em.VIRTUALENV:
        python_bin_path = _install_python(
            python_env.python, pyenv_root=python_install_dir, capture_output=capture_output
        )
        _logger.info(f"Creating a new environment in {env_dir} with {python_bin_path}")
        env_creation_cmd = [
            sys.executable,
            "-m",
            "virtualenv",
            "--python",
            python_bin_path,
            env_dir,
        ]
        install_deps_cmd_prefix = "python -m pip install"
    elif env_manager == em.UV:
        _logger.info(
            f"Creating a new environment in {env_dir} with python "
            f"version {python_env.python} using uv"
        )
        env_creation_cmd = _get_uv_env_creation_command(env_dir, python_env.python)
        install_deps_cmd_prefix = "uv pip install"
        if python_install_dir:
            # Setting `UV_PYTHON_INSTALL_DIR` to make `uv env` install python into
            # the directory it points to.
            env_creation_extra_env["UV_PYTHON_INSTALL_DIR"] = python_install_dir
        if _MLFLOW_TESTING.get():
            os.environ["RUST_LOG"] = "uv=debug"
    with remove_on_error(
        env_dir,
        onerror=lambda e: _logger.warning(
            "Encountered an unexpected error: %s while creating a virtualenv environment in %s, "
            "removing the environment directory...",
            repr(e),
            env_dir,
        ),
    ):
        _exec_cmd(
            env_creation_cmd,
            capture_output=capture_output,
            extra_env=env_creation_extra_env,
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
                cmd = _join_commands(activate_cmd, f"{install_deps_cmd_prefix} -r {tmp_req_file}")
                _exec_cmd(cmd, capture_output=capture_output, cwd=tmpdir, extra_env=extra_env)

        if pip_requirements_override:
            _logger.info(
                "Installing additional dependencies specified by "
                f"pip_requirements_override: {pip_requirements_override}"
            )
            cmd = _join_commands(
                activate_cmd,
                f"{install_deps_cmd_prefix} --quiet {' '.join(pip_requirements_override)}",
            )
            _exec_cmd(cmd, capture_output=capture_output, extra_env=extra_env)

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


def _get_or_create_virtualenv(
    local_model_path,
    env_id=None,
    env_root_dir=None,
    capture_output=False,
    pip_requirements_override: list[str] | None = None,
    env_manager: Literal["virtualenv", "uv"] = em.UV,
    extra_envs: dict[str, str] | None = None,
):
    """Restores an MLflow model's environment in a virtual environment and returns a command
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
        env_manager: Specifies the environment manager to use to create the environment.
            Defaults to "uv".
        extra_envs: If specified, a dictionary of extra environment variables will be passed to the
            environment creation command.

            .. tip::
                It is highly recommended to use "uv" as it has significant performance improvements
                over "virtualenv".

    Returns:
        Command to activate the created virtual environment
        (e.g. "source /path/to/bin/activate").

    """
    if env_manager == em.VIRTUALENV:
        _validate_virtualenv_is_available()

    local_model_path = Path(local_model_path)
    python_env = _get_python_env(local_model_path)

    if env_root_dir is None:
        virtual_envs_root_path = Path(_get_mlflow_virtualenv_root())
        python_install_dir = None
    else:
        virtual_envs_root_path = Path(env_root_dir) / _VIRTUALENV_ENVS_DIR
        pyenv_root_path = Path(env_root_dir) / _PYENV_ROOT_DIR
        pyenv_root_path.mkdir(parents=True, exist_ok=True)
        python_install_dir = str(pyenv_root_path)

    virtual_envs_root_path.mkdir(parents=True, exist_ok=True)
    env_name = _get_virtualenv_name(python_env, local_model_path, env_id)
    env_dir = virtual_envs_root_path / env_name
    try:
        env_dir.exists()
    except PermissionError:
        if is_in_databricks_runtime():
            # Updating env_name only doesn't work because the cluster may not have
            # permission to access the original virtual_envs_root_path
            virtual_envs_root_path = (
                Path(env_root_dir) / f"{_VIRTUALENV_ENVS_DIR}_{uuid.uuid4().hex[:8]}"
            )
            virtual_envs_root_path.mkdir(parents=True, exist_ok=True)
            env_dir = virtual_envs_root_path / env_name
        else:
            _logger.warning(
                f"Existing virtual environment directory {env_dir} cannot be accessed "
                "due to permission error. Check the permissions of the directory and "
                "try again. If the issue persists, consider cleaning up the directory manually."
            )
            raise

    extra_envs = extra_envs or {}
    extra_envs |= _get_virtualenv_extra_env_vars(env_root_dir)

    # Create an environment
    return _create_virtualenv(
        local_model_path=local_model_path,
        python_env=python_env,
        env_dir=env_dir,
        python_install_dir=python_install_dir,
        env_manager=env_manager,
        extra_env=extra_envs,
        capture_output=capture_output,
        pip_requirements_override=pip_requirements_override,
    )
