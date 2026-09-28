import argparse
import os
import shutil
import threading
from pathlib import Path

from mlflow.server.jobs.utils import MLFLOW_ORIGINAL_PARENT_PID_ENV_VAR, _exit_when_orphaned
from mlflow.utils.file_utils import ExclusiveFileLock
from mlflow.utils.process import _exec_cmd, _join_commands
from mlflow.utils.virtualenv import (
    _get_uv_env_creation_command,
    _get_virtualenv_activate_cmd,
    _get_virtualenv_extra_env_vars,
)


def _get_incomplete_marker_path(env_dir: Path) -> Path:
    return env_dir.with_name(f".{env_dir.name}.mlflow-job-setup-incomplete")


def _pip_install(env_dir: Path, requirements_file: Path) -> None:
    install_command = _join_commands(
        _get_virtualenv_activate_cmd(env_dir),
        f"uv pip install -r {requirements_file.name}",
    )
    _exec_cmd(
        install_command,
        cwd=str(requirements_file.parent),
        extra_env=_get_virtualenv_extra_env_vars(),
        capture_output=False,
    )


def _setup_job_environment(
    env_dir: Path,
    python_version: str,
    requirements_file: Path,
    build_requirements_file: Path | None = None,
) -> None:
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete_marker = _get_incomplete_marker_path(env_dir)

    with ExclusiveFileLock(f"{env_dir}.lock"):
        if env_dir.exists() and not incomplete_marker.exists():
            return

        if env_dir.exists():
            shutil.rmtree(env_dir)
        incomplete_marker.touch()

        try:
            _exec_cmd(
                _get_uv_env_creation_command(env_dir, python_version),
                capture_output=False,
            )
            # Install build dependencies (pinned pip/setuptools/wheel, etc.)
            # before the regular dependencies, mirroring the canonical MLflow
            # environment-restore ordering.
            if build_requirements_file is not None:
                _pip_install(env_dir, build_requirements_file)
            _pip_install(env_dir, requirements_file)
        except Exception:
            if env_dir.exists():
                shutil.rmtree(env_dir)
            incomplete_marker.unlink(missing_ok=True)
            raise
        else:
            incomplete_marker.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-dir", type=Path, required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--requirements-file", type=Path, required=True)
    parser.add_argument("--build-requirements-file", type=Path, default=None)
    args = parser.parse_args()

    # Exit (releasing the environment lock) if the server that launched this
    # setup process dies and the process is reparented. The lock is held via
    # fcntl.flock, which the kernel releases automatically when the process
    # exits, so a stuck setup cannot block future jobs after a server crash.
    if os.environ.get(MLFLOW_ORIGINAL_PARENT_PID_ENV_VAR):
        threading.Thread(
            target=_exit_when_orphaned,
            name="exit_when_orphaned",
            daemon=True,
        ).start()

    _setup_job_environment(
        args.env_dir,
        args.python_version,
        args.requirements_file,
        args.build_requirements_file,
    )


if __name__ == "__main__":
    main()
