import argparse
import shutil
from pathlib import Path

from mlflow.utils.file_utils import ExclusiveFileLock
from mlflow.utils.process import _exec_cmd, _join_commands
from mlflow.utils.virtualenv import (
    _get_uv_env_creation_command,
    _get_virtualenv_activate_cmd,
    _get_virtualenv_extra_env_vars,
)


def _get_incomplete_marker_path(env_dir: Path) -> Path:
    return env_dir.with_name(f".{env_dir.name}.mlflow-job-setup-incomplete")


def _setup_job_environment(
    env_dir: Path,
    python_version: str,
    requirements_file: Path,
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
    args = parser.parse_args()
    _setup_job_environment(args.env_dir, args.python_version, args.requirements_file)


if __name__ == "__main__":
    main()
