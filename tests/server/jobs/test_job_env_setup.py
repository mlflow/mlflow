import os
from pathlib import Path

import pytest

from mlflow.server.jobs._job_env_setup import (
    _get_incomplete_marker_path,
    _setup_job_environment,
)
from mlflow.server.jobs.utils import _prepare_job_subprocess
from mlflow.utils.environment import _PythonEnv

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="MLflow job environment setup is not supported on Windows",
)


def test_setup_job_environment_replaces_incomplete_environment(
    monkeypatch,
    tmp_path: Path,
):
    env_dir = tmp_path / "env"
    incomplete_marker = _get_incomplete_marker_path(env_dir)
    env_dir.mkdir()
    (env_dir / "stale").touch()
    incomplete_marker.touch()
    requirements_file = tmp_path / "requirements.txt"
    requirements_file.write_text("pytest<9")
    call_count = 0

    def _exec_cmd(command, **kwargs):
        nonlocal call_count
        call_count += 1
        assert not (env_dir / "stale").exists()
        assert incomplete_marker.exists()
        if call_count == 1:
            (env_dir / "bin").mkdir(parents=True)
            (env_dir / "bin" / "activate").touch()
        else:
            (env_dir / "installed").touch()

    monkeypatch.setattr("mlflow.server.jobs._job_env_setup._exec_cmd", _exec_cmd)

    _setup_job_environment(env_dir, "3.11", requirements_file)

    assert call_count == 2
    assert (env_dir / "installed").exists()
    assert not incomplete_marker.exists()


def test_setup_job_environment_removes_partial_environment_after_failure(
    monkeypatch,
    tmp_path: Path,
):
    env_dir = tmp_path / "env"
    incomplete_marker = _get_incomplete_marker_path(env_dir)
    requirements_file = tmp_path / "requirements.txt"
    requirements_file.write_text("pytest<9")
    call_count = 0

    def _exec_cmd(command, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            (env_dir / "bin").mkdir(parents=True)
            (env_dir / "bin" / "activate").touch()
        else:
            raise RuntimeError("installation failed")

    monkeypatch.setattr("mlflow.server.jobs._job_env_setup._exec_cmd", _exec_cmd)

    with pytest.raises(RuntimeError, match="installation failed"):
        _setup_job_environment(env_dir, "3.11", requirements_file)

    assert not env_dir.exists()
    assert not incomplete_marker.exists()


def test_setup_job_environment_removes_partial_environment_when_requirements_fail_after_build_deps(
    monkeypatch,
    tmp_path: Path,
):
    env_dir = tmp_path / "env"
    incomplete_marker = _get_incomplete_marker_path(env_dir)
    requirements_file = tmp_path / "requirements.txt"
    requirements_file.write_text("pytest<9")
    build_requirements_file = tmp_path / "build-requirements.txt"
    build_requirements_file.write_text("pip==24.0")
    call_count = 0

    def _exec_cmd(command, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            (env_dir / "bin").mkdir(parents=True)
            (env_dir / "bin" / "activate").touch()
        elif call_count == 3:
            # Build deps (call 2) succeed; the regular deps install fails.
            raise RuntimeError("installation failed")

    monkeypatch.setattr("mlflow.server.jobs._job_env_setup._exec_cmd", _exec_cmd)

    with pytest.raises(RuntimeError, match="installation failed"):
        _setup_job_environment(env_dir, "3.11", requirements_file, build_requirements_file)

    assert call_count == 3
    assert not env_dir.exists()
    assert not incomplete_marker.exists()


def test_prepare_job_subprocess_retries_environment_marked_incomplete(
    monkeypatch,
    tmp_path: Path,
):
    env_root = tmp_path / "envs"
    env_dir = env_root / "test-env"
    env_dir.mkdir(parents=True)
    incomplete_marker = _get_incomplete_marker_path(env_dir)
    incomplete_marker.touch()
    monkeypatch.setattr("mlflow.server.jobs.utils.shutil.which", lambda command: "/usr/bin/uv")
    monkeypatch.setattr(
        "mlflow.utils.virtualenv._get_mlflow_virtualenv_root",
        lambda: str(env_root),
    )
    monkeypatch.setattr(
        "mlflow.utils.virtualenv._get_virtualenv_name",
        lambda python_env, work_dir_path: env_dir.name,
    )
    python_env = _PythonEnv(
        python="3.11",
        build_dependencies=[],
        dependencies=["pytest<9"],
    )

    incomplete = _prepare_job_subprocess(
        function_fullname="tests.server.jobs.test_jobs.basic_job_fun",
        params={"x": 1, "y": 2},
        python_env=python_env,
        transient_error_classes=None,
        tmpdir=str(tmp_path),
        job_id="job-1",
        job_name="basic_job_fun",
        workspace=None,
    )

    assert len(incomplete.setup_commands) == 1
    assert "mlflow.server.jobs._job_env_setup" in incomplete.setup_commands[0].command

    incomplete_marker.unlink()
    complete = _prepare_job_subprocess(
        function_fullname="tests.server.jobs.test_jobs.basic_job_fun",
        params={"x": 1, "y": 2},
        python_env=python_env,
        transient_error_classes=None,
        tmpdir=str(tmp_path),
        job_id="job-2",
        job_name="basic_job_fun",
        workspace=None,
    )

    assert complete.setup_commands == ()
