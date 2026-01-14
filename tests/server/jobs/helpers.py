"""Shared test helpers for job execution tests."""

import os
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from mlflow.entities._job_status import JobStatus
from mlflow.server import (
    ARTIFACT_ROOT_ENV_VAR,
    BACKEND_STORE_URI_ENV_VAR,
    HUEY_STORAGE_PATH_ENV_VAR,
    handlers,
)
from mlflow.server.jobs import (
    _ALLOWED_JOB_NAME_LIST,
    _SUPPORTED_JOB_FUNCTION_LIST,
    get_job,
)
from mlflow.server.jobs.utils import _launch_job_runner
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore


def _get_mlflow_repo_home():
    root = str(Path(__file__).resolve().parents[3])
    return f"{root}{os.pathsep}{path}" if (path := os.environ.get("PYTHONPATH")) else root


@contextmanager
def _launch_job_runner_for_test():
    new_pythonpath = _get_mlflow_repo_home()
    with _launch_job_runner(
        {"PYTHONPATH": new_pythonpath},
        os.getpid(),
    ) as proc:
        try:
            yield proc
        finally:
            proc.kill()


@contextmanager
def _setup_job_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    supported_job_functions: list[str],
    allowed_job_names: list[str],
    backend_store_uri: str | None = None,
):
    backend_store_uri = backend_store_uri or f"sqlite:///{tmp_path / 'mlflow.db'}"
    huey_store_path = tmp_path / "huey_store"
    huey_store_path.mkdir()
    default_artifact_root = str(tmp_path / "artifacts")
    try:
        monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
        monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, backend_store_uri)
        monkeypatch.setenv(ARTIFACT_ROOT_ENV_VAR, default_artifact_root)
        monkeypatch.setenv(HUEY_STORAGE_PATH_ENV_VAR, str(huey_store_path))
        monkeypatch.setenv("_MLFLOW_SUPPORTED_JOB_FUNCTION_LIST", ",".join(supported_job_functions))
        monkeypatch.setenv("_MLFLOW_ALLOWED_JOB_NAME_LIST", ",".join(allowed_job_names))
        _SUPPORTED_JOB_FUNCTION_LIST.clear()
        _SUPPORTED_JOB_FUNCTION_LIST.extend(supported_job_functions)
        _ALLOWED_JOB_NAME_LIST.clear()
        _ALLOWED_JOB_NAME_LIST.extend(allowed_job_names)

        # Pre-initialize the database before launching the job runner subprocess
        # to prevent race conditions during concurrent Alembic migrations
        SqlAlchemyJobStore(backend_store_uri)

        with _launch_job_runner_for_test() as job_runner_proc:
            time.sleep(10)
            yield job_runner_proc
    finally:
        # Clear the huey instance cache AFTER killing the runner to ensure clean state for next test
        import mlflow.server.jobs.utils

        mlflow.server.jobs.utils._huey_instance_map.clear()
        if handlers._job_store is not None:
            # close all db connections and drops connection pool
            handlers._job_store.engine.dispose()
        handlers._job_store = None


def wait_job_finalize(job_id, timeout=60):
    beg_time = time.time()
    while time.time() - beg_time <= timeout:
        job = get_job(job_id)
        if JobStatus.is_finalized(job.status):
            return
        time.sleep(0.5)
    raise TimeoutError("The job is not finalized within the timeout.")
