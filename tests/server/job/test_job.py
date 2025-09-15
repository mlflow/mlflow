import os
import shutil
import tempfile
import time
from os.path import dirname
from contextlib import contextmanager
from mlflow.server import BACKEND_STORE_URI_ENV_VAR, HUEY_STORAGE_PATH_ENV_VAR, ARTIFACT_ROOT_ENV_VAR
from mlflow.server.job import _start_job_runner, submit_job, query_job
from mlflow.entities._job_status import JobStatus


@contextmanager
def _setup_job_queue(max_job_parallelism, monkeypatch):
    tmp_dir = tempfile.mkdtemp()
    backend_store_uri = f"sqlite:///{os.path.join(tmp_dir, 'mlflow.db')}"
    huey_store_path = os.path.join(tmp_dir, "huey.db")
    default_artifact_root = os.path.join(tmp_dir, "artifacts")
    try:
        monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
        monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, backend_store_uri)
        monkeypatch.setenv(ARTIFACT_ROOT_ENV_VAR, default_artifact_root)
        monkeypatch.setenv(HUEY_STORAGE_PATH_ENV_VAR, huey_store_path)
        job_runner_proc = _start_job_runner(
            {
                "_IS_MLFLOW_JOB_RUNNER": "1",
                "PYTHONPATH": dirname(__file__),
            },
            max_job_parallelism,
            os.getpid(),
        )
        time.sleep(5)  # wait for huey consumer to spin up.
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)
        job_runner_proc.terminate()


def basic_job_fun(x, y):
    return x + y


def test_basic_job(monkeypatch):
    with _setup_job_queue(1, monkeypatch):
        job_id = submit_job(basic_job_fun, x=3, y=4)
        time.sleep(3)
        status, result = query_job(job_id)
        assert status == JobStatus.DONE
        assert result == 7
