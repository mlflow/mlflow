import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from multiprocessing import Pool as MultiProcPool
from os.path import dirname

import pytest
from pathlib import Path

import mlflow.server.handlers
from mlflow.entities._job_status import JobStatus
from mlflow.server import (
    ARTIFACT_ROOT_ENV_VAR,
    BACKEND_STORE_URI_ENV_VAR,
    HUEY_STORAGE_PATH_ENV_VAR,
)
from mlflow.server.handlers import _get_job_store
from mlflow.server.job import _reinit_huey_queue, _start_job_runner, query_job, submit_job

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="MLflow job execution is not supported on Windows"
)


def _start_job_runner_for_test(max_job_parallelism, start_new_runner):
    proc = _start_job_runner(
        {"PYTHONPATH": dirname(__file__)},
        max_job_parallelism,
        os.getpid(),
        start_new_runner,
    )
    time.sleep(5)  # wait for huey consumer to spin up.
    return proc


@contextmanager
def _setup_job_runner(max_job_parallelism, monkeypatch, backend_store_uri=None):
    with tempfile.TemporaryDirectory() as tmp_dir:
        backend_store_uri = backend_store_uri or f"sqlite:///{os.path.join(tmp_dir, 'mlflow.db')}"
        huey_store_path = os.path.join(tmp_dir, "huey.db")
        default_artifact_root = os.path.join(tmp_dir, "artifacts")
        try:
            monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
            monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, backend_store_uri)
            monkeypatch.setenv(ARTIFACT_ROOT_ENV_VAR, default_artifact_root)
            monkeypatch.setenv(HUEY_STORAGE_PATH_ENV_VAR, huey_store_path)
            job_runner_proc = _start_job_runner_for_test(max_job_parallelism, True)
            _reinit_huey_queue()
            yield job_runner_proc
        finally:
            mlflow.server.handlers._job_store = None
            job_runner_proc.kill()
            time.sleep(1)


def basic_job_fun(x, y, sleep_secs=0):
    if sleep_secs > 0:
        time.sleep(sleep_secs)
    return x + y


def test_basic_job(monkeypatch):
    with _setup_job_runner(1, monkeypatch):
        job = submit_job(basic_job_fun, {"x": 3, "y": 4})
        job_id = job.job_id
        time.sleep(1)
        status, result = query_job(job_id)
        assert status == JobStatus.DONE
        assert result == 7

        store = _get_job_store()
        job = store.get_job(job_id)

        # check database record correctness.
        assert job.job_id == job_id
        assert job.function_fullname == "test_job.basic_job_fun"
        assert job.params == '{"x": 3, "y": 4}'
        assert job.timeout is None
        assert job.result == "7"
        assert job.status == JobStatus.DONE
        assert job.retry_count == 0


def json_in_out_fun(data):
    x = data["x"]
    y = data["y"]
    return {"res": x + y}


def test_job_json_input_output(monkeypatch):
    with _setup_job_runner(1, monkeypatch):
        job = submit_job(json_in_out_fun, {"data": {"x": 3, "y": 4}})
        job_id = job.job_id
        time.sleep(1)
        status, result = query_job(job_id)
        assert status == JobStatus.DONE
        assert result == {"res": 7}

        store = _get_job_store()
        job = store.get_job(job_id)

        # check database record correctness.
        assert job.job_id == job_id
        assert job.function_fullname == "test_job.json_in_out_fun"
        assert job.params == '{"data": {"x": 3, "y": 4}}'
        assert job.result == '{"res": 7}'
        assert job.status == JobStatus.DONE
        assert job.retry_count == 0


def err_fun(data):
    raise RuntimeError()


def test_error_job(monkeypatch):
    with _setup_job_runner(1, monkeypatch):
        job = submit_job(err_fun, {"data": None})
        job_id = job.job_id
        time.sleep(0.5)
        status, result = query_job(job_id)
        assert status == JobStatus.FAILED
        assert result == "RuntimeError()"

        store = _get_job_store()
        job = store.get_job(job_id)

        # check database record correctness.
        assert job.job_id == job_id
        assert job.function_fullname == "test_job.err_fun"
        assert job.params == '{"data": null}'
        assert job.result == "RuntimeError()"
        assert job.status == JobStatus.FAILED
        assert job.retry_count == 0


def test_job_resume_on_job_runner_restart(monkeypatch):
    with _setup_job_runner(1, monkeypatch) as job_runner_proc:
        job1_id = submit_job(basic_job_fun, {"x": 3, "y": 4, "sleep_secs": 0}).job_id
        job2_id = submit_job(basic_job_fun, {"x": 5, "y": 6, "sleep_secs": 2}).job_id
        job3_id = submit_job(basic_job_fun, {"x": 7, "y": 8, "sleep_secs": 0}).job_id
        time.sleep(1)
        job_runner_proc.kill()
        job_runner_proc.wait()  # ensure the job runner process is killed.

        # assert that job1 has done, job2 is running, and job3 is pending.
        assert query_job(job1_id) == (JobStatus.DONE, 7)
        assert query_job(job2_id) == (JobStatus.RUNNING, None)
        assert query_job(job3_id) == (JobStatus.PENDING, None)

        # restart the job runner, and verify it resumes unfinished jobs (job2 and job3)
        _start_job_runner_for_test(1, False)
        time.sleep(2.5)

        # assert all jobs are done.
        assert query_job(job1_id) == (JobStatus.DONE, 7)
        assert query_job(job2_id) == (JobStatus.DONE, 11)
        assert query_job(job3_id) == (JobStatus.DONE, 15)


def test_job_resume_on_new_job_runner(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        backend_store_uri = f"sqlite:///{os.path.join(tmp_dir, 'mlflow.db')}"

        with _setup_job_runner(1, monkeypatch, backend_store_uri) as job_runner_proc:
            job1_id = submit_job(basic_job_fun, {"x": 3, "y": 4, "sleep_secs": 0}).job_id
            job2_id = submit_job(basic_job_fun, {"x": 5, "y": 6, "sleep_secs": 10}).job_id
            job3_id = submit_job(basic_job_fun, {"x": 7, "y": 8, "sleep_secs": 0}).job_id
            time.sleep(1)

        # ensure the job runner process is killed.
        job_runner_proc.wait()

        with _setup_job_runner(1, monkeypatch, backend_store_uri):
            # assert that job1 has done, job2 is running, and job3 is pending.
            assert query_job(job1_id) == (JobStatus.DONE, 7)
            assert query_job(job2_id) == (JobStatus.RUNNING, None)
            assert query_job(job3_id) == (JobStatus.PENDING, None)
            time.sleep(10)
            # assert all jobs are done.
            assert query_job(job1_id) == (JobStatus.DONE, 7)
            assert query_job(job2_id) == (JobStatus.DONE, 11)
            assert query_job(job3_id) == (JobStatus.DONE, 15)


def test_job_queue_parallelism(monkeypatch):
    # test job queue parallelism=2 and each job consumes 2 seconds.
    with _setup_job_runner(2, monkeypatch):
        job_ids = [submit_job(basic_job_fun, {"x": x, "y": 1, "sleep_secs": 2}).job_id for x in range(4)]
        time.sleep(2.5)

        # assert that job1 and job2 are done, and job3 and job4 are running
        assert query_job(job_ids[0]) == (JobStatus.DONE, 1)
        assert query_job(job_ids[1]) == (JobStatus.DONE, 2)
        assert query_job(job_ids[2]) == (JobStatus.RUNNING, None)
        assert query_job(job_ids[3]) == (JobStatus.RUNNING, None)

        time.sleep(2.5)
        assert query_job(job_ids[2]) == (JobStatus.DONE, 3)
        assert query_job(job_ids[3]) == (JobStatus.DONE, 4)


def transient_err_fun(tmp_dir, succeed_on_nth_run):
    """
    This function will raise `Timeout` exception on the first (`succeed_on_nth_run` -1) runs.
    and return 100 on the `succeed_on_nth_run` run.
    the `tmp_dir` is for recording the function run status.
    """
    from mlflow.server.job import TransientError

    if len(os.listdir(tmp_dir)) == succeed_on_nth_run:
        return 100
    with open(os.path.join(tmp_dir, uuid.uuid4().hex), "w") as f:
        f.close()
    raise TransientError(RuntimeError("test transient error."))


def test_job_retry_on_transient_error(monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY", "1")
    with _setup_job_runner(1, monkeypatch):
        store = _get_job_store()
        with tempfile.TemporaryDirectory() as tmp_dir:
            job1_id = submit_job(transient_err_fun, {"tmp_dir": tmp_dir, "succeed_on_nth_run": 4}).job_id
            time.sleep(15)
            assert query_job(job1_id) == (JobStatus.FAILED, "RuntimeError('test transient error.')")
            job1 = store.get_job(job1_id)
            assert job1.status == JobStatus.FAILED
            assert job1.result == "RuntimeError('test transient error.')"
            assert job1.retry_count == 3

        with tempfile.TemporaryDirectory() as tmp_dir:
            job2_id = submit_job(transient_err_fun, {"tmp_dir": tmp_dir, "succeed_on_nth_run": 1}).job_id
            time.sleep(3)
            assert query_job(job2_id) == (JobStatus.DONE, 100)
            job2 = store.get_job(job2_id)
            assert job2.status == JobStatus.DONE
            assert job2.result == "100"
            assert job2.retry_count == 1


# `submit_job` API is designed to be called inside MLflow server handler,
# MLflow server handler might be executed in multiple MLflow server workers
# so that we need a test to cover the case that executes `submit_job` in
# multi-processes case.
def test_submit_jobs_from_multi_processes(monkeypatch):
    with _setup_job_runner(4, monkeypatch), MultiProcPool() as pool:
        async_res_list = [
            pool.apply_async(
                submit_job,
                args=(basic_job_fun,),
                kwds={"params": {"x": x, "y": 1, "sleep_secs": 2}},
            )
            for x in range(4)
        ]
        job_ids = [async_res.get().job_id for async_res in async_res_list]
        time.sleep(3)
        for x in range(4):
            assert query_job(job_ids[x]) == (JobStatus.DONE, x + 1)


def sleep_fun(sleep_secs, tmp_dir):
    (Path(tmp_dir) / "pid").write_text(str(os.getpid()))
    time.sleep(sleep_secs)


def test_job_timeout(monkeypatch):
    from mlflow.server.job.util import is_process_alive

    with _setup_job_runner(1, monkeypatch) as job_runner_proc, \
            tempfile.TemporaryDirectory() as tmp_dir:
        job_id = submit_job(sleep_fun, {"sleep_secs": 10, "tmp_dir": tmp_dir}, timeout=5).job_id
        time.sleep(6)
        pid = int((Path(tmp_dir) / "pid").read_text())
        # assert timeout job process is killed.
        assert not is_process_alive(pid)

        status, result = query_job(job_id)
        assert status == JobStatus.TIMEOUT
        assert result is None

        store = _get_job_store()
        job = store.get_job(job_id)

        # check database record correctness.
        assert job.job_id == job_id
        assert job.function_fullname == "test_job.sleep_fun"
        assert job.timeout == 5
        assert job.result is None
        assert job.status == JobStatus.TIMEOUT
        assert job.retry_count == 0

        submit_job(sleep_fun, {"sleep_secs": 10, "tmp_dir": tmp_dir}, timeout=15)
        time.sleep(5)
        pid = int((Path(tmp_dir) / "pid").read_text())
        assert is_process_alive(pid)
        job_runner_proc.kill()
        time.sleep(2)
        # assert the job process is killed after job runner process is killed.
        assert not is_process_alive(pid)


def test_list_job_pagination(monkeypatch):
    with _setup_job_runner(1, monkeypatch):
        job_ids = []
        for x in range(10):
            job_id = submit_job(basic_job_fun, {"x": x, "y": 4}).job_id
            job_ids.append(job_id)

        listed_jobs = _get_job_store().list_jobs(page_size=3)
        assert [job.job_id for job in listed_jobs] == job_ids
