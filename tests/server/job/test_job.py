import os
import time
import uuid
from contextlib import contextmanager
from multiprocessing import Pool as MultiProcPool
from os.path import dirname
from pathlib import Path

import pytest

import mlflow.server.handlers
from mlflow.entities._job_status import JobStatus
from mlflow.server import (
    ARTIFACT_ROOT_ENV_VAR,
    BACKEND_STORE_URI_ENV_VAR,
    HUEY_STORAGE_PATH_ENV_VAR,
)
from mlflow.server.handlers import _get_job_store
from mlflow.server.jobs import _reinit_huey_queue, _start_job_runner, query_job, submit_job

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="MLflow job execution is not supported on Windows"
)


@contextmanager
def _start_job_runner_for_test(max_job_parallelism, start_new_runner):
    proc = _start_job_runner(
        {"PYTHONPATH": dirname(__file__)},
        max_job_parallelism,
        os.getpid(),
        start_new_runner,
    )
    time.sleep(6)  # wait for huey consumer to spin up.
    try:
        yield proc
    finally:
        proc.kill()


@contextmanager
def _setup_job_runner(max_job_parallelism, monkeypatch, tmp_path, backend_store_uri=None):
    backend_store_uri = backend_store_uri or f"sqlite:///{tmp_path / 'mlflow.db'}"
    huey_store_path = str(tmp_path / "huey.db")
    default_artifact_root = str(tmp_path / "artifacts")
    try:
        monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
        monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, backend_store_uri)
        monkeypatch.setenv(ARTIFACT_ROOT_ENV_VAR, default_artifact_root)
        monkeypatch.setenv(HUEY_STORAGE_PATH_ENV_VAR, huey_store_path)

        with _start_job_runner_for_test(max_job_parallelism, True) as job_runner_proc:
            _reinit_huey_queue()
            yield job_runner_proc
    finally:
        mlflow.server.handlers._job_store = None


def basic_job_fun(x, y, sleep_secs=0):
    if sleep_secs > 0:
        time.sleep(sleep_secs)
    return x + y


def test_basic_job(monkeypatch, tmp_path):
    with _setup_job_runner(1, monkeypatch, tmp_path):
        submitted_job = submit_job(basic_job_fun, {"x": 3, "y": 4})
        wait_job_finalize(submitted_job.job_id, timeout=2)
        job = query_job(submitted_job.job_id)
        assert job.job_id == submitted_job.job_id
        assert job.function_fullname == "test_job.basic_job_fun"
        assert job.params == '{"x": 3, "y": 4}'
        assert job.timeout is None
        assert job.result == "7"
        assert job.parsed_result == 7
        assert job.status == JobStatus.SUCCEEDED
        assert job.retry_count == 0


def json_in_out_fun(data):
    x = data["x"]
    y = data["y"]
    return {"res": x + y}


def test_job_json_input_output(monkeypatch, tmp_path):
    with _setup_job_runner(1, monkeypatch, tmp_path):
        submitted_job = submit_job(json_in_out_fun, {"data": {"x": 3, "y": 4}})
        wait_job_finalize(submitted_job.job_id, timeout=2)
        job = query_job(submitted_job.job_id)
        assert job.job_id == submitted_job.job_id
        assert job.function_fullname == "test_job.json_in_out_fun"
        assert job.params == '{"data": {"x": 3, "y": 4}}'
        assert job.result == '{"res": 7}'
        assert job.parsed_result == {"res": 7}
        assert job.status == JobStatus.SUCCEEDED
        assert job.retry_count == 0


def err_fun(data):
    raise RuntimeError()


def test_error_job(monkeypatch, tmp_path):
    with _setup_job_runner(1, monkeypatch, tmp_path):
        submitted_job = submit_job(err_fun, {"data": None})
        wait_job_finalize(submitted_job.job_id, timeout=2)
        job = query_job(submitted_job.job_id)

        # check database record correctness.
        assert job.job_id == submitted_job.job_id
        assert job.function_fullname == "test_job.err_fun"
        assert job.params == '{"data": null}'
        assert job.result == "RuntimeError()"
        assert job.parsed_result == "RuntimeError()"
        assert job.status == JobStatus.FAILED
        assert job.retry_count == 0


def assert_job_result(job_id, expected_status, expected_result):
    job = query_job(job_id)
    assert job.status == expected_status
    assert job.parsed_result == expected_result


def test_job_resume_on_job_runner_restart(monkeypatch, tmp_path):
    with _setup_job_runner(1, monkeypatch, tmp_path) as job_runner_proc:
        job1_id = submit_job(basic_job_fun, {"x": 3, "y": 4, "sleep_secs": 0}).job_id
        job2_id = submit_job(basic_job_fun, {"x": 5, "y": 6, "sleep_secs": 2}).job_id
        job3_id = submit_job(basic_job_fun, {"x": 7, "y": 8, "sleep_secs": 0}).job_id
        time.sleep(1)
        job_runner_proc.kill()
        job_runner_proc.wait()  # ensure the job runner process is killed.

        # assert that job1 has done, job2 is running, and job3 is pending.
        assert_job_result(job1_id, JobStatus.SUCCEEDED, 7)
        assert_job_result(job2_id, JobStatus.RUNNING, None)
        assert_job_result(job3_id, JobStatus.PENDING, None)

        # restart the job runner, and verify it resumes unfinished jobs (job2 and job3)
        with _start_job_runner_for_test(1, False):
            time.sleep(3)

            # assert all jobs are done.
            assert_job_result(job1_id, JobStatus.SUCCEEDED, 7)
            assert_job_result(job2_id, JobStatus.SUCCEEDED, 11)
            assert_job_result(job3_id, JobStatus.SUCCEEDED, 15)


def test_job_resume_on_new_job_runner(monkeypatch, tmp_path):
    db_tmp_path = tmp_path / "db"
    db_tmp_path.mkdir()
    runner1_tmp_path = tmp_path / "runner1"
    runner1_tmp_path.mkdir()
    runner2_tmp_path = tmp_path / "runner2"
    runner2_tmp_path.mkdir()

    backend_store_uri = f"sqlite:///{db_tmp_path / 'mlflow.db'!s}"

    with _setup_job_runner(1, monkeypatch, runner1_tmp_path, backend_store_uri) as job_runner_proc:
        job1_id = submit_job(basic_job_fun, {"x": 3, "y": 4, "sleep_secs": 0}).job_id
        job2_id = submit_job(basic_job_fun, {"x": 5, "y": 6, "sleep_secs": 10}).job_id
        job3_id = submit_job(basic_job_fun, {"x": 7, "y": 8, "sleep_secs": 0}).job_id
        time.sleep(1)

    # ensure the job runner process is killed.
    job_runner_proc.wait()

    with _setup_job_runner(1, monkeypatch, runner2_tmp_path, backend_store_uri):
        # assert that job1 has done, job2 is running, and job3 is pending.
        assert_job_result(job1_id, JobStatus.SUCCEEDED, 7)
        assert_job_result(job2_id, JobStatus.RUNNING, None)
        assert_job_result(job3_id, JobStatus.PENDING, None)
        time.sleep(10)
        # assert all jobs are done.
        assert_job_result(job1_id, JobStatus.SUCCEEDED, 7)
        assert_job_result(job2_id, JobStatus.SUCCEEDED, 11)
        assert_job_result(job3_id, JobStatus.SUCCEEDED, 15)


def test_job_queue_parallelism(monkeypatch, tmp_path):
    # test job queue parallelism=2 and each job consumes 2 seconds.
    with _setup_job_runner(2, monkeypatch, tmp_path):
        job_ids = [
            submit_job(basic_job_fun, {"x": x, "y": 1, "sleep_secs": 3}).job_id for x in range(4)
        ]
        wait_job_finalize(job_ids[0], timeout=4)
        wait_job_finalize(job_ids[1], timeout=4)

        # assert that job1 and job2 are done, and job3 and job4 are running
        assert_job_result(job_ids[0], JobStatus.SUCCEEDED, 1)
        assert_job_result(job_ids[1], JobStatus.SUCCEEDED, 2)
        assert_job_result(job_ids[2], JobStatus.RUNNING, None)
        assert_job_result(job_ids[3], JobStatus.RUNNING, None)

        wait_job_finalize(job_ids[2], timeout=4)
        wait_job_finalize(job_ids[3], timeout=4)
        assert_job_result(job_ids[2], JobStatus.SUCCEEDED, 3)
        assert_job_result(job_ids[3], JobStatus.SUCCEEDED, 4)


def transient_err_fun(tmp_dir: str, succeed_on_nth_run: int):
    """
    This function will raise `TransientError` on the first (`succeed_on_nth_run` - 1) runs,
    then return 100 on the `succeed_on_nth_run` run. The `tmp_dir` records the run state.
    """
    from mlflow.server.jobs import TransientError

    if len(os.listdir(tmp_dir)) == succeed_on_nth_run:
        return 100
    with open(os.path.join(tmp_dir, uuid.uuid4().hex), "w") as f:
        f.close()
    raise TransientError(RuntimeError("test transient error."))


def wait_job_finalize(job_id, timeout):
    beg_time = time.time()
    while time.time() - beg_time <= timeout:
        job = query_job(job_id)
        if JobStatus.is_finalized(job.status):
            return
        time.sleep(0.1)
    raise TimeoutError("The job is not finalized within the timeout.")


def test_job_retry_on_transient_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY", "1")
    with _setup_job_runner(1, monkeypatch, tmp_path):
        store = _get_job_store()

        job1_tmp_path = tmp_path / "job1"
        job1_tmp_path.mkdir()

        job1_id = submit_job(
            transient_err_fun, {"tmp_dir": str(job1_tmp_path), "succeed_on_nth_run": 4}
        ).job_id
        wait_job_finalize(job1_id, timeout=15)
        assert_job_result(job1_id, JobStatus.FAILED, "RuntimeError('test transient error.')")
        job1 = store.get_job(job1_id)
        assert job1.status == JobStatus.FAILED
        assert job1.result == "RuntimeError('test transient error.')"
        assert job1.retry_count == 3

        job2_tmp_path = tmp_path / "job2"
        job2_tmp_path.mkdir()

        job2_id = submit_job(
            transient_err_fun, {"tmp_dir": str(job2_tmp_path), "succeed_on_nth_run": 1}
        ).job_id
        time.sleep(3)
        assert_job_result(job2_id, JobStatus.SUCCEEDED, 100)
        job2 = store.get_job(job2_id)
        assert job2.status == JobStatus.SUCCEEDED
        assert job2.result == "100"
        assert job2.retry_count == 1


# `submit_job` API is designed to be called inside MLflow server handler,
# MLflow server handler might be executed in multiple MLflow server workers
# so that we need a test to cover the case that executes `submit_job` in
# multi-processes case.
def test_submit_jobs_from_multi_processes(monkeypatch, tmp_path):
    with _setup_job_runner(4, monkeypatch, tmp_path), MultiProcPool() as pool:
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
            assert_job_result(job_ids[x], JobStatus.SUCCEEDED, x + 1)


def sleep_fun(sleep_secs, tmp_dir):
    (Path(tmp_dir) / "pid").write_text(str(os.getpid()))
    time.sleep(sleep_secs)


def test_job_timeout(monkeypatch, tmp_path):
    from mlflow.server.jobs.util import is_process_alive

    with _setup_job_runner(1, monkeypatch, tmp_path) as job_runner_proc:
        job_tmp_path = tmp_path / "job"
        job_tmp_path.mkdir()
        job_id = submit_job(
            sleep_fun, {"sleep_secs": 10, "tmp_dir": str(job_tmp_path)}, timeout=5
        ).job_id
        wait_job_finalize(job_id, timeout=6)
        pid = int((job_tmp_path / "pid").read_text())
        # assert timeout job process is killed.
        assert not is_process_alive(pid)

        assert_job_result(job_id, JobStatus.TIMEOUT, None)

        store = _get_job_store()
        job = store.get_job(job_id)

        # check database record correctness.
        assert job.job_id == job_id
        assert job.function_fullname == "test_job.sleep_fun"
        assert job.timeout == 5
        assert job.result is None
        assert job.status == JobStatus.TIMEOUT
        assert job.retry_count == 0

        submit_job(sleep_fun, {"sleep_secs": 10, "tmp_dir": str(job_tmp_path)}, timeout=15)
        time.sleep(5)
        pid = int((job_tmp_path / "pid").read_text())
        assert is_process_alive(pid)
        job_runner_proc.kill()
        time.sleep(2)
        # assert the job process is killed after job runner process is killed.
        assert not is_process_alive(pid)


def test_list_job_pagination(monkeypatch, tmp_path):
    import mlflow.store.jobs.sqlalchemy_store

    monkeypatch.setattr(mlflow.store.jobs.sqlalchemy_store, "_LIST_JOB_PAGE_SIZE", 3)
    with _setup_job_runner(1, monkeypatch, tmp_path):
        job_ids = []
        for x in range(10):
            job_id = submit_job(basic_job_fun, {"x": x, "y": 4}).job_id
            job_ids.append(job_id)

        listed_jobs = _get_job_store().list_jobs()
        assert [job.job_id for job in listed_jobs] == job_ids
