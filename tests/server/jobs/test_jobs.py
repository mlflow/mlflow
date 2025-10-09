import multiprocessing
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

import mlflow.server.handlers
from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException
from mlflow.server import (
    ARTIFACT_ROOT_ENV_VAR,
    BACKEND_STORE_URI_ENV_VAR,
    HUEY_STORAGE_PATH_ENV_VAR,
)
from mlflow.server.handlers import _get_job_store
from mlflow.server.jobs import get_job, job, submit_job
from mlflow.server.jobs.utils import _launch_job_runner
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

# TODO: Remove `pytest.mark.xfail` after fixing flakiness
pytestmark = [
    pytest.mark.skipif(os.name == "nt", reason="MLflow job execution is not supported on Windows"),
    pytest.mark.xfail,
]


@contextmanager
def _launch_job_runner_for_test():
    root = str(Path(__file__).resolve().parents[3])
    new_pythonpath = f"{root}{os.pathsep}{path}" if (path := os.environ.get("PYTHONPATH")) else root
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend_store_uri: str | None = None
):
    backend_store_uri = backend_store_uri or f"sqlite:///{tmp_path / 'mlflow.db'}"
    # Pre-initialize the database to prevent race conditions when the tracking store and job store
    # attempt to initialize the database simultaneously.
    store = SqlAlchemyStore(backend_store_uri, (tmp_path / "artifacts").as_uri())
    store.engine.dispose()
    huey_store_path = tmp_path / "huey_store"
    huey_store_path.mkdir()
    default_artifact_root = str(tmp_path / "artifacts")
    try:
        monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
        monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, backend_store_uri)
        monkeypatch.setenv(ARTIFACT_ROOT_ENV_VAR, default_artifact_root)
        monkeypatch.setenv(HUEY_STORAGE_PATH_ENV_VAR, str(huey_store_path))

        with _launch_job_runner_for_test() as job_runner_proc:
            yield job_runner_proc
    finally:
        mlflow.server.handlers._job_store = None


@job(max_workers=1, use_process=False)
def basic_job_fun(x, y, sleep_secs=0):
    if sleep_secs > 0:
        time.sleep(sleep_secs)
    return x + y


def test_basic_job(monkeypatch, tmp_path):
    with _setup_job_runner(monkeypatch, tmp_path):
        submitted_job = submit_job(basic_job_fun, {"x": 3, "y": 4})
        wait_job_finalize(submitted_job.job_id)
        job = get_job(submitted_job.job_id)
        assert job.job_id == submitted_job.job_id
        assert job.function_fullname == "tests.server.jobs.test_jobs.basic_job_fun"
        assert job.params == '{"x": 3, "y": 4}'
        assert job.timeout is None
        assert job.result == "7"
        assert job.parsed_result == 7
        assert job.status == JobStatus.SUCCEEDED
        assert job.retry_count == 0


@job(max_workers=1, use_process=False)
def json_in_out_fun(data):
    x = data["x"]
    y = data["y"]
    return {"res": x + y}


def test_job_json_input_output(monkeypatch, tmp_path):
    with _setup_job_runner(monkeypatch, tmp_path):
        submitted_job = submit_job(json_in_out_fun, {"data": {"x": 3, "y": 4}})
        wait_job_finalize(submitted_job.job_id)
        job = get_job(submitted_job.job_id)
        assert job.job_id == submitted_job.job_id
        assert job.function_fullname == "tests.server.jobs.test_jobs.json_in_out_fun"
        assert job.params == '{"data": {"x": 3, "y": 4}}'
        assert job.result == '{"res": 7}'
        assert job.parsed_result == {"res": 7}
        assert job.status == JobStatus.SUCCEEDED
        assert job.retry_count == 0


@job(max_workers=1, use_process=False)
def err_fun(data):
    raise RuntimeError()


def test_error_job(monkeypatch, tmp_path):
    with _setup_job_runner(monkeypatch, tmp_path):
        submitted_job = submit_job(err_fun, {"data": None})
        wait_job_finalize(submitted_job.job_id)
        job = get_job(submitted_job.job_id)

        # check database record correctness.
        assert job.job_id == submitted_job.job_id
        assert job.function_fullname == "tests.server.jobs.test_jobs.err_fun"
        assert job.params == '{"data": null}'
        assert job.result.startswith("RuntimeError()")
        assert job.status == JobStatus.FAILED
        assert job.retry_count == 0


def assert_job_result(job_id, expected_status, expected_result):
    job = get_job(job_id)
    assert job.status == expected_status
    assert job.parsed_result == expected_result


def test_job_resume_on_job_runner_restart(monkeypatch, tmp_path):
    with _setup_job_runner(monkeypatch, tmp_path) as job_runner_proc:
        job1_id = submit_job(basic_job_fun, {"x": 3, "y": 4, "sleep_secs": 0}).job_id
        job2_id = submit_job(basic_job_fun, {"x": 5, "y": 6, "sleep_secs": 2}).job_id
        job3_id = submit_job(basic_job_fun, {"x": 7, "y": 8, "sleep_secs": 0}).job_id
        wait_job_finalize(job1_id)
        job_runner_proc.kill()
        job_runner_proc.wait()  # ensure the job runner process is killed.

        # assert that job1 has done, job2 is running, and job3 is pending.
        assert_job_result(job1_id, JobStatus.SUCCEEDED, 7)
        assert_job_result(job2_id, JobStatus.RUNNING, None)
        assert_job_result(job3_id, JobStatus.PENDING, None)

        # restart the job runner, and verify it resumes unfinished jobs (job2 and job3)
        with _launch_job_runner_for_test():
            wait_job_finalize(job2_id)
            wait_job_finalize(job3_id)
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

    with _setup_job_runner(monkeypatch, runner1_tmp_path, backend_store_uri) as job_runner_proc:
        job1_id = submit_job(basic_job_fun, {"x": 3, "y": 4, "sleep_secs": 0}).job_id
        job2_id = submit_job(basic_job_fun, {"x": 5, "y": 6, "sleep_secs": 2}).job_id
        job3_id = submit_job(basic_job_fun, {"x": 7, "y": 8, "sleep_secs": 0}).job_id
        wait_job_finalize(job1_id)

        # assert that job1 has done, job2 is running, and job3 is pending.
        assert_job_result(job1_id, JobStatus.SUCCEEDED, 7)
        assert_job_result(job2_id, JobStatus.RUNNING, None)
        assert_job_result(job3_id, JobStatus.PENDING, None)

    # ensure the job runner process is killed.
    job_runner_proc.wait()

    with _setup_job_runner(monkeypatch, runner2_tmp_path, backend_store_uri):
        wait_job_finalize(job2_id)
        wait_job_finalize(job3_id)
        # assert all jobs are done.
        assert_job_result(job1_id, JobStatus.SUCCEEDED, 7)
        assert_job_result(job2_id, JobStatus.SUCCEEDED, 11)
        assert_job_result(job3_id, JobStatus.SUCCEEDED, 15)


@job(max_workers=2, use_process=False)
def job_fun_parallelism2(x, y, sleep_secs=0):
    if sleep_secs > 0:
        time.sleep(sleep_secs)
    return x + y


@job(max_workers=3, use_process=False)
def job_fun_parallelism3(x, y, sleep_secs=0):
    if sleep_secs > 0:
        time.sleep(sleep_secs)
    return x + y


def test_job_queue_parallelism(monkeypatch, tmp_path):
    with _setup_job_runner(monkeypatch, tmp_path):
        # warm up:
        # The first job submission of each job function triggers setting up
        # the corresponding huey consumer process.
        job1_id = submit_job(job_fun_parallelism2, {"x": 1, "y": 1, "sleep_secs": 0}).job_id
        job2_id = submit_job(job_fun_parallelism3, {"x": 1, "y": 1, "sleep_secs": 0}).job_id
        wait_job_finalize(job1_id)
        wait_job_finalize(job2_id)

        job_p2_ids = [
            submit_job(job_fun_parallelism2, {"x": x, "y": 1, "sleep_secs": 3}).job_id
            for x in range(4)
        ]
        job_p3_ids = [
            submit_job(job_fun_parallelism3, {"x": x, "y": 1, "sleep_secs": 3}).job_id
            for x in range(4)
        ]

        time.sleep(3.5)
        assert_job_result(job_p2_ids[0], JobStatus.SUCCEEDED, 1)
        assert_job_result(job_p2_ids[1], JobStatus.SUCCEEDED, 2)
        assert_job_result(job_p2_ids[2], JobStatus.RUNNING, None)
        assert_job_result(job_p2_ids[3], JobStatus.RUNNING, None)

        assert_job_result(job_p3_ids[0], JobStatus.SUCCEEDED, 1)
        assert_job_result(job_p3_ids[1], JobStatus.SUCCEEDED, 2)
        assert_job_result(job_p3_ids[2], JobStatus.SUCCEEDED, 3)
        assert_job_result(job_p3_ids[3], JobStatus.RUNNING, None)

        time.sleep(3.5)
        assert_job_result(job_p2_ids[2], JobStatus.SUCCEEDED, 3)
        assert_job_result(job_p2_ids[3], JobStatus.SUCCEEDED, 4)
        assert_job_result(job_p3_ids[3], JobStatus.SUCCEEDED, 4)


@job(max_workers=1, use_process=False)
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


@job(max_workers=1, use_process=False, transient_error_classes=[TimeoutError])
def transient_err_fun2(tmp_dir: str, succeed_on_nth_run: int):
    """
    This function will raise `TimeoutError` on the first (`succeed_on_nth_run` - 1) runs,
    then return 100 on the `succeed_on_nth_run` run. The `tmp_dir` records the run state.
    """
    if len(os.listdir(tmp_dir)) == succeed_on_nth_run:
        return 100
    with open(os.path.join(tmp_dir, uuid.uuid4().hex), "w") as f:
        f.close()
    raise TimeoutError("test transient timeout error.")


def wait_job_finalize(job_id, timeout=60):
    beg_time = time.time()
    while time.time() - beg_time <= timeout:
        job = get_job(job_id)
        if JobStatus.is_finalized(job.status):
            return
        time.sleep(0.5)
    raise TimeoutError("The job is not finalized within the timeout.")


def test_job_retry_on_transient_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY", "1")
    monkeypatch.setenv("MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES", "2")
    with _setup_job_runner(monkeypatch, tmp_path):
        store = _get_job_store()

        job1_tmp_path = tmp_path / "job1"
        job1_tmp_path.mkdir()

        job1_id = submit_job(
            transient_err_fun, {"tmp_dir": str(job1_tmp_path), "succeed_on_nth_run": 3}
        ).job_id
        wait_job_finalize(job1_id)
        assert_job_result(job1_id, JobStatus.FAILED, "RuntimeError('test transient error.')")
        job1 = store.get_job(job1_id)
        assert job1.status == JobStatus.FAILED
        assert job1.result == "RuntimeError('test transient error.')"
        assert job1.retry_count == 2

        job2_tmp_path = tmp_path / "job2"
        job2_tmp_path.mkdir()

        job2_id = submit_job(
            transient_err_fun, {"tmp_dir": str(job2_tmp_path), "succeed_on_nth_run": 2}
        ).job_id
        wait_job_finalize(job2_id)
        assert_job_result(job2_id, JobStatus.SUCCEEDED, 100)
        job2 = store.get_job(job2_id)
        assert job2.status == JobStatus.SUCCEEDED
        assert job2.result == "100"
        assert job2.retry_count == 2

        job3_tmp_path = tmp_path / "job3"
        job3_tmp_path.mkdir()

        job3_id = submit_job(
            transient_err_fun2, {"tmp_dir": str(job3_tmp_path), "succeed_on_nth_run": 2}
        ).job_id
        wait_job_finalize(job3_id)
        assert_job_result(job3_id, JobStatus.SUCCEEDED, 100)
        job3 = store.get_job(job3_id)
        assert job3.status == JobStatus.SUCCEEDED
        assert job3.result == "100"
        assert job3.retry_count == 2


# `submit_job` API is designed to be called inside MLflow server handler,
# MLflow server handler might be executed in multiple MLflow server workers
# so that we need a test to cover the case that executes `submit_job` in
# multi-processes case.
def test_submit_jobs_from_multi_processes(monkeypatch, tmp_path):
    context = multiprocessing.get_context("spawn")
    with _setup_job_runner(monkeypatch, tmp_path), context.Pool(2) as pool:
        job_id = submit_job(basic_job_fun, {"x": 1, "y": 1, "sleep_secs": 0}).job_id
        wait_job_finalize(job_id)

        async_res_list = [
            pool.apply_async(
                submit_job,
                args=(basic_job_fun,),
                kwds={"params": {"x": x, "y": 1, "sleep_secs": 2}},
            )
            for x in range(2)
        ]
        job_ids = [async_res.get().job_id for async_res in async_res_list]
        for job_id in job_ids:
            wait_job_finalize(job_id)
        for x in range(2):
            assert_job_result(job_ids[x], JobStatus.SUCCEEDED, x + 1)


@job(max_workers=1)
def sleep_fun(sleep_secs, tmp_dir):
    (Path(tmp_dir) / "pid").write_text(str(os.getpid()))
    time.sleep(sleep_secs)


def test_job_timeout(monkeypatch, tmp_path):
    from mlflow.server.jobs.utils import is_process_alive

    with _setup_job_runner(monkeypatch, tmp_path):
        job_tmp_path = tmp_path / "job"
        job_tmp_path.mkdir()

        # warm up
        job_id = submit_job(sleep_fun, {"sleep_secs": 0, "tmp_dir": str(job_tmp_path)}).job_id
        wait_job_finalize(job_id)

        job_id = submit_job(
            sleep_fun, {"sleep_secs": 10, "tmp_dir": str(job_tmp_path)}, timeout=3
        ).job_id
        wait_job_finalize(job_id)
        pid = int((job_tmp_path / "pid").read_text())
        # assert timeout job process is killed.
        assert not is_process_alive(pid)

        assert_job_result(job_id, JobStatus.TIMEOUT, None)

        store = _get_job_store()
        job = store.get_job(job_id)

        # check database record correctness.
        assert job.job_id == job_id
        assert job.function_fullname == "tests.server.jobs.test_jobs.sleep_fun"
        assert job.timeout == 3.0
        assert job.result is None
        assert job.status == JobStatus.TIMEOUT
        assert job.retry_count == 0


def test_list_job_pagination(monkeypatch, tmp_path):
    import mlflow.store.jobs.sqlalchemy_store

    monkeypatch.setattr(mlflow.store.jobs.sqlalchemy_store, "_LIST_JOB_PAGE_SIZE", 3)
    with _setup_job_runner(monkeypatch, tmp_path):
        job_ids = []
        for x in range(10):
            job_id = submit_job(basic_job_fun, {"x": x, "y": 4}).job_id
            job_ids.append(job_id)

        listed_jobs = _get_job_store().list_jobs()
        assert [job.job_id for job in listed_jobs] == job_ids


def bad_job_function() -> None:
    return


def test_job_function_without_decorator(monkeypatch, tmp_path):
    with _setup_job_runner(monkeypatch, tmp_path):
        with pytest.raises(
            MlflowException,
            match="The job function tests.server.jobs.test_jobs.bad_job_function is not decorated",
        ):
            submit_job(bad_job_function, params={})


@job(max_workers=1, use_process=True)
def job_use_process(tmp_dir):
    (Path(tmp_dir) / str(os.getpid())).write_text("")


def test_job_use_process(monkeypatch, tmp_path):
    with _setup_job_runner(monkeypatch, tmp_path):
        job_tmp_path = tmp_path / "job"
        job_tmp_path.mkdir()

        job_id1 = submit_job(job_use_process, {"tmp_dir": str(job_tmp_path)}).job_id
        job_id2 = submit_job(job_use_process, {"tmp_dir": str(job_tmp_path)}).job_id
        wait_job_finalize(job_id1)
        wait_job_finalize(job_id2)
        assert len(os.listdir(str(job_tmp_path))) == 2


def test_submit_job_bad_call(monkeypatch, tmp_path):
    with _setup_job_runner(monkeypatch, tmp_path):
        with pytest.raises(
            MlflowException,
            match="When calling 'submit_job', the 'params' argument must be a dict.",
        ):
            submit_job(basic_job_fun, params=None)
