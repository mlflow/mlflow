import json
import os
import threading
import time
from pathlib import Path
from unittest import mock

import pytest

from mlflow.entities._job import Job
from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.server.jobs import _ALLOWED_JOB_NAME_LIST, _SUPPORTED_JOB_FUNCTION_LIST, job, submit_job
from mlflow.server.jobs import _executor_runner as runner
from mlflow.server.jobs.executor import JobExecutorConfig, JobResult
from mlflow.server.jobs.executor_registry import shutdown_executor_registry
from mlflow.server.jobs.local_executor import LocalJobExecutor
from mlflow.server.jobs.utils import (
    _build_job_name_to_fn_fullname_map,
    _exec_job,
    _job_name_to_fn_fullname_map,
)
from mlflow.store.jobs.abstract_store import JobUpdateStatus
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.jobs.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyJobStore
from mlflow.utils.workspace_context import WorkspaceContext

pytestmark = [
    pytest.mark.skipif(os.name == "nt", reason="MLflow job execution is not supported on Windows"),
]


class _EngineTransientError(RuntimeError):
    pass


@job(name="executor_engine_add", max_workers=1)
def executor_engine_add(x, y):
    return x + y


@job(name="executor_engine_boom", max_workers=1)
def executor_engine_boom():
    raise RuntimeError("boom")


@job(name="executor_engine_sleep", max_workers=1)
def executor_engine_sleep(sleep_secs):
    time.sleep(sleep_secs)


@job(
    name="executor_engine_flaky",
    max_workers=1,
    transient_error_classes=[_EngineTransientError],
)
def executor_engine_flaky(marker_path):
    # Fail transiently on the first attempt, then succeed. The attempt count is tracked in
    # a file so it survives across the per-attempt subprocesses the executor spawns.
    marker = Path(marker_path)
    attempts = (int(marker.read_text()) if marker.exists() else 0) + 1
    marker.write_text(str(attempts))
    if attempts < 2:
        raise _EngineTransientError("flaky")
    return attempts


_JOB_FULLNAMES = [
    "tests.server.jobs.test_executor_engine.executor_engine_add",
    "tests.server.jobs.test_executor_engine.executor_engine_boom",
    "tests.server.jobs.test_executor_engine.executor_engine_sleep",
    "tests.server.jobs.test_executor_engine.executor_engine_flaky",
]
_JOB_NAMES = [
    "executor_engine_add",
    "executor_engine_boom",
    "executor_engine_sleep",
    "executor_engine_flaky",
]


@pytest.fixture
def registered_jobs():
    _SUPPORTED_JOB_FUNCTION_LIST.extend(_JOB_FULLNAMES)
    _ALLOWED_JOB_NAME_LIST.extend(_JOB_NAMES)
    _build_job_name_to_fn_fullname_map()
    try:
        yield
    finally:
        for name in _JOB_NAMES:
            _job_name_to_fn_fullname_map.pop(name, None)
            if name in _ALLOWED_JOB_NAME_LIST:
                _ALLOWED_JOB_NAME_LIST.remove(name)
        for fullname in _JOB_FULLNAMES:
            if fullname in _SUPPORTED_JOB_FUNCTION_LIST:
                _SUPPORTED_JOB_FUNCTION_LIST.remove(fullname)


@pytest.fixture
def job_store(tmp_path):
    return SqlAlchemyJobStore(f"sqlite:///{tmp_path / 'jobs.db'}")


@pytest.fixture(autouse=True)
def _reset_executor_registry():
    # _select_executor() builds the process-global executor registry singleton; tear it
    # down after each test so it does not leak into later tests.
    yield
    shutdown_executor_registry()


@pytest.fixture
def executor():
    ex = LocalJobExecutor(JobExecutorConfig(default_timeout=60.0))
    ex.start_executor()
    try:
        yield ex
    finally:
        ex.stop_executor()


def test_select_executor_defaults_to_local():
    assert isinstance(runner._select_executor(), LocalJobExecutor)


def test_loop_runs_pending_job_to_success(registered_jobs, job_store, executor):
    created = job_store.create_job("executor_engine_add", json.dumps({"x": 3, "y": 4}))

    executed = runner._poll_and_execute_once(job_store, executor, lease_duration=60.0)

    assert executed == 1
    updated = job_store.get_job(created.job_id)
    assert updated.status == JobStatus.SUCCEEDED
    assert updated.result == "7"
    assert updated.retry_count == 0


def test_loop_records_failure(registered_jobs, job_store, executor):
    created = job_store.create_job("executor_engine_boom", "{}")

    runner._poll_and_execute_once(job_store, executor, lease_duration=60.0)

    updated = job_store.get_job(created.job_id)
    assert updated.status == JobStatus.FAILED


def test_loop_no_pending_jobs_is_noop(registered_jobs, job_store, executor):
    assert runner._poll_and_execute_once(job_store, executor, lease_duration=60.0) == 0


def test_loop_marks_timed_out_job(registered_jobs, job_store, executor):
    created = job_store.create_job(
        "executor_engine_sleep", json.dumps({"sleep_secs": 30}), timeout=1.0
    )

    runner._poll_and_execute_once(job_store, executor, lease_duration=60.0)

    updated = job_store.get_job(created.job_id)
    assert updated.status == JobStatus.TIMEOUT


def test_loop_retries_transient_error_then_succeeds(
    registered_jobs, job_store, executor, tmp_path, monkeypatch
):
    marker = tmp_path / "attempts.txt"
    created = job_store.create_job(
        "executor_engine_flaky", json.dumps({"marker_path": str(marker)})
    )

    # Avoid the real exponential backoff sleep between retries.
    monkeypatch.setattr(runner, "_backoff_after_transient_retry", lambda retry_count: None)

    # First poll: transient error -> retry_or_fail_job resets the job to PENDING.
    runner._poll_and_execute_once(job_store, executor, lease_duration=60.0)
    after_first = job_store.get_job(created.job_id)
    assert after_first.status == JobStatus.PENDING
    assert after_first.retry_count == 1

    # Second poll: the retried job is re-claimed and succeeds.
    runner._poll_and_execute_once(job_store, executor, lease_duration=60.0)
    after_second = job_store.get_job(created.job_id)
    assert after_second.status == JobStatus.SUCCEEDED
    assert after_second.result == "2"


def test_loop_runs_job_in_non_default_workspace(registered_jobs, tmp_path, executor, monkeypatch):
    from mlflow.entities import Workspace

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    store = WorkspaceAwareSqlAlchemyJobStore(f"sqlite:///{tmp_path / 'ws.db'}")

    with WorkspaceContext("workspace-b"):
        created = store.create_job("executor_engine_add", json.dumps({"x": 5, "y": 6}))

    # The loop enumerates workspaces via _workspace_contexts_for_recovery(), which reads the
    # workspace store. Without the per-workspace context the loop would not see this job.
    mock_workspace_store = mock.MagicMock()
    mock_workspace_store.list_workspaces.return_value = [Workspace(name="workspace-b")]
    with mock.patch(
        "mlflow.server.workspace_helpers._get_workspace_store",
        return_value=mock_workspace_store,
    ):
        executed = runner._poll_and_execute_once(store, executor, lease_duration=60.0)

    assert executed == 1
    with WorkspaceContext("workspace-b"):
        updated = store.get_job(created.job_id)
    assert updated.status == JobStatus.SUCCEEDED
    assert updated.result == "11"


@pytest.mark.parametrize(
    ("job_name", "params", "expected_status"),
    [
        ("executor_engine_add", {"x": 3, "y": 4}, JobStatus.SUCCEEDED),
        ("executor_engine_boom", {}, JobStatus.FAILED),
    ],
)
def test_engines_produce_equivalent_terminal_state(
    registered_jobs,
    tmp_path,
    executor,
    monkeypatch,
    job_name,
    params,
    expected_status,
):
    # Executor engine: drive the loop directly.
    exec_store = SqlAlchemyJobStore(f"sqlite:///{tmp_path / 'exec.db'}")
    exec_job = exec_store.create_job(job_name, json.dumps(params))
    runner._poll_and_execute_once(exec_store, executor, lease_duration=60.0)
    exec_result = exec_store.get_job(exec_job.job_id)

    # Huey engine: drive _exec_job directly, pointing _get_job_store at the huey store.
    huey_store = SqlAlchemyJobStore(f"sqlite:///{tmp_path / 'huey.db'}")
    monkeypatch.setattr("mlflow.server.handlers._get_job_store", lambda *args, **kwargs: huey_store)
    huey_job = huey_store.create_job(job_name, json.dumps(params))
    _exec_job(huey_job.job_id, None, job_name, params, None)
    huey_result = huey_store.get_job(huey_job.job_id)

    # Both engines must reach the same terminal state from the same input.
    assert exec_result.status == huey_result.status == expected_status
    assert exec_result.result == huey_result.result
    assert exec_result.retry_count == huey_result.retry_count == 0
    if expected_status == JobStatus.SUCCEEDED:
        assert exec_result.result == "7"


def _make_job(status=JobStatus.RUNNING, retry_count=0) -> Job:
    return Job(
        job_id="job-1",
        creation_time=0,
        job_name="executor_engine_add",
        params="{}",
        timeout=None,
        status=status,
        result=None,
        retry_count=retry_count,
        last_update_time=0,
    )


@pytest.mark.parametrize(
    ("result", "expected_call"),
    [
        (
            JobResult(status=JobStatus.SUCCEEDED, result="7"),
            mock.call("job-1", JobStatus.SUCCEEDED, result="7"),
        ),
        (
            JobResult(status=JobStatus.TIMEOUT),
            mock.call("job-1", JobStatus.TIMEOUT, error_message=None),
        ),
        (
            JobResult(status=JobStatus.FAILED, error_message="boom"),
            mock.call("job-1", JobStatus.FAILED, error_message="boom"),
        ),
    ],
)
def test_record_result_terminal_mapping(result, expected_call):
    store = mock.MagicMock()
    store.get_job.return_value = _make_job(status=JobStatus.RUNNING)
    runner._record_result(store, "job-1", "executor_engine_add", result)
    assert store.report_job_result.call_args == expected_call


def test_record_result_transient_error_retries():
    store = mock.MagicMock(**{"retry_or_fail_job.return_value": 1})
    store.get_job.return_value = _make_job(status=JobStatus.RUNNING)
    with mock.patch.object(runner, "_backoff_after_transient_retry") as backoff:
        runner._record_result(
            store,
            "job-1",
            "executor_engine_add",
            JobResult(status=JobStatus.FAILED, error_message="temp", is_transient_error=True),
        )
    store.retry_or_fail_job.assert_called_once_with("job-1", "temp")
    store.fail_job.assert_not_called()
    backoff.assert_called_once_with(1)


def test_record_result_transient_error_exhausted_does_not_backoff():
    store = mock.MagicMock(**{"retry_or_fail_job.return_value": None})
    store.get_job.return_value = _make_job(status=JobStatus.RUNNING)
    with mock.patch.object(runner, "_backoff_after_transient_retry") as backoff:
        runner._record_result(
            store,
            "job-1",
            "executor_engine_add",
            JobResult(status=JobStatus.FAILED, error_message="temp", is_transient_error=True),
        )
    store.retry_or_fail_job.assert_called_once()
    backoff.assert_not_called()


def _submit_job_env(monkeypatch, engine=None):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
    if engine is not None:
        monkeypatch.setenv("MLFLOW_SERVER_JOB_EXECUTION_ENGINE", engine)


def test_submit_job_defaults_to_huey_enqueue(monkeypatch, registered_jobs):
    _submit_job_env(monkeypatch)  # engine unset -> defaults to "huey"
    store = mock.MagicMock()
    store.create_job.return_value = _make_job(status=JobStatus.PENDING)
    huey_instance = mock.MagicMock()
    with (
        mock.patch("mlflow.server.jobs._get_job_store", return_value=store),
        mock.patch("mlflow.server.jobs.utils._check_requirements"),
        mock.patch(
            "mlflow.server.jobs.utils._get_or_init_huey_instance", return_value=huey_instance
        ),
    ):
        submit_job(executor_engine_add, {"x": 1, "y": 2})

    # workspaces disabled -> workspace arg is None; exclusive=False and extra_envs=None.
    huey_instance.submit_task.assert_called_once_with(
        "job-1", None, "executor_engine_add", {"x": 1, "y": 2}, None, False, None
    )


@pytest.mark.parametrize("engine", ["executor", "Executor", "EXECUTOR"])
def test_submit_job_executor_engine_skips_huey(monkeypatch, registered_jobs, engine):
    _submit_job_env(monkeypatch, engine=engine)  # value is case-normalized
    store = mock.MagicMock()
    store.create_job.return_value = _make_job(status=JobStatus.PENDING)
    with (
        mock.patch("mlflow.server.jobs._get_job_store", return_value=store),
        mock.patch("mlflow.server.jobs.utils._check_requirements"),
        mock.patch("mlflow.server.jobs.utils._get_or_init_huey_instance") as get_huey,
    ):
        submit_job(executor_engine_add, {"x": 1, "y": 2})

    store.create_job.assert_called_once_with(
        "executor_engine_add", json.dumps({"x": 1, "y": 2}), None
    )
    get_huey.assert_not_called()


def test_submit_job_executor_engine_rejects_extra_envs(monkeypatch, registered_jobs):
    _submit_job_env(monkeypatch, engine="executor")
    store = mock.MagicMock()
    store.create_job.return_value = _make_job(status=JobStatus.PENDING)
    with (
        mock.patch("mlflow.server.jobs._get_job_store", return_value=store),
        mock.patch("mlflow.server.jobs.utils._check_requirements"),
        pytest.raises(MlflowException, match="extra_envs is not yet supported"),
    ):
        submit_job(executor_engine_add, {"x": 1, "y": 2}, extra_envs={"TOKEN": "secret"})

    # Rejected before persisting, so no runnable PENDING row is left behind.
    store.create_job.assert_not_called()


@pytest.mark.parametrize("engine", ["spark", "kubernetes", ""])
def test_submit_job_rejects_invalid_engine(monkeypatch, registered_jobs, engine):
    _submit_job_env(monkeypatch, engine=engine)
    store = mock.MagicMock()
    store.create_job.return_value = _make_job(status=JobStatus.PENDING)
    with (
        mock.patch("mlflow.server.jobs._get_job_store", return_value=store),
        mock.patch("mlflow.server.jobs.utils._check_requirements"),
        pytest.raises(
            MlflowException, match="Invalid value for MLFLOW_SERVER_JOB_EXECUTION_ENGINE"
        ),
    ):
        submit_job(executor_engine_add, {"x": 1, "y": 2})

    # Invalid engine is rejected before persisting, so no PENDING row is left behind.
    store.create_job.assert_not_called()


def test_record_result_skips_job_canceled_after_claim(registered_jobs, job_store):
    # A job canceled after it was claimed runs to completion; recording a terminal result
    # then must not raise (the cancel already finalized the row) and must leave it CANCELED.
    created = job_store.create_job("executor_engine_add", json.dumps({"x": 1, "y": 2}))
    job_store.claim_job(created.job_id, lease_duration=60.0)
    job_store.cancel_job(created.job_id)
    assert job_store.get_job(created.job_id).status == JobStatus.CANCELED

    runner._record_result(
        job_store,
        created.job_id,
        "executor_engine_add",
        JobResult(status=JobStatus.SUCCEEDED, result="3"),
    )

    assert job_store.get_job(created.job_id).status == JobStatus.CANCELED


@pytest.mark.parametrize(
    ("engine", "expected"),
    [
        (None, "_launch_job_runner"),
        ("huey", "_launch_job_runner"),
        ("executor", "_launch_executor_runner"),
    ],
)
def test_launch_job_execution_runner_dispatch(monkeypatch, engine, expected):
    from mlflow.server.jobs.utils import _launch_job_execution_runner

    if engine is not None:
        monkeypatch.setenv("MLFLOW_SERVER_JOB_EXECUTION_ENGINE", engine)
    env_map = {"FOO": "bar"}
    with (
        mock.patch("mlflow.server.jobs.utils._launch_executor_runner") as launch_executor,
        mock.patch("mlflow.server.jobs.utils._launch_job_runner") as launch_huey,
    ):
        _launch_job_execution_runner(env_map, 1234)

    launchers = {"_launch_executor_runner": launch_executor, "_launch_job_runner": launch_huey}
    launchers.pop(expected).assert_called_once_with(env_map, 1234)
    for unused in launchers.values():
        unused.assert_not_called()


# ---------------------------------------------------------------------------
# Job-lease renewal
# ---------------------------------------------------------------------------


def test_lease_renewer_renews_while_running_then_stops():
    store = mock.MagicMock()
    calls = []
    reached_two = threading.Event()

    def _renew(job_id, lease_duration):
        calls.append(job_id)
        if len(calls) >= 2:
            reached_two.set()
        return JobUpdateStatus.APPLIED

    store.renew_job_lease.side_effect = _renew
    with runner._LeaseRenewer(store, "job-1", lease_duration=0.6):
        # Wait for two renewals to fire, then exit; no wall-clock sleep drives the count.
        assert reached_two.wait(timeout=5.0)
    # __exit__ joins the renewer thread, so the count is final and no renewals fire after.
    assert store.renew_job_lease.call_count >= 2
    store.renew_job_lease.assert_called_with("job-1", 0.6)


def test_lease_renewer_stops_when_lease_no_longer_renewable():
    store = mock.MagicMock()
    renewed = threading.Event()

    def _renew(job_id, lease_duration):
        renewed.set()
        return JobUpdateStatus.WRONG_STATE

    store.renew_job_lease.side_effect = _renew
    with runner._LeaseRenewer(store, "job-1", lease_duration=0.6):
        assert renewed.wait(timeout=5.0)
    # A non-APPLIED status makes the renewer stop on its own after a single attempt.
    assert store.renew_job_lease.call_count == 1


def test_lease_renewer_survives_renew_error_and_keeps_renewing():
    store = mock.MagicMock()
    calls = []
    reached_second = threading.Event()

    def _renew(job_id, lease_duration):
        calls.append(job_id)
        if len(calls) == 1:
            raise RuntimeError("transient store error")
        reached_second.set()
        return JobUpdateStatus.APPLIED

    store.renew_job_lease.side_effect = _renew
    with mock.patch.object(runner, "_logger") as mock_logger:
        with runner._LeaseRenewer(store, "job-1", lease_duration=0.6):
            # The first renewal raises; the renewer must log and keep going to a second.
            assert reached_second.wait(timeout=5.0)
    mock_logger.exception.assert_called()
    assert store.renew_job_lease.call_count >= 2


def test_long_running_job_lease_is_renewed(registered_jobs, job_store, executor, monkeypatch):
    created = job_store.create_job("executor_engine_sleep", json.dumps({"sleep_secs": 1.0}))
    renew_calls = []
    real_renew = job_store.renew_job_lease

    def _spy(job_id, lease_duration):
        renew_calls.append(job_id)
        return real_renew(job_id, lease_duration)

    monkeypatch.setattr(job_store, "renew_job_lease", _spy)
    # Short lease so renewal (~lease/3) fires several times during the ~1s job.
    runner._poll_and_execute_once(job_store, executor, lease_duration=0.6)

    assert created.job_id in renew_calls
    assert job_store.get_job(created.job_id).status == JobStatus.SUCCEEDED
