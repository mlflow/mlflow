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
from mlflow.protos.databricks_pb2 import TEMPORARILY_UNAVAILABLE
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
from mlflow.utils.workspace_context import WorkspaceContext, get_request_workspace

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


@job(name="executor_engine_parallel", max_workers=2)
def executor_engine_parallel(x):
    return x


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
    "tests.server.jobs.test_executor_engine.executor_engine_parallel",
    "tests.server.jobs.test_executor_engine.executor_engine_flaky",
]
_JOB_NAMES = [
    "executor_engine_add",
    "executor_engine_boom",
    "executor_engine_sleep",
    "executor_engine_parallel",
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
def _backend_store_uri(tmp_path, monkeypatch):
    # _build_execution_context hands jobs the backend store URI, which the real server always sets.
    # The test jobs don't touch the store from their subprocess, so any valid URI suffices here.
    from mlflow.server.constants import BACKEND_STORE_URI_ENV_VAR

    monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, f"sqlite:///{tmp_path / 'backend.db'}")


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


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met in time")


def _wait_submitted(scheduler, job_id, timeout=10.0):
    # The worker marks the job submitted just after the executor's submit_job returns, on its own
    # thread. The scheduler forwards cancellation only after that, so tests wait for it here rather
    # than racing the worker.
    _wait_until(
        lambda: (h := scheduler._in_flight.get(job_id)) is not None and h.submitted, timeout
    )


def _run_to_completion(job_store, executor, lease_duration=60.0):
    """Schedule all currently-PENDING jobs and wait for their worker threads to finish.

    The scheduler runs each claimed job in a worker thread, so tests join before asserting the
    terminal state. Returns the number of jobs scheduled by the single tick.
    """
    scheduler = runner._JobScheduler(job_store, executor, lease_duration)
    scheduled = scheduler.tick()
    scheduler.join(timeout=60.0)
    return scheduled


class _BlockingExecutor:
    """Executor stub whose ``wait_for_job`` blocks until released.

    Holding a job in ``wait_for_job`` lets a test observe the scheduler's state while a job is
    in flight (e.g. concurrency limits, cancellation forwarding), which a real executor that runs
    to completion immediately would not allow.
    """

    def __init__(self):
        self._release = threading.Event()
        self._lock = threading.Lock()
        self.submitted = []
        self.canceled = []
        # run_executor_loop reads executor.config.job_lease_ttl.
        self.config = JobExecutorConfig(default_timeout=60.0)
        # job_id -> workspace resolved on the worker thread at submit time. Records what
        # get_request_workspace() returns inside each worker, to prove per-thread isolation.
        self.observed_workspace = {}

    def start_executor(self):
        pass

    def stop_executor(self):
        pass

    def submit_job(self, *, job_id, **kwargs):
        with self._lock:
            self.submitted.append(job_id)
            self.observed_workspace[job_id] = get_request_workspace()

    def wait_for_job(self, job_id):
        self._release.wait(30.0)
        return JobResult(status=JobStatus.SUCCEEDED, result="ok")

    def cancel_job(self, job_id):
        with self._lock:
            self.canceled.append(job_id)
        self._release.set()

    def wait_until_submitted(self, count=1, timeout=10.0):
        # Workers submit from their own threads, so tests wait for the jobs to actually reach the
        # executor before inspecting or cancelling them, rather than racing the workers.
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if len(self.submitted) >= count:
                    return
            if time.monotonic() > deadline:
                raise AssertionError(
                    f"only {len(self.submitted)} of {count} jobs submitted in time"
                )
            time.sleep(0.02)

    def release(self):
        self._release.set()


def test_select_executor_defaults_to_local():
    assert isinstance(runner._select_executor(), LocalJobExecutor)


def test_loop_runs_pending_job_to_success(registered_jobs, job_store, executor):
    created = job_store.create_job("executor_engine_add", json.dumps({"x": 3, "y": 4}))

    executed = _run_to_completion(job_store, executor, lease_duration=60.0)

    assert executed == 1
    updated = job_store.get_job(created.job_id)
    assert updated.status == JobStatus.SUCCEEDED
    assert updated.result == "7"
    assert updated.retry_count == 0


def test_loop_records_failure(registered_jobs, job_store, executor):
    created = job_store.create_job("executor_engine_boom", "{}")

    _run_to_completion(job_store, executor, lease_duration=60.0)

    updated = job_store.get_job(created.job_id)
    assert updated.status == JobStatus.FAILED


def test_loop_no_pending_jobs_is_noop(registered_jobs, job_store, executor):
    assert _run_to_completion(job_store, executor, lease_duration=60.0) == 0


def test_loop_marks_timed_out_job(registered_jobs, job_store, executor):
    created = job_store.create_job(
        "executor_engine_sleep", json.dumps({"sleep_secs": 30}), timeout=1.0
    )

    _run_to_completion(job_store, executor, lease_duration=60.0)

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
    _run_to_completion(job_store, executor, lease_duration=60.0)
    after_first = job_store.get_job(created.job_id)
    assert after_first.status == JobStatus.PENDING
    assert after_first.retry_count == 1

    # Second poll: the retried job is re-claimed and succeeds.
    _run_to_completion(job_store, executor, lease_duration=60.0)
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
        executed = _run_to_completion(store, executor, lease_duration=60.0)

    assert executed == 1
    with WorkspaceContext("workspace-b"):
        updated = store.get_job(created.job_id)
    assert updated.status == JobStatus.SUCCEEDED
    assert updated.result == "11"


def test_scheduler_limits_concurrency_to_max_workers(registered_jobs, job_store):
    # executor_engine_add is declared with max_workers=1, so only one may run at a time; a second
    # PENDING job of the same type must stay queued until the first frees its slot.
    j1 = job_store.create_job("executor_engine_add", json.dumps({"x": 1, "y": 2}))
    j2 = job_store.create_job("executor_engine_add", json.dumps({"x": 3, "y": 4}))
    ex = _BlockingExecutor()
    scheduler = runner._JobScheduler(job_store, ex, lease_duration=60.0)

    # First tick fills the single slot with one job; the other cannot be claimed.
    assert scheduler.tick() == 1
    statuses = {job_store.get_job(j1.job_id).status, job_store.get_job(j2.job_id).status}
    assert statuses == {JobStatus.RUNNING, JobStatus.PENDING}
    # The slot is still held, so a further tick schedules nothing new.
    assert scheduler.tick() == 0

    # Let the in-flight job finish; its worker releases the slot.
    ex.release()
    scheduler.join(timeout=30.0)

    # With the slot free, the queued job is now schedulable.
    assert scheduler.tick() == 1
    scheduler.join(timeout=30.0)
    assert job_store.get_job(j1.job_id).status == JobStatus.SUCCEEDED
    assert job_store.get_job(j2.job_id).status == JobStatus.SUCCEEDED


def test_scheduler_forwards_cancellation_to_executor(registered_jobs, job_store):
    # A job cancelled while running must be stopped in the executor, not just marked CANCELED in
    # the store, so it cannot keep performing side effects after the caller cancelled it.
    created = job_store.create_job("executor_engine_add", json.dumps({"x": 1, "y": 2}))
    ex = _BlockingExecutor()
    scheduler = runner._JobScheduler(job_store, ex, lease_duration=60.0)

    assert scheduler.tick() == 1  # claims the job and starts its (blocked) worker
    _wait_submitted(scheduler, created.job_id)
    assert ex.submitted == [created.job_id]

    # The caller cancels the running job; the store row transitions to CANCELED.
    job_store.cancel_job(created.job_id)

    # The next tick's cancel sweep forwards the cancellation to the executor.
    scheduler.tick()
    scheduler.join(timeout=30.0)

    assert ex.canceled == [created.job_id]
    assert job_store.get_job(created.job_id).status == JobStatus.CANCELED
    # The worker finished and removed its bookkeeping, so nothing is left in flight.
    assert scheduler._in_flight == {}


def test_scheduler_isolates_concurrent_workspaces(registered_jobs, tmp_path, monkeypatch):
    from mlflow.entities import Workspace

    # Two jobs of a max_workers=2 type run concurrently, one per workspace. Each worker must
    # resolve its own workspace via the thread-local ContextVar; if they instead shared the
    # process-global env, the two in-flight workers would race and observe the wrong tenant.
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    store = WorkspaceAwareSqlAlchemyJobStore(f"sqlite:///{tmp_path / 'ws.db'}")
    with WorkspaceContext("workspace-a"):
        job_a = store.create_job("executor_engine_parallel", json.dumps({"x": 1}))
    with WorkspaceContext("workspace-b"):
        job_b = store.create_job("executor_engine_parallel", json.dumps({"x": 2}))

    ex = _BlockingExecutor()
    scheduler = runner._JobScheduler(store, ex, lease_duration=60.0)
    mock_workspace_store = mock.MagicMock()
    mock_workspace_store.list_workspaces.return_value = [
        Workspace(name="workspace-a"),
        Workspace(name="workspace-b"),
    ]
    with mock.patch(
        "mlflow.server.workspace_helpers._get_workspace_store",
        return_value=mock_workspace_store,
    ):
        # Both jobs are scheduled in one tick (two slots) and their workers block in wait_for_job.
        assert scheduler.tick() == 2
        ex.wait_until_submitted(count=2)
        ex.release()
        scheduler.join(timeout=30.0)

    # Each worker resolved its own workspace, not the other's.
    assert ex.observed_workspace == {job_a.job_id: "workspace-a", job_b.job_id: "workspace-b"}
    with WorkspaceContext("workspace-a"):
        assert store.get_job(job_a.job_id).status == JobStatus.SUCCEEDED
    with WorkspaceContext("workspace-b"):
        assert store.get_job(job_b.job_id).status == JobStatus.SUCCEEDED


def test_scheduler_fails_job_with_unresolvable_function(registered_jobs, job_store):
    # A persisted job whose function can't be resolved (e.g. renamed or removed across an upgrade)
    # must reach a terminal FAILED state, not stay PENDING and re-log on every tick.
    created = job_store.create_job("executor_engine_unknown", "{}")
    ex = _BlockingExecutor()
    scheduler = runner._JobScheduler(job_store, ex, lease_duration=60.0)

    assert scheduler.tick() == 0
    assert ex.submitted == []
    assert job_store.get_job(created.job_id).status == JobStatus.FAILED


def test_scheduler_skips_pending_job_already_in_flight(registered_jobs, job_store):
    # A job a prior worker re-pended (e.g. during a transient-retry backoff) is still in
    # _in_flight. Even with a free slot (max_workers=2), the scheduler must not re-claim it, or
    # the backoff would be bypassed by a concurrent re-run.
    created = job_store.create_job("executor_engine_parallel", json.dumps({"x": 1}))
    ex = _BlockingExecutor()
    scheduler = runner._JobScheduler(job_store, ex, lease_duration=60.0)
    with scheduler._in_flight_lock:
        scheduler._in_flight[created.job_id] = runner._InFlightJob(
            workspace=None, thread=threading.current_thread()
        )

    assert scheduler._schedule_pending() == 0
    assert ex.submitted == []
    assert job_store.get_job(created.job_id).status == JobStatus.PENDING


def test_scheduler_retries_cancel_forward_when_executor_raises(registered_jobs, job_store):
    # If forwarding a cancellation to the executor raises, the job must not be marked as
    # forwarded, so a later tick retries the cancel instead of letting the canceled job run on.
    created = job_store.create_job("executor_engine_add", json.dumps({"x": 1, "y": 2}))
    ex = _BlockingExecutor()
    cancel_calls = []
    forward_ok = ex.cancel_job

    def flaky_cancel(job_id):
        cancel_calls.append(job_id)
        if len(cancel_calls) == 1:
            raise RuntimeError("backend blip")
        forward_ok(job_id)

    ex.cancel_job = flaky_cancel
    scheduler = runner._JobScheduler(job_store, ex, lease_duration=60.0)

    assert scheduler.tick() == 1
    _wait_submitted(scheduler, created.job_id)
    job_store.cancel_job(created.job_id)

    scheduler.tick()  # first forward attempt raises and is not marked forwarded
    assert cancel_calls == [created.job_id]
    scheduler.tick()  # retried on the next tick
    scheduler.join(timeout=30.0)

    # The forward was attempted again after the first failure (the retry is the point here; the
    # row was already CANCELED by the cancel_job call above, so its status alone proves nothing).
    assert cancel_calls == [created.job_id, created.job_id]


def test_record_result_canceled_result_finalizes_running_row(registered_jobs, job_store):
    # An executor that reports CANCELED while the store row is still RUNNING (e.g. a backend
    # self-cancel) must finalize the row to CANCELED, not leave it stuck RUNNING.
    created = job_store.create_job("executor_engine_add", json.dumps({"x": 1, "y": 2}))
    job_store.claim_job(created.job_id, lease_duration=60.0)
    assert job_store.get_job(created.job_id).status == JobStatus.RUNNING

    runner._record_result(
        job_store, created.job_id, "executor_engine_add", JobResult(status=JobStatus.CANCELED)
    )

    assert job_store.get_job(created.job_id).status == JobStatus.CANCELED


def test_scheduler_skips_submission_for_job_canceled_before_submit(registered_jobs, job_store):
    # A job cancelled after the claim but before the worker submits must not be submitted at all:
    # there is no backend job to cancel yet, so submitting would run the already-canceled job.
    created = job_store.create_job("executor_engine_add", json.dumps({"x": 1, "y": 2}))
    job_store.claim_job(created.job_id, lease_duration=60.0)
    job_store.cancel_job(created.job_id)
    ex = _BlockingExecutor()

    # _execute_claimed_job runs on the worker; drive it directly for the pre-submit check.
    runner._execute_claimed_job(job_store, ex, created)

    assert ex.submitted == []
    assert job_store.get_job(created.job_id).status == JobStatus.CANCELED


def test_build_execution_context_uses_backend_store_uri(monkeypatch, registered_jobs):
    from mlflow.server.constants import BACKEND_STORE_URI_ENV_VAR

    # The job runs privileged work against the store, so it gets the backend store URI directly
    # rather than the server's HTTP tracking URI (which has no general auth story when protected).
    # An HTTP MLFLOW_TRACKING_URI in the runner env must be ignored for the job's tracking URI.
    monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, "sqlite:///backend.db")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    job = _make_job(status=JobStatus.PENDING)

    context = runner._build_execution_context(job)

    assert context.tracking_uri == "sqlite:///backend.db"


@pytest.mark.parametrize("orphan_status", [JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY])
def test_recover_orphaned_executor_jobs_resets_to_pending(
    registered_jobs, job_store, monkeypatch, orphan_status
):
    # A job left RUNNING or NEEDS_RECOVERY by a previous server generation (created at/before this
    # launch) is reset to PENDING so the scheduler re-claims it, while a RUNNING job created after
    # launch (i.e. on the live server) is left alone. NEEDS_RECOVERY is the state the shutdown
    # handoff (mark_orphans_for_recovery) writes, so both must be recovered.
    old = job_store.create_job("executor_engine_add", json.dumps({"x": 1, "y": 2}))
    job_store.claim_job(old.job_id, lease_duration=60.0)
    if orphan_status == JobStatus.NEEDS_RECOVERY:
        job_store.mark_job_needs_recovery(old.job_id)
    assert job_store.get_job(old.job_id).status == orphan_status
    launch_ts = job_store.get_job(old.job_id).creation_time
    monkeypatch.setenv("_MLFLOW_SERVER_UP_TIME", str(launch_ts))

    time.sleep(0.01)  # ensure the next job's creation_time is strictly after launch_ts
    new = job_store.create_job("executor_engine_add", json.dumps({"x": 3, "y": 4}))
    job_store.claim_job(new.job_id, lease_duration=60.0)
    assert job_store.get_job(new.job_id).creation_time > launch_ts

    runner._recover_orphaned_executor_jobs(job_store)

    assert job_store.get_job(old.job_id).status == JobStatus.PENDING
    assert job_store.get_job(new.job_id).status == JobStatus.RUNNING


def test_recover_orphaned_executor_jobs_skips_without_launch_time(
    registered_jobs, job_store, monkeypatch
):
    # Without a recorded launch time, recovery cannot bound itself to the previous generation, so
    # it must skip rather than risk resetting freshly submitted jobs.
    created = job_store.create_job("executor_engine_add", json.dumps({"x": 1, "y": 2}))
    job_store.claim_job(created.job_id, lease_duration=60.0)
    monkeypatch.delenv("_MLFLOW_SERVER_UP_TIME", raising=False)

    runner._recover_orphaned_executor_jobs(job_store)

    assert job_store.get_job(created.job_id).status == JobStatus.RUNNING


def test_fail_claimed_job_retries_once_on_transient_store_error(registered_jobs):
    # The managed session surfaces a transient DB error as MlflowException(TEMPORARILY_UNAVAILABLE)
    # — the type the real store actually raises — and it must be retried so a RUNNING row is not
    # stranded.
    store = mock.MagicMock()
    store.fail_job.side_effect = [
        MlflowException("db unavailable", error_code=TEMPORARILY_UNAVAILABLE),
        None,
    ]
    scheduler = runner._JobScheduler(store, _BlockingExecutor(), lease_duration=60.0)

    scheduler._fail_claimed_job("job-1", None, "boom")

    assert store.fail_job.call_count == 2
    store.fail_job.assert_called_with("job-1", "boom")


def test_fail_claimed_job_logs_and_stops_on_unexpected_error(registered_jobs):
    # A non-MlflowException from the store is unexpected and not retryable: log once and stop
    # (don't retry, don't propagate out of the fail path).
    store = mock.MagicMock()
    store.fail_job.side_effect = RuntimeError("unexpected")
    scheduler = runner._JobScheduler(store, _BlockingExecutor(), lease_duration=60.0)

    scheduler._fail_claimed_job("job-1", None, "boom")

    store.fail_job.assert_called_once_with("job-1", "boom")


def test_schedule_pending_releases_slot_if_start_worker_raises(registered_jobs, job_store):
    # If _start_worker raises unexpectedly after the slot is acquired and the job claimed, the
    # scheduler must release the slot so the job type is not permanently at capacity.
    job_store.create_job("executor_engine_add", json.dumps({"x": 1, "y": 2}))
    ex = _BlockingExecutor()
    scheduler = runner._JobScheduler(job_store, ex, lease_duration=60.0)

    with mock.patch.object(scheduler, "_start_worker", side_effect=RuntimeError("boom")):
        assert scheduler._schedule_pending() == 0

    # The slot (max_workers=1 for executor_engine_add) was released, so it can be acquired again.
    assert scheduler._slot_for("executor_engine_add").acquire(blocking=False)


def test_fail_claimed_job_does_not_retry_invalid_transition(registered_jobs):
    # An invalid transition (the row is no longer RUNNING) surfaces as an INVALID_PARAMETER_VALUE
    # MlflowException; it can't succeed on retry, so it is logged once and not retried.
    store = mock.MagicMock()
    store.fail_job.side_effect = MlflowException.invalid_parameter_value("invalid transition")
    scheduler = runner._JobScheduler(store, _BlockingExecutor(), lease_duration=60.0)

    scheduler._fail_claimed_job("job-1", None, "boom")

    store.fail_job.assert_called_once_with("job-1", "boom")


def test_run_executor_loop_recovers_then_exits_on_stop(registered_jobs, job_store, monkeypatch):
    # run_executor_loop assumes an already-started executor, recovers orphaned rows before the
    # claim loop, and exits cleanly when stop_event is set. Pre-set the stop_event so the loop
    # body does not run and the test stays deterministic.
    orphan = job_store.create_job("executor_engine_add", json.dumps({"x": 1, "y": 2}))
    job_store.claim_job(orphan.job_id, lease_duration=60.0)
    monkeypatch.setenv(
        "_MLFLOW_SERVER_UP_TIME", str(job_store.get_job(orphan.job_id).creation_time)
    )

    ex = _BlockingExecutor()
    stop = threading.Event()
    stop.set()
    with mock.patch("mlflow.server.handlers._get_job_store", return_value=job_store):
        runner.run_executor_loop(ex, stop_event=stop, poll_interval=0.01)

    # Recovery ran before the loop returned.
    assert job_store.get_job(orphan.job_id).status == JobStatus.PENDING


def test_scheduler_marks_in_flight_jobs_for_recovery_on_shutdown(registered_jobs, job_store):
    # Workers are daemon threads; any still in flight at shutdown must be flagged so the next
    # launch's recovery can re-queue them instead of leaving the row stuck RUNNING.
    created = job_store.create_job("executor_engine_add", json.dumps({"x": 1, "y": 2}))
    ex = _BlockingExecutor()
    scheduler = runner._JobScheduler(job_store, ex, lease_duration=60.0)
    assert scheduler.tick() == 1
    _wait_submitted(scheduler, created.job_id)

    scheduler.mark_orphans_for_recovery()

    assert job_store.get_job(created.job_id).status == JobStatus.NEEDS_RECOVERY
    ex.release()
    scheduler.join(timeout=30.0)


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
    _run_to_completion(exec_store, executor, lease_duration=60.0)
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
        "executor_engine_add",
        json.dumps({"x": 1, "y": 2}),
        None,
        creator=None,
        executor_backend="local",
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


# "huey" is explicitly rejected: unset uses the default engine, and "executor" is the only
# settable value, so pinning "huey" by name is not allowed (it would break once Huey is retired).
@pytest.mark.parametrize("engine", ["spark", "kubernetes", "huey", ""])
def test_submit_job_rejects_invalid_engine(monkeypatch, registered_jobs, engine):
    _submit_job_env(monkeypatch, engine=engine)
    store = mock.MagicMock()
    store.create_job.return_value = _make_job(status=JobStatus.PENDING)
    with (
        mock.patch("mlflow.server.jobs._get_job_store", return_value=store),
        mock.patch("mlflow.server.jobs.utils._check_requirements"),
        pytest.raises(MlflowException, match="may only be set to 'executor'"),
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
        (None, "_launch_job_runner"),  # unset -> default engine (Huey)
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
    with runner._LeaseRenewer(store, "job-1", lease_duration=0.6, workspace=None):
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
    with runner._LeaseRenewer(store, "job-1", lease_duration=0.6, workspace=None):
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
        with runner._LeaseRenewer(store, "job-1", lease_duration=0.6, workspace=None):
            # The first renewal raises; the renewer must log and keep going to a second.
            assert reached_second.wait(timeout=5.0)
    mock_logger.exception.assert_called()
    assert store.renew_job_lease.call_count >= 2


@pytest.mark.parametrize("lease_duration", [runner._MIN_LEASE_TTL, 60.0])
def test_lease_renewer_interval_below_ttl_and_at_or_above_floor(lease_duration):
    # For any accepted TTL (>= _MIN_LEASE_TTL) the renewal interval fires before the lease expires
    # (strictly below the TTL) while never dropping below the busy-loop floor.
    renewer = runner._LeaseRenewer(mock.MagicMock(), "job-1", lease_duration, workspace=None)
    assert runner._MIN_LEASE_RENEW_INTERVAL <= renewer._interval < lease_duration


def test_scheduler_rejects_lease_ttl_below_minimum():
    # A configured TTL too small to renew before expiry is rejected at startup rather than left to
    # busy-loop the store; the boundary value itself is accepted.
    with pytest.raises(MlflowException, match="job lease TTL must be at least"):
        runner._JobScheduler(mock.MagicMock(), mock.MagicMock(), lease_duration=0.1)
    runner._JobScheduler(mock.MagicMock(), mock.MagicMock(), lease_duration=runner._MIN_LEASE_TTL)


def test_lease_renewed_during_submit_job(registered_jobs, job_store, monkeypatch):
    # Renewal must cover submit_job, not just wait_for_job: a submit_job that outlasts the lease
    # (e.g. installing a job environment) must still see renewals. Prove it by blocking submit_job
    # until a renewal has landed — which can only happen if the renewer started before submission.
    created = job_store.create_job("executor_engine_add", json.dumps({"x": 1, "y": 2}))
    job_store.claim_job(created.job_id, lease_duration=60.0)
    job = job_store.get_job(created.job_id)

    renewed = threading.Event()
    real_renew = job_store.renew_job_lease

    def _spy(job_id, lease_duration):
        renewed.set()
        return real_renew(job_id, lease_duration)

    monkeypatch.setattr(job_store, "renew_job_lease", _spy)

    executor = mock.MagicMock()
    executor.submit_job.side_effect = lambda **kwargs: renewed.wait(timeout=5.0)
    executor.wait_for_job.return_value = JobResult(status=JobStatus.SUCCEEDED, result="ok")

    runner._execute_claimed_job(
        job_store, executor, job, lease_duration=runner._MIN_LEASE_TTL, workspace=None
    )

    executor.submit_job.assert_called_once()
    executor.wait_for_job.assert_called_once()
    assert renewed.is_set()


def test_long_running_job_lease_is_renewed(registered_jobs, job_store, executor, monkeypatch):
    created = job_store.create_job("executor_engine_sleep", json.dumps({"sleep_secs": 1.0}))
    renewed_ok = []
    real_renew = job_store.renew_job_lease

    def _spy(job_id, lease_duration):
        # Record only successful renewals, so the assertion proves renewal actually landed rather
        # than merely that it was attempted.
        status = real_renew(job_id, lease_duration)
        renewed_ok.append(job_id)
        return status

    monkeypatch.setattr(job_store, "renew_job_lease", _spy)
    # Short lease so renewal (~lease/3) fires several times during the ~1s job.
    _run_to_completion(job_store, executor, lease_duration=0.6)

    assert created.job_id in renewed_ok
    assert job_store.get_job(created.job_id).status == JobStatus.SUCCEEDED


def test_long_running_job_lease_is_renewed_with_workspaces(registered_jobs, tmp_path, monkeypatch):
    # The renewer runs on its own thread; with workspaces enabled it must rebind the workspace so
    # renew_job_lease resolves the tenant instead of raising "Active workspace is required". Without
    # the workspace threading this fails: no renewal lands and the job would still succeed, so the
    # renewal assertion is what proves the cross-thread workspace fix.
    from mlflow.entities import Workspace

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    store = WorkspaceAwareSqlAlchemyJobStore(f"sqlite:///{tmp_path / 'ws.db'}")
    with WorkspaceContext("workspace-b"):
        created = store.create_job("executor_engine_sleep", json.dumps({"sleep_secs": 1.0}))

    renewed_ok = []
    real_renew = store.renew_job_lease

    def _spy(job_id, lease_duration):
        # real_renew raises "Active workspace is required" if the renewer thread didn't rebind the
        # workspace (the bug), so recording only after it returns makes this assertion prove the
        # cross-thread workspace fix rather than just the attempt.
        status = real_renew(job_id, lease_duration)
        renewed_ok.append(job_id)
        return status

    monkeypatch.setattr(store, "renew_job_lease", _spy)

    ex = LocalJobExecutor(JobExecutorConfig(default_timeout=60.0))
    ex.start_executor()
    mock_workspace_store = mock.MagicMock()
    mock_workspace_store.list_workspaces.return_value = [Workspace(name="workspace-b")]
    try:
        with mock.patch(
            "mlflow.server.workspace_helpers._get_workspace_store",
            return_value=mock_workspace_store,
        ):
            _run_to_completion(store, ex, lease_duration=0.6)
    finally:
        ex.stop_executor()

    assert created.job_id in renewed_ok
    with WorkspaceContext("workspace-b"):
        assert store.get_job(created.job_id).status == JobStatus.SUCCEEDED
