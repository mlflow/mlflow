"""Executor-engine job runner.

Launched in place of the Huey ``_job_runner`` when
``MLFLOW_SERVER_JOB_EXECUTION_ENGINE=executor``. A scheduler loop claims PENDING jobs from the
job store and runs each in its own worker thread through the configured ``AbstractJobExecutor``
backend (``LocalJobExecutor`` by default), then records the terminal state back to the store.

Concurrency is bounded per job function by a semaphore sized to that function's ``max_workers``,
so one job type cannot starve another and the scheduler thread never blocks on a running job.
Each tick also forwards store-side cancellations to the executor for in-flight jobs.

Only job execution moves to the executor framework here. Periodic tasks (e.g. the online scoring
scheduler) still run on Huey via ``_launch_periodic_tasks_consumer``.
"""

import json
import logging
import threading
import time

from mlflow.entities._job import Job
from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import (
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_GATEWAY_URI,
    MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND,
    MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY,
    MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY,
    MLFLOW_TRACKING_URI,
)
from mlflow.server.jobs.executor import AbstractJobExecutor, JobExecutionContext, JobResult
from mlflow.server.jobs.executor_registry import get_executor_registry
from mlflow.store.jobs.abstract_store import AbstractJobStore, JobUpdateStatus
from mlflow.utils.workspace_context import ServerWorkspaceContext

# Use an explicit logger name rather than __name__: this module is launched as
# ``python -m mlflow.server.jobs._executor_runner``, where __name__ is "__main__" and would
# fall outside the "mlflow" logger hierarchy, so its logs would miss MLflow's log handler.
_logger = logging.getLogger("mlflow.server.jobs._executor_runner")

# How long the scheduler sleeps between ticks (workers run in the background between ticks).
_POLL_INTERVAL = 1.0
# On shutdown, how long to wait for each in-flight worker before stopping the executor.
_SHUTDOWN_JOIN_TIMEOUT = 5.0


def _select_executor() -> AbstractJobExecutor:
    """Resolve the executor backend named by ``MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND``.

    Defaults to ``local`` (``LocalJobExecutor``); a configured plugin backend is honored too.

    TODO (follow-up): select the backend per job at submit time via a job-executor router
    (matching the job against the configured executor) rather than a single process-wide
    backend, so different job types can target different executors.
    """
    backend = MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND.get()
    return get_executor_registry().get(backend)


def _build_execution_context(job: Job) -> JobExecutionContext:
    workspace = job.workspace if MLFLOW_ENABLE_WORKSPACES.get() else None
    return JobExecutionContext(
        job_id=job.job_id,
        tracking_uri=MLFLOW_TRACKING_URI.get(),
        gateway_uri=MLFLOW_GATEWAY_URI.get(),
        workspace=workspace,
    )


def _backoff_after_transient_retry(retry_count: int) -> None:
    # Mirror the Huey path's exponential backoff. This runs on the per-job worker thread, so it
    # only delays that job's slot — it does not block the scheduler or other job types.
    base_delay = MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY.get()
    max_delay = MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY.get()
    time.sleep(min(base_delay * (2 ** (retry_count - 1)), max_delay))


def _record_result(
    job_store: AbstractJobStore, job_id: str, job_name: str, result: JobResult
) -> None:
    """Map an executor ``JobResult`` onto the terminal job-store transition.

    Mirrors the outcome handling in the Huey ``_exec_job`` path.
    """
    # If the job was canceled after it was claimed, it still ran to completion (or was killed by
    # a forwarded cancel), but the cancel already finalized the row. Recording a terminal result
    # now would raise an invalid-transition error and log a scary traceback for a normal user
    # action, so skip it.
    if job_store.get_job(job_id).status == JobStatus.CANCELED:
        return

    if result.status == JobStatus.SUCCEEDED:
        job_store.report_job_result(job_id, JobStatus.SUCCEEDED, result=result.result)
    elif result.status == JobStatus.TIMEOUT:
        job_store.report_job_result(job_id, JobStatus.TIMEOUT, error_message=result.error_message)
    elif result.status == JobStatus.CANCELED:
        # Defensive: cancellation moves the job to a terminal CANCELED state through
        # cancel_job(), so there is nothing to record here.
        return
    elif result.is_transient_error:
        # A transient error resets the job to PENDING (non-terminal) so a later poll can
        # re-claim it, so this cannot go through report_job_result.
        retry_count = job_store.retry_or_fail_job(job_id, result.error_message or "")
        if retry_count is not None:
            _backoff_after_transient_retry(retry_count)
    else:
        _logger.error(f"Job {job_id} ({job_name}) failed with error: {result.error_message}")
        job_store.report_job_result(
            job_id, JobStatus.FAILED, error_message=result.error_message or ""
        )


def _execute_claimed_job(
    job_store: AbstractJobStore, executor: AbstractJobExecutor, job: Job
) -> None:
    """Execute a job that has already been claimed (moved to RUNNING) and record its result."""
    from mlflow.server.jobs.utils import _load_function, get_job_fn_fullname

    fn_fullname = get_job_fn_fullname(job.job_name)
    function = _load_function(fn_fullname)
    python_env = function._job_fn_metadata.python_env
    params = json.loads(job.params)
    context = _build_execution_context(job)

    backend = MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND.get()
    _logger.info(f"Executor engine running job {job.job_id} ({job.job_name}) on backend {backend}")

    # Contract: every submit_job pairs with exactly one wait_for_job.
    executor.submit_job(
        job_id=job.job_id,
        job_name=job.job_name,
        fn_fullname=fn_fullname,
        params=params,
        context=context,
        python_env=python_env,
        timeout=job.timeout,
    )
    result = executor.wait_for_job(job.job_id)
    _logger.info(f"Executor engine job {job.job_id} finished with status {result.status.value}")
    _record_result(job_store, job.job_id, job.job_name, result)


def _max_workers_for(job_name: str) -> int:
    """Concurrency limit for a job function, from its ``@job(max_workers=...)`` metadata."""
    from mlflow.server.jobs.utils import _load_function, get_job_fn_fullname

    max_workers = _load_function(get_job_fn_fullname(job_name))._job_fn_metadata.max_workers
    return max(1, max_workers or 1)


class _JobScheduler:
    """Non-blocking scheduler for the executor engine.

    Each ``tick()`` (1) forwards store-side cancellations to the executor for in-flight jobs, then
    (2) claims PENDING jobs it has capacity for and runs each in a worker thread. A per-``job_name``
    semaphore sized to that function's ``max_workers`` bounds how many jobs of a given type run at
    once (a process-wide budget per job type, matching the Huey engine's per-function pool), so the
    scheduler thread never blocks on a running job.

    Worker threads re-enter the job's workspace with ``ServerWorkspaceContext``, which binds only
    the thread-local request ContextVar. ``WorkspaceContext`` must not be used here: it also mutates
    the process-global ``MLFLOW_WORKSPACE`` env, which concurrent workers in different workspaces
    would race on. The store resolves the ContextVar first (see ``get_request_workspace``).
    """

    def __init__(
        self,
        job_store: AbstractJobStore,
        executor: AbstractJobExecutor,
        lease_duration: float | None,
    ) -> None:
        self._job_store = job_store
        self._executor = executor
        self._lease_duration = lease_duration
        # One semaphore per job_name (value = that function's max_workers).
        self._slots: dict[str, threading.Semaphore] = {}
        self._slots_lock = threading.Lock()
        # job_id -> (workspace, worker thread), for the cancel sweep and shutdown join.
        self._in_flight: dict[str, tuple[str | None, threading.Thread]] = {}
        # job_ids whose cancellation was already forwarded to the executor, so cancel_job is
        # called at most once per in-flight job (AbstractJobExecutor.cancel_job does not promise
        # idempotency for plugin backends). Guarded by _in_flight_lock.
        self._canceled_forwarded: set[str] = set()
        self._in_flight_lock = threading.Lock()

    def _slot_for(self, job_name: str) -> threading.Semaphore:
        with self._slots_lock:
            sem = self._slots.get(job_name)
            if sem is None:
                sem = threading.Semaphore(_max_workers_for(job_name))
                self._slots[job_name] = sem
            return sem

    def tick(self) -> int:
        """Run one scheduler iteration. Returns the number of jobs newly scheduled."""
        self._forward_cancellations()
        return self._schedule_pending()

    def _forward_cancellations(self) -> None:
        with self._in_flight_lock:
            snapshot = list(self._in_flight.items())
        for job_id, (workspace, _thread) in snapshot:
            try:
                with ServerWorkspaceContext(workspace):
                    canceled = self._job_store.get_job(job_id).status == JobStatus.CANCELED
            except Exception:
                _logger.exception("Failed to check cancellation state for job %s", job_id)
                continue
            if not canceled:
                continue
            with self._in_flight_lock:
                if job_id in self._canceled_forwarded:
                    continue
                self._canceled_forwarded.add(job_id)
            try:
                # Stop the still-running job so it cannot keep performing side effects after the
                # caller cancelled it. The worker's wait_for_job then returns and _record_result
                # skips the already-CANCELED row.
                self._executor.cancel_job(job_id)
            except Exception:
                _logger.exception("Failed to forward cancellation to executor for %s", job_id)

    def _schedule_pending(self) -> int:
        from mlflow.server.jobs.utils import _workspace_contexts_for_recovery

        scheduled = 0
        for workspace_ctx in _workspace_contexts_for_recovery():
            with workspace_ctx as workspace:
                # TODO (follow-up): exclusive job-locking by key (e.g. keeping online scoring
                # exclusive per experiment) is not enforced here; it builds on the job-lock work
                # in #25200 and lands once that is available.
                for job in list(self._job_store.list_jobs(statuses=[JobStatus.PENDING])):
                    try:
                        sem = self._slot_for(job.job_name)
                    except Exception as exc:
                        # The function backing this job cannot be resolved (e.g. it was renamed or
                        # removed across an upgrade). Fail it, matching the old serial path, so it
                        # reaches a terminal state instead of staying PENDING and re-logging on
                        # every tick.
                        self._fail_unschedulable_job(job, exc)
                        continue
                    if not sem.acquire(blocking=False):
                        # This job type is already at max_workers; leave it PENDING for a
                        # later tick, once a slot frees.
                        continue
                    try:
                        claimed = (
                            self._job_store.claim_job(job.job_id, self._lease_duration)
                            == JobUpdateStatus.APPLIED
                        )
                    except Exception:
                        sem.release()
                        _logger.exception("Failed to claim job %s", job.job_id)
                        continue
                    if not claimed:
                        # A concurrent worker claimed it first, or its status changed.
                        sem.release()
                        continue
                    if self._start_worker(job, workspace, sem):
                        scheduled += 1
        return scheduled

    def _fail_unschedulable_job(self, job: Job, exc: Exception) -> None:
        """Claim and fail a PENDING job whose function cannot be resolved."""
        try:
            claimed = (
                self._job_store.claim_job(job.job_id, self._lease_duration)
                == JobUpdateStatus.APPLIED
            )
        except Exception:
            _logger.exception("Failed to claim unschedulable job %s", job.job_id)
            return
        if not claimed:
            return
        _logger.error("Job %s (%s) is not runnable: %r", job.job_id, job.job_name, exc)
        try:
            self._job_store.fail_job(job.job_id, repr(exc))
        except Exception:
            _logger.exception("Failed to transition unschedulable job %s to FAILED", job.job_id)

    def _start_worker(self, job: Job, workspace: str | None, sem: threading.Semaphore) -> bool:
        """Start the worker thread for a claimed job. Returns whether it started.

        If the thread cannot be started (e.g. the OS thread limit is hit), the claim's effects are
        undone — the slot is released and the job is failed — so the slot is not leaked and the job
        does not stay RUNNING with nothing running it.
        """
        thread = threading.Thread(
            target=self._run_worker,
            args=(job, workspace, sem),
            name=f"mlflow-executor-job-{job.job_id}",
            daemon=True,
        )
        with self._in_flight_lock:
            self._in_flight[job.job_id] = (workspace, thread)
        try:
            thread.start()
        except Exception:
            _logger.exception("Failed to start worker thread for job %s", job.job_id)
            with self._in_flight_lock:
                self._in_flight.pop(job.job_id, None)
            sem.release()
            try:
                with ServerWorkspaceContext(workspace):
                    self._job_store.fail_job(job.job_id, "Failed to start executor worker thread")
            except Exception:
                _logger.exception(
                    "Failed to transition job %s to FAILED after worker start failure", job.job_id
                )
            return False
        return True

    def _run_worker(self, job: Job, workspace: str | None, sem: threading.Semaphore) -> None:
        try:
            with ServerWorkspaceContext(workspace):
                _execute_claimed_job(self._job_store, self._executor, job)
        except Exception as exc:
            # A claimed job left RUNNING would be stuck; fail it so recovery is not needed.
            _logger.error(
                "Job %s (%s) raised in the executor worker: %r",
                job.job_id,
                job.job_name,
                exc,
                exc_info=True,
            )
            try:
                with ServerWorkspaceContext(workspace):
                    self._job_store.fail_job(job.job_id, repr(exc))
            except Exception:
                _logger.exception("Failed to transition job %s to FAILED", job.job_id)
        finally:
            sem.release()
            with self._in_flight_lock:
                self._in_flight.pop(job.job_id, None)
                self._canceled_forwarded.discard(job.job_id)

    def join(self, timeout: float | None = None) -> None:
        """Best-effort wait for in-flight workers to finish.

        ``timeout`` is a total budget across all workers, not per worker, so shutdown is bounded.
        """
        with self._in_flight_lock:
            threads = [thread for _workspace, thread in self._in_flight.values()]
        if timeout is None:
            for thread in threads:
                thread.join()
            return
        deadline = time.monotonic() + timeout
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(remaining)


def run_executor_loop(
    stop_event: threading.Event | None = None, poll_interval: float = _POLL_INTERVAL
) -> None:
    """Run the executor job-execution loop until ``stop_event`` is set."""
    from mlflow.server.handlers import _get_job_store

    # Only PENDING jobs are claimed here. Recovering orphaned RUNNING jobs on server
    # restart is a follow-up; this loop does not handle it yet.
    stop_event = stop_event or threading.Event()
    job_store = _get_job_store()
    executor = _select_executor()
    lease_duration = executor.config.job_lease_ttl
    scheduler: _JobScheduler | None = None
    # start_executor() runs inside the try so a failure during startup (e.g. a plugin backend
    # that allocates resources and then raises) still triggers stop_executor() for cleanup.
    try:
        executor.start_executor()
        _logger.info(
            "Started executor-backed job runner "
            f"(backend={MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND.get()})"
        )
        scheduler = _JobScheduler(job_store, executor, lease_duration)
        while not stop_event.is_set():
            try:
                scheduled = scheduler.tick()
            except Exception:
                # The tick's store and workspace calls run outside the per-job worker. A
                # transient error here must not kill the scheduler (nothing would restart it),
                # so log it and try again on the next poll.
                _logger.exception("Executor job scheduler tick failed; continuing.")
                scheduled = 0
            if scheduled:
                _logger.debug("Executor engine scheduled %d job(s) this tick.", scheduled)
            stop_event.wait(poll_interval)
    finally:
        if scheduler is not None:
            scheduler.join(timeout=_SHUTDOWN_JOIN_TIMEOUT)
        executor.stop_executor()


def main() -> None:
    from mlflow.server.jobs.logging_utils import configure_logging_for_jobs
    from mlflow.server.jobs.utils import (
        _launch_periodic_tasks_consumer,
        _start_watcher_to_kill_job_runner_if_mlflow_server_dies,
    )

    configure_logging_for_jobs()
    _start_watcher_to_kill_job_runner_if_mlflow_server_dies()
    # Periodic tasks (e.g. the online scoring scheduler) still run on Huey.
    _launch_periodic_tasks_consumer()
    run_executor_loop()


if __name__ == "__main__":
    main()
