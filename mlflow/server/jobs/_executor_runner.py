"""Executor-engine job runner.

Launched in place of the Huey ``_job_runner`` when
``MLFLOW_SERVER_JOB_EXECUTION_ENGINE=executor``. It runs a loop that claims
PENDING jobs from the job store and executes them through the configured
``AbstractJobExecutor`` backend (``LocalJobExecutor`` by default), then records
the terminal state back to the store.

Only job execution moves to the executor framework here. Periodic tasks (e.g. the
online scoring scheduler) still run on Huey via ``_launch_periodic_tasks_consumer``.
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

# Use an explicit logger name rather than __name__: this module is launched as
# ``python -m mlflow.server.jobs._executor_runner``, where __name__ is "__main__" and would
# fall outside the "mlflow" logger hierarchy, so its logs would miss MLflow's log handler.
_logger = logging.getLogger("mlflow.server.jobs._executor_runner")

# How long the loop sleeps when it finds no PENDING jobs to run.
_POLL_INTERVAL = 1.0


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
    # Mirror the Huey path's exponential backoff. The loop is serial for now, so
    # this briefly blocks other jobs; non-blocking scheduling is a later refinement.
    base_delay = MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY.get()
    max_delay = MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY.get()
    time.sleep(min(base_delay * (2 ** (retry_count - 1)), max_delay))


def _record_result(
    job_store: AbstractJobStore, job_id: str, job_name: str, result: JobResult
) -> None:
    """Map an executor ``JobResult`` onto the terminal job-store transition.

    Mirrors the outcome handling in the Huey ``_exec_job`` path.
    """
    # If the job was canceled after it was claimed, it still ran to completion, but the
    # cancel already finalized the row. Recording a terminal result now would raise an
    # invalid-transition error and log a scary traceback for a normal user action, so skip it.
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


def _poll_and_execute_once(
    job_store: AbstractJobStore, executor: AbstractJobExecutor, lease_duration: float | None
) -> int:
    """Claim and run every currently-PENDING job. Returns the number executed.

    A single iteration of the loop; separated out so it can be driven directly in tests.

    Workspace scoping mirrors the Huey ``_enqueue_unfinished_jobs`` path: on a
    workspace-enabled deployment the store only returns jobs for the active workspace, so
    the listing, claim, execution, and result-recording for each tenant all run inside that
    tenant's ``WorkspaceContext``. When workspaces are disabled this is a single
    ``nullcontext`` and behavior is unchanged.
    """
    from mlflow.server.jobs.utils import _workspace_contexts_for_recovery

    executed = 0
    for workspace_ctx in _workspace_contexts_for_recovery():
        with workspace_ctx:
            # TODO (follow-up): this claims and runs PENDING jobs serially. Two things are not
            # yet enforced here and land in follow-ups: (1) per-function ``max_workers`` and
            # concurrent scheduling, so one job type cannot starve another; (2) exclusive job
            # locking by key (e.g. keeping online scoring exclusive per experiment), which
            # builds on the job-lock work in #25200.
            for job in list(job_store.list_jobs(statuses=[JobStatus.PENDING])):
                if job_store.claim_job(job.job_id, lease_duration) != JobUpdateStatus.APPLIED:
                    # A concurrent worker claimed it first, or its status changed.
                    continue
                try:
                    _execute_claimed_job(job_store, executor, job)
                except Exception as exc:
                    # A claimed job left RUNNING would be stuck; fail it so recovery is
                    # not needed.
                    _logger.error(
                        f"Job {job.job_id} ({job.job_name}) raised in the executor loop: {exc!r}",
                        exc_info=True,
                    )
                    try:
                        job_store.fail_job(job.job_id, repr(exc))
                    except Exception:
                        _logger.exception(f"Failed to transition job {job.job_id} to FAILED")
                executed += 1
    return executed


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
    # start_executor() runs inside the try so a failure during startup (e.g. a plugin backend
    # that allocates resources and then raises) still triggers stop_executor() for cleanup.
    try:
        executor.start_executor()
        _logger.info(
            "Started executor-backed job runner "
            f"(backend={MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND.get()})"
        )
        while not stop_event.is_set():
            try:
                executed = _poll_and_execute_once(job_store, executor, lease_duration)
            except Exception:
                # The loop-level store and workspace calls run outside the per-job handler.
                # A transient error here must not kill the runner (nothing would restart it),
                # so log it and try again on the next poll.
                _logger.exception("Executor job loop iteration failed; continuing.")
                executed = 0
            if executed:
                _logger.debug("Executor engine processed %d job(s) this poll.", executed)
            if executed == 0:
                stop_event.wait(poll_interval)
    finally:
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
