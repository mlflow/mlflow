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
from contextlib import nullcontext

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
from mlflow.utils.time import get_current_time_millis

# Use an explicit logger name rather than __name__: this module is launched as
# ``python -m mlflow.server.jobs._executor_runner``, where __name__ is "__main__" and would
# fall outside the "mlflow" logger hierarchy, so its logs would miss MLflow's log handler.
_logger = logging.getLogger("mlflow.server.jobs._executor_runner")

# How long the loop sleeps when it finds no PENDING jobs to run.
_POLL_INTERVAL = 1.0

# How many times a running job's lease is renewed per lease TTL: the renew interval is the
# TTL divided by this, so the lease is refreshed well before it expires. The interval is
# floored so a very short TTL cannot cause a busy loop.
_LEASE_RENEWALS_PER_TTL = 3.0
_MIN_LEASE_RENEW_INTERVAL = 0.2


class _LeaseRenewer:
    """Keeps a running job's lease alive until the job finishes.

    ``claim_job`` sets an initial lease when it moves a job to RUNNING. A job that runs
    longer than the lease TTL would otherwise look abandoned to stale-job recovery, so while
    the job runs this renews the lease in the background and stops as soon as the job returns.
    Used as a context manager around the blocking ``wait_for_job`` call.
    """

    def __init__(self, job_store: AbstractJobStore, job_id: str, lease_duration: float) -> None:
        self._job_store = job_store
        self._job_id = job_id
        self._lease_duration = lease_duration
        self._interval = max(lease_duration / _LEASE_RENEWALS_PER_TTL, _MIN_LEASE_RENEW_INTERVAL)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._renew_until_stopped,
            name=f"mlflow-job-lease-renewer-{job_id}",
            daemon=True,
        )

    def _renew_until_stopped(self) -> None:
        # ``Event.wait`` returns True once stopped and False on each timeout; renew on timeout.
        while not self._stop.wait(self._interval):
            try:
                status = self._job_store.renew_job_lease(self._job_id, self._lease_duration)
            except Exception:
                # A transient store error should not silently kill renewal: log and retry on
                # the next tick so the lease keeps being refreshed while the job runs.
                _logger.exception(f"Failed to renew lease for job {self._job_id}; will retry")
                continue
            if status != JobUpdateStatus.APPLIED:
                # The job row is no longer renewable (finalized or reset elsewhere); stop.
                return

    def __enter__(self) -> "_LeaseRenewer":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=self._interval + 5.0)


def _select_executor() -> AbstractJobExecutor:
    """Resolve the executor backend named by ``MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND``.

    For now this is always ``local`` (``LocalJobExecutor``).
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
    job_store: AbstractJobStore,
    executor: AbstractJobExecutor,
    job: Job,
    lease_duration: float | None,
) -> None:
    """Execute a job that has already been claimed (moved to RUNNING) and record its result.

    While the job runs, its lease is renewed in the background so a long-running job is not
    treated as abandoned by stale-job recovery.
    """
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
    # The lease is set at claim time; the renewer only covers wait_for_job. If submit_job does
    # long setup first (e.g. installing a job's environment), the lease can lapse before the
    # first renewal, but that renewal self-heals it since it only applies while the job is still
    # RUNNING. If that window ever matters, raise the initial lease TTL rather than move this.
    lease_renewer = (
        _LeaseRenewer(job_store, job.job_id, lease_duration)
        if lease_duration is not None
        else nullcontext()
    )
    with lease_renewer:
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
            for job in list(job_store.list_jobs(statuses=[JobStatus.PENDING])):
                if job_store.claim_job(job.job_id, lease_duration) != JobUpdateStatus.APPLIED:
                    # A concurrent worker claimed it first, or its status changed.
                    continue
                try:
                    _execute_claimed_job(job_store, executor, job, lease_duration)
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


def _recover_orphaned_jobs(job_store: AbstractJobStore) -> None:
    """Requeue jobs a previous runner left mid-flight before the poll loop starts.

    The loop only claims PENDING jobs, so a job that was RUNNING when the server stopped
    abruptly would otherwise stay RUNNING forever. On startup, reset such jobs back to
    PENDING so the loop re-claims them. A job whose lease is still valid is left alone: a
    live worker (for example another replica) may still be running it.
    """
    from mlflow.server.jobs.utils import _workspace_contexts_for_recovery

    now = get_current_time_millis()
    for workspace_ctx in _workspace_contexts_for_recovery():
        try:
            with workspace_ctx:
                unfinished = list(
                    job_store.list_jobs(statuses=[JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY])
                )
                for job in unfinished:
                    # This assumes a single active runner: recovery runs once at startup,
                    # before the poll loop, so no lease renewer is running at the same time and
                    # resetting an expired-lease job is safe. With multiple runners active at
                    # once, a live worker on another runner could renew this lease between the
                    # list_jobs call above and the reset_job call below, and the reset would then
                    # requeue a job that is still running. Fully closing that window needs a
                    # store-side conditional reset (reset only if the stored lease is still
                    # expired), which is left to the multi-runner coordination work.
                    lease_expired = job.lease_expires_at is None or job.lease_expires_at <= now
                    if job.status != JobStatus.NEEDS_RECOVERY and not lease_expired:
                        # A live worker still holds this job's lease; leave it running.
                        continue
                    _logger.info(
                        f"Recovering orphaned job {job.job_id} ({job.job_name}) in status "
                        f"{job.status.value}; resetting to PENDING"
                    )
                    try:
                        job_store.reset_job(job.job_id)
                    except Exception:
                        # The job may have left RUNNING/NEEDS_RECOVERY between listing and reset
                        # (e.g. another worker finalized it), which makes reset_job raise. Skip
                        # this job and keep recovering the rest.
                        _logger.exception(f"Failed to reset orphaned job {job.job_id}; skipping")
        except Exception:
            # One workspace failing to list or recover must not abort recovery for the others.
            _logger.exception("Stale-job recovery failed for a workspace; continuing.")


def run_executor_loop(
    stop_event: threading.Event | None = None, poll_interval: float = _POLL_INTERVAL
) -> None:
    """Run the executor job-execution loop until ``stop_event`` is set."""
    from mlflow.server.handlers import _get_job_store

    stop_event = stop_event or threading.Event()
    job_store = _get_job_store()
    executor = _select_executor()
    lease_duration = executor.config.job_lease_ttl
    executor.start_executor()
    _logger.info(
        f"Started executor-backed job runner (backend={MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND.get()})"
    )
    try:
        try:
            _recover_orphaned_jobs(job_store)
        except Exception:
            # Recovery is best-effort; a failure here must not stop the runner from starting.
            _logger.exception("Stale-job recovery failed at startup; continuing.")
        while not stop_event.is_set():
            try:
                executed = _poll_and_execute_once(job_store, executor, lease_duration)
            except Exception:
                # The loop-level store and workspace calls run outside the per-job handler.
                # A transient error here must not kill the runner (nothing would restart it),
                # so log it and try again on the next poll.
                _logger.exception("Executor job loop iteration failed; continuing.")
                executed = 0
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
