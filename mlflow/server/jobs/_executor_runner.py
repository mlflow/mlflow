"""Executor-engine job runner.

Launched in place of the Huey ``_job_runner`` when
``MLFLOW_SERVER_JOB_EXECUTION_ENGINE=executor``. A scheduler loop claims PENDING jobs from the
job store and runs each in its own worker thread through the configured ``AbstractJobExecutor``
backend (``LocalJobExecutor`` by default), then records the terminal state back to the store.

Concurrency is bounded per job function by a semaphore sized to that function's ``max_workers``,
so one job type cannot starve another and the scheduler thread never blocks on a running job.
Each tick also forwards store-side cancellations to the executor for in-flight jobs. On startup the
runner recovers jobs left unfinished by a previous server generation.

Only job execution moves to the executor framework here. Periodic tasks (e.g. the online scoring
scheduler) still run on Huey via ``_launch_periodic_tasks_consumer``.
"""

import json
import logging
import os
import random
import signal
import threading
import time
from dataclasses import dataclass
from typing import Callable

from mlflow.entities._job import Job
from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import (
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_GATEWAY_URI,
    MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND,
    MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY,
    MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import TEMPORARILY_UNAVAILABLE, ErrorCode
from mlflow.server.constants import BACKEND_STORE_URI_ENV_VAR, MLFLOW_SERVER_UP_TIME
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
    # Jobs get the backend store URI (the DB), not MLFLOW_TRACKING_URI. The runner is launched with
    # MLFLOW_TRACKING_URI set to the server's own HTTP URI, but a job can't authenticate to the
    # tracking API over HTTP (its only credential, the internal token, is gateway-only), and jobs
    # do privileged work directly against the store anyway — scorer jobs already call
    # _get_tracking_store(), which flips tracking to the DB. Gateway routing still needs the HTTP
    # URI, so it travels separately as MLFLOW_GATEWAY_URI.
    return JobExecutionContext(
        job_id=job.job_id,
        tracking_uri=os.environ.get(BACKEND_STORE_URI_ENV_VAR),
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
        # The executor reported the job as canceled while the store row is not CANCELED (a row
        # canceled through the store is already handled by the early return above). Record the
        # terminal CANCELED state so the claimed row is not left RUNNING forever. Tolerate a
        # concurrent store cancel that finalized the row between the check above and here.
        try:
            job_store.report_job_result(job_id, JobStatus.CANCELED)
        except MlflowException:
            if job_store.get_job(job_id).status != JobStatus.CANCELED:
                raise
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
    on_submitted: Callable[[], None] | None = None,
) -> None:
    """Execute a job that has already been claimed (moved to RUNNING) and record its result.

    ``on_submitted`` is invoked right after the job reaches the executor, so the scheduler can
    start forwarding cancellations only once there is a backend job to cancel.
    """
    from mlflow.server.jobs.utils import _load_function, get_job_fn_fullname

    fn_fullname = get_job_fn_fullname(job.job_name)
    function = _load_function(fn_fullname)
    python_env = function._job_fn_metadata.python_env
    params = json.loads(job.params)
    context = _build_execution_context(job)

    # A cancellation that arrived after the claim but before submission can't be forwarded to the
    # executor (there is no backend job yet). Check the store here so such a job is not started at
    # all; the row is already terminal CANCELED, so there is nothing to record.
    if job_store.get_job(job.job_id).status == JobStatus.CANCELED:
        return

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
    if on_submitted is not None:
        on_submitted()
    result = executor.wait_for_job(job.job_id)
    _logger.info(f"Executor engine job {job.job_id} finished with status {result.status.value}")
    _record_result(job_store, job.job_id, job.job_name, result)


def _max_workers_for(job_name: str) -> int:
    """Concurrency limit for a job function, from its ``@job(max_workers=...)`` metadata."""
    from mlflow.server.jobs.utils import _load_function, get_job_fn_fullname

    max_workers = _load_function(get_job_fn_fullname(job_name))._job_fn_metadata.max_workers
    return max(1, max_workers or 1)


@dataclass
class _InFlightJob:
    """Bookkeeping for a claimed job while its worker thread runs."""

    workspace: str | None
    thread: threading.Thread
    # Set once the job has reached the executor. Cancellation is forwarded only after this, since
    # there is no backend job to cancel before submission.
    submitted: bool = False
    # Set once a store-side cancellation has been successfully forwarded to the executor, so it is
    # forwarded at most once (AbstractJobExecutor.cancel_job is not required to be idempotent).
    cancel_forwarded: bool = False


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
        # job_id -> _InFlightJob, for the cancel sweep and shutdown join. Guarded by
        # _in_flight_lock.
        self._in_flight: dict[str, _InFlightJob] = {}
        self._in_flight_lock = threading.Lock()

    def _slot_for(self, job_name: str) -> threading.Semaphore:
        with self._slots_lock:
            sem = self._slots.get(job_name)
        if sem is not None:
            return sem
        # Resolve max_workers (which may import the job module) before taking the lock, so the
        # import runs outside the lock's critical section. This is a no-op under the single
        # scheduler thread today, but keeps the hold time minimal if the scheduler ever runs
        # multi-threaded.
        max_workers = _max_workers_for(job_name)
        with self._slots_lock:
            return self._slots.setdefault(job_name, threading.Semaphore(max_workers))

    def _mark_submitted(self, job_id: str) -> None:
        with self._in_flight_lock:
            if (handle := self._in_flight.get(job_id)) is not None:
                handle.submitted = True

    def tick(self) -> int:
        """Run one scheduler iteration. Returns the number of jobs newly scheduled."""
        self._forward_cancellations()
        return self._schedule_pending()

    def _forward_cancellations(self) -> None:
        with self._in_flight_lock:
            snapshot = list(self._in_flight.items())
        for job_id, handle in snapshot:
            # Only forward once the job has reached the executor and not already forwarded. Before
            # submission there is no backend job to cancel; the worker's own pre-submit check
            # (see _execute_claimed_job) handles a cancel that arrives in that window.
            if not handle.submitted or handle.cancel_forwarded:
                continue
            try:
                with ServerWorkspaceContext(handle.workspace):
                    canceled = self._job_store.get_job(job_id).status == JobStatus.CANCELED
            except Exception:
                _logger.exception("Failed to check cancellation state for job %s", job_id)
                continue
            if not canceled:
                continue
            try:
                # Stop the still-running job so it cannot keep performing side effects after the
                # caller cancelled it. The worker's wait_for_job then returns and _record_result
                # skips the already-CANCELED row.
                self._executor.cancel_job(job_id)
            except Exception:
                # Leave cancel_forwarded false so the next tick retries rather than letting the
                # canceled job run to normal completion. Log every failure: a repeated failure may
                # differ each time, so silencing later ones could hide something useful.
                _logger.exception(
                    "Failed to forward cancellation to executor for %s; retrying next tick", job_id
                )
                continue
            with self._in_flight_lock:
                # cancel_job may already have unblocked the worker, whose finally pops _in_flight;
                # only record on the live handle so a finished job's id is not left behind.
                if (live := self._in_flight.get(job_id)) is not None:
                    live.cancel_forwarded = True

    def _schedule_pending(self) -> int:
        from mlflow.server.jobs.utils import _workspace_contexts_for_recovery

        # Randomize the per-workspace order so a fixed (e.g. alphabetical) order can't let the
        # earliest workspaces consistently claim the max_workers slots and starve the rest.
        #
        # TODO: with workspaces enabled, _workspace_contexts_for_recovery() returns every defined
        # workspace, and the loop below runs one list_jobs(status=PENDING) query per workspace on
        # every tick (~1s) even when most have nothing pending — cost scales with total workspaces,
        # not active ones. To make it scale with active workspaces instead:
        #   1. add an AbstractJobStore method that returns, in a single cross-workspace query, the
        #      distinct workspaces that currently have PENDING jobs (the workspace-aware store
        #      filters by the active workspace today, so this query must not be workspace-scoped);
        #   2. have the scheduler iterate only those workspaces here instead of all of them;
        #   3. cover it with workspace-aware tests (see the jobs store's workspace test module).
        # Left as a follow-up since the per-tick cost only becomes material at multi-tenant scale.
        workspace_contexts = list(_workspace_contexts_for_recovery())
        random.shuffle(workspace_contexts)
        scheduled = 0
        for workspace_ctx in workspace_contexts:
            with workspace_ctx as workspace:
                # TODO (follow-up): exclusive job-locking by key (e.g. keeping online scoring
                # exclusive per experiment) is not enforced here; it builds on the job-lock work
                # in #25200 and lands once that is available.
                for job in list(self._job_store.list_jobs(statuses=[JobStatus.PENDING])):
                    with self._in_flight_lock:
                        already_in_flight = job.job_id in self._in_flight
                    if already_in_flight:
                        # A prior worker for this job is still finishing (e.g. backing off before
                        # it re-pends a transient failure). Skip it so the job stays unclaimable
                        # until that worker releases it, otherwise a free slot would re-claim it
                        # mid-backoff and bypass the retry delay.
                        continue
                    try:
                        sem = self._slot_for(job.job_name)
                    except Exception as exc:
                        # The function backing this job cannot be resolved (e.g. it was renamed or
                        # removed across an upgrade). Fail it, matching the old serial path, so it
                        # reaches a terminal state instead of staying PENDING and re-logging on
                        # every tick.
                        self._fail_unschedulable_job(job, workspace, exc)
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
                    try:
                        started = self._start_worker(job, workspace, sem)
                    except Exception:
                        # _start_worker owns releasing the slot on the failure paths it handles;
                        # this guards the unexpected case so the slot is not leaked if it raises.
                        sem.release()
                        _logger.exception("Failed to start worker for job %s", job.job_id)
                        continue
                    if started:
                        scheduled += 1
        return scheduled

    def _fail_claimed_job(self, job_id: str, workspace: str | None, error: str) -> None:
        """Transition a claimed (RUNNING) job to FAILED, retrying once on a transient store error.

        A claimed job left RUNNING with no worker is only reclaimable by startup recovery, so a
        transient store failure on the first attempt should not strand it there. The managed
        session surfaces a transient DB error as ``MlflowException`` with
        ``error_code == TEMPORARILY_UNAVAILABLE``; that is retried once. Anything else — an invalid
        transition (the row is no longer RUNNING) or any other error — won't succeed on a retry, so
        it is logged once and not retried.
        """
        temporarily_unavailable = ErrorCode.Name(TEMPORARILY_UNAVAILABLE)
        for attempt in (1, 2):
            try:
                with ServerWorkspaceContext(workspace):
                    self._job_store.fail_job(job_id, error)
                return
            except MlflowException as e:
                if attempt == 1 and e.error_code == temporarily_unavailable:
                    _logger.warning("Transient store error failing job %s; retrying once", job_id)
                    continue
                _logger.exception("Could not transition job %s to FAILED", job_id)
                return
            except Exception:
                # The store wraps its errors as MlflowException; a bare exception here is
                # unexpected and not something a retry would fix, so log once and stop.
                _logger.exception("Unexpected error transitioning job %s to FAILED", job_id)
                return

    def _fail_unschedulable_job(self, job: Job, workspace: str | None, exc: Exception) -> None:
        """Claim and fail a PENDING job whose function cannot be resolved.

        ``workspace`` is the active workspace the claim runs under, so the fail targets the same
        workspace scope as the claim.
        """
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
        self._fail_claimed_job(job.job_id, workspace, repr(exc))

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
            self._in_flight[job.job_id] = _InFlightJob(workspace=workspace, thread=thread)
        try:
            thread.start()
        except Exception:
            _logger.exception("Failed to start worker thread for job %s", job.job_id)
            with self._in_flight_lock:
                self._in_flight.pop(job.job_id, None)
            sem.release()
            self._fail_claimed_job(job.job_id, workspace, "Failed to start executor worker thread")
            return False
        return True

    def _run_worker(self, job: Job, workspace: str | None, sem: threading.Semaphore) -> None:
        try:
            with ServerWorkspaceContext(workspace):
                _execute_claimed_job(
                    self._job_store,
                    self._executor,
                    job,
                    on_submitted=lambda: self._mark_submitted(job.job_id),
                )
        except Exception as exc:
            # A claimed job left RUNNING would be stuck; fail it so recovery is not needed.
            _logger.error(
                "Job %s (%s) raised in the executor worker: %r",
                job.job_id,
                job.job_name,
                exc,
                exc_info=True,
            )
            self._fail_claimed_job(job.job_id, workspace, repr(exc))
        finally:
            sem.release()
            with self._in_flight_lock:
                self._in_flight.pop(job.job_id, None)

    def join(self, timeout: float | None = None) -> None:
        """Best-effort wait for in-flight workers to finish.

        ``timeout`` is a total budget across all workers, not per worker, so shutdown is bounded.
        """
        with self._in_flight_lock:
            threads = [handle.thread for handle in self._in_flight.values()]
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

    def mark_orphans_for_recovery(self) -> None:
        """Flag any still-in-flight jobs as needing recovery on shutdown.

        Workers are daemon threads: if any are still running when the process exits, their
        ``finally`` blocks may not run, leaving the store row RUNNING with nothing executing it.
        Marking those rows NEEDS_RECOVERY lets startup recovery re-queue them on the next launch.
        """
        with self._in_flight_lock:
            orphans = [(job_id, handle.workspace) for job_id, handle in self._in_flight.items()]
        if not orphans:
            return
        _logger.warning(
            "Shutdown cut off %d still-running job(s) before completion; marking them for "
            "recovery on the next launch: %s",
            len(orphans),
            ", ".join(job_id for job_id, _ in orphans),
        )
        for job_id, workspace in orphans:
            try:
                with ServerWorkspaceContext(workspace):
                    self._job_store.mark_job_needs_recovery(job_id)
            except Exception:
                _logger.exception("Failed to mark orphaned job %s for recovery", job_id)


def run_executor_loop(
    executor: AbstractJobExecutor,
    stop_event: threading.Event | None = None,
    poll_interval: float = _POLL_INTERVAL,
) -> None:
    """Run the executor claim loop until ``stop_event`` is set.

    ``executor`` must already be started; the caller (``main``) owns ``start_executor()`` /
    ``stop_executor()``. Keeping lifecycle out of this loop lets the caller start and stop several
    configured backends independently, once more than one backend can be configured at a time,
    rather than tying it to this single-executor claim loop.
    """
    from mlflow.server.handlers import _get_job_store

    stop_event = stop_event or threading.Event()
    job_store = _get_job_store()
    lease_duration = executor.config.job_lease_ttl
    # Reclaim jobs left RUNNING/NEEDS_RECOVERY by a previous server generation (a crash, or a
    # shutdown that killed daemon workers) so they are re-scheduled instead of stuck. A transient
    # store error here must not crash the runner (same reasoning as the tick loop below): log it
    # and proceed to claim PENDING jobs; the next launch's recovery retries the reset.
    try:
        _recover_orphaned_executor_jobs(job_store)
    except Exception:
        _logger.exception("Executor job recovery failed at startup; continuing.")
    scheduler = _JobScheduler(job_store, executor, lease_duration)
    try:
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
        scheduler.join(timeout=_SHUTDOWN_JOIN_TIMEOUT)
        # Any worker still in flight after the join budget won't get to finalize its row on a
        # daemon-thread exit; flag it so the next launch's recovery re-queues it.
        scheduler.mark_orphans_for_recovery()


def _recover_orphaned_executor_jobs(job_store: AbstractJobStore) -> None:
    """Reset jobs left unfinished by a previous server generation back to PENDING.

    Only jobs created before this server launch are touched, so it never races jobs submitted to
    the running server. Unlike the Huey recovery path there is nothing to re-enqueue: the scheduler
    claims PENDING jobs on its next tick.

    This is the single-instance crash/restart recovery. Reclaiming a job whose lease expires while
    the runner is still live (a wedged worker, or another replica's jobs) is a follow-up that the
    lease primitive exists to enable once there is more than one owner.
    """
    from mlflow.server.jobs.utils import _for_each_unfinished_job

    server_up_time = os.environ.get(MLFLOW_SERVER_UP_TIME)
    if server_up_time is None:
        # Set by the server that launches this runner; without it we cannot bound recovery to the
        # previous generation, so skip rather than risk resetting freshly submitted jobs.
        _logger.debug("%s is unset; skipping executor job recovery.", MLFLOW_SERVER_UP_TIME)
        return
    try:
        launch_ts = int(server_up_time)
    except ValueError:
        _logger.warning(
            "%s is not an integer (%r); skipping executor job recovery.",
            MLFLOW_SERVER_UP_TIME,
            server_up_time,
        )
        return

    def _reset(job: Job, _workspace: str | None) -> None:
        try:
            job_store.reset_job(job.job_id)
            _logger.info("Recovered orphaned job %s (%s) to PENDING", job.job_id, job.job_name)
        except Exception:
            _logger.exception("Failed to recover orphaned job %s", job.job_id)

    _for_each_unfinished_job(
        job_store, [JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY], launch_ts, _reset
    )


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

    # Own the executor lifecycle here rather than inside the claim loop, so that when more than
    # one backend can be configured at a time (e.g. a custom-scorer backend alongside the default)
    # main() can start and stop each configured executor independently. Resolving and starting the
    # executor both run inside the try so any failure there (the runner re-validates the backend
    # registry independently of the parent) still triggers cleanup and the SIGTERM below.
    executor: AbstractJobExecutor | None = None
    fatal_exit = False
    try:
        executor = _select_executor()
        executor.start_executor()
        _logger.info(
            "Started executor-backed job runner "
            f"(backend={MLFLOW_JOB_DEFAULT_EXECUTOR_BACKEND.get()})"
        )
        run_executor_loop(executor)
    except Exception:
        # (KeyboardInterrupt/SystemExit are intentionally not caught: a clean shutdown signal
        # should propagate normally rather than be reported as an unexpected exit.)
        _logger.exception("Executor job runner exited unexpectedly; terminating process.")
        fatal_exit = True
    finally:
        # Guard stop_executor so a failure here (e.g. a half-initialized backend when
        # start_executor raised) cannot skip the SIGTERM below — that signal is the only thing
        # that tears the process down, since the periodic-tasks consumer thread is non-daemon.
        if executor is not None:
            try:
                executor.stop_executor()
            except Exception:
                _logger.exception("Failed to stop executor during shutdown.")
    if fatal_exit:
        # A non-daemon thread (the periodic-tasks consumer) keeps this process alive, so an
        # unhandled exit from the loop — e.g. a startup failure — would otherwise leave a live
        # process with no executor while jobs pile up PENDING. Terminate so the failure is visible.
        # Sent after stop_executor() above so backend cleanup is not skipped by the signal.
        os.kill(os.getpid(), signal.SIGTERM)


if __name__ == "__main__":
    main()
