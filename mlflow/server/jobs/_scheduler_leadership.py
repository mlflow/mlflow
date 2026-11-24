import logging
import os
import queue
import random
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from mlflow.server.jobs.lock_manager import JobLockManager, SchedulerLease
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore

_logger = logging.getLogger(__name__)

_SCHEDULER_LEASE_KEY = "periodic-tasks-scheduler"
_SCHEDULER_LEASE_TTL_SECONDS = 30
_SCHEDULER_LEASE_RENEW_INTERVAL_SECONDS = 10.0
_SCHEDULER_FOLLOWER_RETRY_SECONDS = 5.0
_SCHEDULER_FOLLOWER_RETRY_JITTER_SECONDS = 1.0
_SCHEDULER_CHILD_POLL_INTERVAL_SECONDS = 1.0
_SCHEDULER_CHILD_STOP_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class _RenewalAttempt:
    lease: SchedulerLease | None
    started_at: float
    completed_at: float


class _SchedulerLeadershipCoordinator:
    """Runs the periodic-task consumer only while this process holds the scheduler lease."""

    def __init__(
        self,
        lock_manager: JobLockManager,
        start_consumer: Callable[[], subprocess.Popen],
        stop_event: threading.Event | None = None,
        lease_ttl_seconds: int = _SCHEDULER_LEASE_TTL_SECONDS,
        renew_interval_seconds: float = _SCHEDULER_LEASE_RENEW_INTERVAL_SECONDS,
        follower_retry_seconds: float = _SCHEDULER_FOLLOWER_RETRY_SECONDS,
        follower_retry_jitter_seconds: float = _SCHEDULER_FOLLOWER_RETRY_JITTER_SECONDS,
        child_poll_interval_seconds: float = _SCHEDULER_CHILD_POLL_INTERVAL_SECONDS,
    ) -> None:
        self._lock_manager = lock_manager
        self._start_consumer = start_consumer
        self._stop_event = stop_event or threading.Event()
        self._lease_ttl_seconds = lease_ttl_seconds
        self._renew_interval_seconds = renew_interval_seconds
        self._follower_retry_seconds = follower_retry_seconds
        self._follower_retry_jitter_seconds = follower_retry_jitter_seconds
        self._child_poll_interval_seconds = child_poll_interval_seconds
        self._consumer: subprocess.Popen | None = None

    def _stop_consumer(self) -> None:
        consumer = self._consumer
        self._consumer = None
        if consumer is None:
            return

        try:
            if consumer.poll() is None:
                try:
                    consumer.kill()
                except ProcessLookupError:
                    pass
            consumer.wait(timeout=_SCHEDULER_CHILD_STOP_TIMEOUT_SECONDS)
        except Exception:
            _logger.exception("Failed to stop the periodic-tasks consumer")

    def _wait_as_follower(self) -> None:
        delay = self._follower_retry_seconds + random.uniform(
            0, self._follower_retry_jitter_seconds
        )
        self._stop_event.wait(delay)

    def _try_acquire(self) -> tuple[SchedulerLease, float] | None:
        started_at = time.monotonic()
        try:
            lease = self._lock_manager.acquire_scheduler_lease(
                _SCHEDULER_LEASE_KEY,
                ttl_seconds=self._lease_ttl_seconds,
            )
        except Exception:
            _logger.exception("Failed to acquire periodic-task scheduler leadership; will retry")
            return None
        if lease is None:
            return None
        return lease, started_at + self._lease_ttl_seconds

    def _renew(self, lease: SchedulerLease) -> SchedulerLease | None:
        try:
            return self._lock_manager.renew_scheduler_lease(
                lease,
                ttl_seconds=self._lease_ttl_seconds,
            )
        except Exception:
            # A database error makes ownership uncertain. Stop leader-only activity rather than
            # continuing while another replica may acquire the expired lease.
            _logger.exception("Failed to renew periodic-task scheduler leadership")
            return None

    def _renew_in_background(
        self,
        lease: SchedulerLease,
        started_at: float,
        result_queue: queue.Queue[_RenewalAttempt],
    ) -> None:
        renewed_lease = self._renew(lease)
        result_queue.put(
            _RenewalAttempt(
                lease=renewed_lease,
                started_at=started_at,
                completed_at=time.monotonic(),
            )
        )

    def _start_or_restart_consumer(self) -> None:
        if self._consumer is not None and self._consumer.poll() is None:
            return
        if self._consumer is not None:
            _logger.warning(
                "Periodic-tasks consumer exited with code %s; restarting while leadership is held",
                self._consumer.returncode,
            )
        try:
            self._consumer = self._start_consumer()
        except Exception:
            self._consumer = None
            _logger.exception("Failed to start the periodic-tasks consumer; will retry")

    def _run_as_leader(self, lease: SchedulerLease, lease_deadline: float | None = None) -> None:
        if lease_deadline is None:
            lease_deadline = time.monotonic() + self._lease_ttl_seconds
        next_renewal = min(
            time.monotonic() + self._renew_interval_seconds,
            lease_deadline,
        )
        renewal_result: queue.Queue[_RenewalAttempt] | None = None
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                if renewal_result is not None:
                    try:
                        attempt = renewal_result.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        renewal_result = None
                        if attempt.lease is None or attempt.completed_at >= lease_deadline:
                            _logger.warning(
                                "Lost periodic-task scheduler leadership; stopping the "
                                "periodic-tasks consumer"
                            )
                            return
                        lease = attempt.lease
                        lease_deadline = attempt.started_at + self._lease_ttl_seconds
                        next_renewal = min(
                            time.monotonic() + self._renew_interval_seconds,
                            lease_deadline,
                        )

                now = time.monotonic()
                if now >= lease_deadline:
                    if renewal_result is not None:
                        _logger.warning(
                            "Periodic-task scheduler lease expired while renewal was in progress; "
                            "stopping the periodic-tasks consumer"
                        )
                    return

                if renewal_result is None and now >= next_renewal:
                    renewal_result = queue.Queue(maxsize=1)
                    renewal_started_at = time.monotonic()
                    threading.Thread(
                        target=self._renew_in_background,
                        args=(lease, renewal_started_at, renewal_result),
                        name="MLflow-periodic-tasks-lease-renewer",
                        daemon=True,
                    ).start()

                self._start_or_restart_consumer()
                wake_at = lease_deadline if renewal_result is not None else next_renewal
                wait_seconds = min(self._child_poll_interval_seconds, max(0.0, wake_at - now))
                if self._stop_event.wait(wait_seconds):
                    return
        finally:
            self._stop_consumer()

    def run(self) -> None:
        while not self._stop_event.is_set():
            acquired = self._try_acquire()
            if acquired is None:
                self._wait_as_follower()
                continue
            lease, lease_deadline = acquired

            _logger.info(
                "Acquired periodic-task scheduler leadership (lease acquired_at=%s, process_id=%s)",
                lease.acquired_at,
                os.getpid(),
            )
            self._run_as_leader(lease, lease_deadline)


def _run_periodic_tasks_consumer_with_leadership(
    start_consumer: Callable[[], subprocess.Popen],
    stop_event: threading.Event | None = None,
) -> None:
    from mlflow.server.handlers import _get_job_store

    job_store = cast(SqlAlchemyJobStore, _get_job_store())
    coordinator = _SchedulerLeadershipCoordinator(
        JobLockManager(job_store),
        start_consumer,
        stop_event=stop_event,
    )
    coordinator.run()
