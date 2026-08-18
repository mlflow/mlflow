"""Lock coordination for multi-replica job execution."""

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import TEMPORARILY_UNAVAILABLE, ErrorCode
from mlflow.store.db.utils import get_current_time_millis_expression
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.tracking.dbmodels.models import SqlJob, SqlJobLock, SqlSchedulerLease
from mlflow.utils.time import get_current_time_millis

_logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")
_JOB_LOCK_ACQUISITION_MAX_DEADLOCK_RETRIES = 2
_SUPPORTED_DRIVERS = {"pymysql", "psycopg2", "pymssql", "pysqlite"}


def _check_time_drift_and_log(app_now_millis: int, db_now_millis: int) -> None:
    """
    Compare the application clock with the database clock and log a warning if the difference is
    more than 1 second.

    Args:
        app_now_millis: Current application clock time in milliseconds since the Unix epoch.
        db_now_millis: Current database clock time in milliseconds since the Unix epoch.
    """

    drift = app_now_millis - db_now_millis
    if abs(drift) > 1000:
        _logger.warning("Time drift > 1s detected APP_time - DB_time = %d ms", drift)


@dataclass(frozen=True)
class SchedulerLease:
    lease_key: str
    acquired_at: int
    ttl_seconds: int


class JobLockManager:
    """
    Manages scheduler lease and job lock coordination for multi-replica
    MLflow deployments.

    Uses the ``scheduler_leases`` and ``job_locks`` tables with a
    SQLAlchemy session factory from a ``SqlAlchemyJobStore``.

    Example usage::

        lock_mgr = JobLockManager(job_store)

        # Acquire scheduler lease
        scheduler_lease = lock_mgr.acquire_scheduler_lease("scheduler", ttl_seconds=90)
        if scheduler_lease is not None:
            # This replica holds the scheduler lease.
            ...

            # Renew scheduler lease
            renewed_lease = lock_mgr.renew_scheduler_lease(scheduler_lease, ttl_seconds=90)
            if renewed_lease is not None:
                # This replica still holds the scheduler lease.
                ...

        # Acquire exclusive lock
        if lock_mgr.acquire_exclusive_lock("exp-123:hash", "job-456"):
            try:
                # ... do work ...
    """

    def __init__(self, job_store: SqlAlchemyJobStore):

        self.db_type = job_store.db_type
        self._session_maker = job_store.ManagedSessionMaker

    def _guard_insert_race(
        self, fn: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
    ) -> _R | None:
        """
        Call ``fn`` and return ``None`` if a unique constraint violation occurs.

        MySQL can raise a deadlock ``OperationalError`` instead of an ``IntegrityError`` when
        two replicas try to insert at the same time. This method tries deadlocks again with
        exponential backoff, up to ``_JOB_LOCK_ACQUISITION_MAX_DEADLOCK_RETRIES`` times.

        Args:
            fn: The function to call.
            *args: Positional arguments for ``fn``.
            **kwargs: Keyword arguments for ``fn``.

        Returns:
            The return value of ``fn``, or ``None`` on a unique constraint violation.
        """

        deadlock_error_code = ErrorCode.Name(TEMPORARILY_UNAVAILABLE)
        for attempt in range(_JOB_LOCK_ACQUISITION_MAX_DEADLOCK_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except MlflowException as e:
                # ManagedSessionMaker wraps all SQLAlchemy exceptions in MlflowException.
                # An IntegrityError can occur at startup when two replicas both find no
                # existing row and race to insert the same lease key.
                if isinstance(e.__cause__, IntegrityError):
                    _logger.debug(
                        "Lease acquisition denied. A concurrent insert conflict occurred."
                    )
                    return None

                # MySQL may throw an OperationalError when replicas race to insert a new lock
                # instead of an IntegrityError. Raise if error is not a deadlock else retry.
                is_deadlock = e.error_code == deadlock_error_code and "deadlock" in str(e).lower()
                if not is_deadlock or attempt >= _JOB_LOCK_ACQUISITION_MAX_DEADLOCK_RETRIES:
                    raise

                # Exponential backoff with jitter, matching `_try_insert_tags`.
                sleep_duration = (2**attempt) - 1
                sleep_duration += random.uniform(0, 1)
                time.sleep(sleep_duration)

    def renew_scheduler_lease(
        self, scheduler_lease: SchedulerLease, ttl_seconds: int
    ) -> SchedulerLease | None:
        """
        Renew an existing scheduler lease. Lease renewal is best effort.

        This method matches all three fields of ``scheduler_lease`` against the
        database row to verify that the caller holds the lease.

        If the row matches, the method renews the lease and returns a new
        ``SchedulerLease``. The renewal succeeds whether the lease is active
        or expired, as long as no other replica has acquired it.

        If the row does not match, the method returns ``None``. The caller
        no longer holds the lease.

        .. important::
            Callers must use the returned ``SchedulerLease`` for all subsequent
            operations. The original object becomes invalid after renewal because
            ``acquired_at`` and ``ttl_seconds`` change.

            Lease renewal is best effort. In the case of sub-millisecond lease
            renewal requests with the same ttl, there is no "renewal" and the
            retuned SchedulerLease will have the same values as the original
            lease.

        Args:
            scheduler_lease: The current lease that proves the caller holds it.
            ttl_seconds: Time-to-live for the renewed lease, in seconds.
                Must be greater than zero.

        Returns:
            A new ``SchedulerLease`` on success. ``None`` if the caller no
            longer holds the lease.

        Raises:
            MlflowException: If ``ttl_seconds`` is zero or negative.
        """

        if ttl_seconds <= 0:
            raise MlflowException.invalid_parameter_value(
                f"ttl_seconds must be greater than zero, got {ttl_seconds}"
            )

        with self._session_maker(read_only=False) as session:
            app_now_millis = get_current_time_millis()
            filter_args = (
                SqlSchedulerLease.lease_key == scheduler_lease.lease_key,
                SqlSchedulerLease.acquired_at == scheduler_lease.acquired_at,
                SqlSchedulerLease.ttl_seconds == scheduler_lease.ttl_seconds,
            )

            update_values = {
                SqlSchedulerLease.acquired_at: get_current_time_millis_expression(self.db_type),
                SqlSchedulerLease.ttl_seconds: ttl_seconds,
            }

            rows_updated = (
                session
                .query(SqlSchedulerLease)
                .filter(*filter_args)
                .update(update_values, synchronize_session=False)
            )

            if rows_updated > 0:
                renewed_lease = (
                    session
                    .query(SqlSchedulerLease)
                    .filter(SqlSchedulerLease.lease_key == scheduler_lease.lease_key)
                    .one()
                )
                _check_time_drift_and_log(app_now_millis, renewed_lease.acquired_at)

                return SchedulerLease(
                    lease_key=renewed_lease.lease_key,
                    acquired_at=renewed_lease.acquired_at,
                    ttl_seconds=renewed_lease.ttl_seconds,
                )

            _logger.debug("Scheduler lease renewal denied, lease no longer valid.")
            return None

    def _acquire_scheduler_lease(self, lease_key: str, ttl_seconds: int) -> SchedulerLease | None:

        with self._session_maker(read_only=False) as session:
            app_now_millis = get_current_time_millis()
            db_now = get_current_time_millis_expression(self.db_type)
            filter_args = (
                SqlSchedulerLease.lease_key == lease_key,
                SqlSchedulerLease.acquired_at + (SqlSchedulerLease.ttl_seconds * 1000) <= db_now,
            )

            update_values = {
                SqlSchedulerLease.acquired_at: db_now,
                SqlSchedulerLease.ttl_seconds: ttl_seconds,
            }

            rows_updated = (
                session
                .query(SqlSchedulerLease)
                .filter(*filter_args)
                .update(update_values, synchronize_session=False)
            )

            if rows_updated > 0:
                new_lease = (
                    session
                    .query(SqlSchedulerLease)
                    .filter(SqlSchedulerLease.lease_key == lease_key)
                    .one()
                )
                _check_time_drift_and_log(app_now_millis, new_lease.acquired_at)

                return SchedulerLease(
                    lease_key=new_lease.lease_key,
                    acquired_at=new_lease.acquired_at,
                    ttl_seconds=new_lease.ttl_seconds,
                )

            existing = (
                session
                .query(SqlSchedulerLease)
                .filter(SqlSchedulerLease.lease_key == lease_key)
                .one_or_none()
            )

            if existing is None:
                insert_values = {SqlSchedulerLease.lease_key: lease_key, **update_values}
                session.execute(insert(SqlSchedulerLease).values(insert_values))

                new_lease = (
                    session
                    .query(SqlSchedulerLease)
                    .filter(SqlSchedulerLease.lease_key == lease_key)
                    .one()
                )
                _check_time_drift_and_log(app_now_millis, new_lease.acquired_at)

                return SchedulerLease(
                    lease_key=new_lease.lease_key,
                    acquired_at=new_lease.acquired_at,
                    ttl_seconds=new_lease.ttl_seconds,
                )

        _logger.debug("Scheduler lease denied, a valid lease exists.")
        return None

    def acquire_scheduler_lease(self, lease_key: str, ttl_seconds: int) -> SchedulerLease | None:
        """
        Acquire the scheduler lease for this replica.

        The method uses ``lease_key`` to identify the lease in the database.

        If no lease exists, or the existing lease has expired, this method acquires
        a new lease and returns a ``SchedulerLease``.

        If a valid lease already exists, this method returns ``None`` without acquiring
        a lease.

        Args:
            lease_key: The key that identifies this scheduler lease in the database.
            ttl_seconds: The duration of the lease in seconds. Must be greater than zero.

        Returns:
            SchedulerLease if this replica now holds the lease. None otherwise.

        Raises:
            MlflowException: Invalid ttl_seconds value.
        """

        if ttl_seconds <= 0:
            raise MlflowException.invalid_parameter_value(
                f"ttl_seconds must be greater than zero, got {ttl_seconds}"
            )

        return self._guard_insert_race(
            fn=self._acquire_scheduler_lease,
            lease_key=lease_key,
            ttl_seconds=ttl_seconds,
        )

    def acquire_exclusive_lock(self, lock_key: str, job_id: str) -> bool:
        """
        Creates an exclusive job lock from ``lock_key`` and ``job_id``.

        Returns True if no lock exists or the existing lock is stale.

        Returns False if a valid lock already exists.

        NOTE:
            This method cleans up stale locks only when a new acquisition
            targets the same lock_key. It does not remove stale rows for
            other keys. Each lock_key persists as a unique row in the
            ``job_locks`` table until it is manually deleted.

        Args:
            lock_key: Framework-computed key
            job_id: The job ID trying to acquire this lock. Must belong
                to an existing ``SqlJob``

        Returns:
            True if lock acquired, False if held by another live job.

        Raises:
            MlflowException: A valid lock already exists for the job_id.
        """
        try:
            with self._session_maker(read_only=False) as session:
                existing_lock = (
                    session
                    .query(SqlJobLock)
                    .filter(SqlJobLock.lock_key == lock_key)
                    .with_for_update()
                    .one_or_none()
                )

                if existing_lock is not None:
                    holding_job = (
                        session
                        .query(SqlJob)
                        .filter(SqlJob.id == existing_lock.job_id)
                        .one_or_none()
                    )

                    job_lock_is_valid = self._is_job_lock_valid(existing_lock, holding_job)

                    # Raise an exception when the requesting job_id already holds a valid lock.
                    # A False return could cause the caller to mark the active job as CANCELED.
                    # The job row claim logic prevents duplicate ownership, but a replica could
                    # still retry a lock it already holds. A stale lock held by the same job_id
                    # can still be evicted and re-acquired.
                    if job_lock_is_valid and job_id == existing_lock.job_id:
                        raise MlflowException.invalid_parameter_value(
                            "A valid lock already exists for this job_id"
                        )

                    if job_lock_is_valid:
                        _logger.debug("Job lock acquisition denied. A valid lock exists.")
                        return False

                    session.delete(existing_lock)
                    session.flush()

                session.add(
                    SqlJobLock(
                        lock_key=lock_key,
                        job_id=job_id,
                        acquired_at=get_current_time_millis(),
                    )
                )

            return True

        except MlflowException as e:
            holding_job_id = None
            with self._session_maker(read_only=True) as session:
                existing_lock = (
                    session.query(SqlJobLock).filter(SqlJobLock.lock_key == lock_key).one_or_none()
                )

                if existing_lock is not None:
                    holding_job_id = existing_lock.job_id

            # Check that the IntegrityError is from two different jobs that tried to
            # acquire the same lock.
            valid_integrity_error = (
                isinstance(e.__cause__, IntegrityError)
                and holding_job_id is not None
                and holding_job_id != job_id
            )
            if valid_integrity_error:
                _logger.debug("Job lock acquisition denied. A concurrent insert conflict occurred.")
                return False

            if isinstance(e.__cause__, IntegrityError):
                _logger.error("An unexpected IntegrityError occurred during job lock acquisition")

            raise

    @staticmethod
    def _is_job_lock_valid(lock: SqlJobLock, holding_job: SqlJob | None) -> bool:
        """
        Check if a lock is still valid.

        Args:
            lock: The ``SqlJobLock`` to validate
            holding_job: The job currently holding the lock

        Returns:
            True if the job lock is still valid. False otherwise.

        Raises:
            MlflowException: ``lock.job_id`` must match ``holding_job.id``
        """

        if holding_job is None:
            return False

        if holding_job.id != lock.job_id:
            raise MlflowException.invalid_parameter_value(
                f"Lock is not held by SqlJob {lock.job_id=} != {holding_job.id=}"
            )

        if JobStatus.is_finalized(JobStatus.from_int(holding_job.status)):
            return False

        now = get_current_time_millis()
        if holding_job.timeout is not None:
            expiration_time = lock.acquired_at + int(holding_job.timeout * 1.15 * 1000)
            if now > expiration_time:
                return False

        return True
