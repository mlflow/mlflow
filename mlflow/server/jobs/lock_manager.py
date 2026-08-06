"""Lock coordination for multi-replica job execution."""

import logging

from sqlalchemy.exc import IntegrityError

from mlflow.exceptions import MlflowException
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.tracking.dbmodels.models import SqlSchedulerLease
from mlflow.utils.time import get_current_time_millis

_logger = logging.getLogger(__name__)


class JobLockManager:
    """
    Manages scheduler lease and job lock coordination for multi-replica
    MLflow deployments.

    Uses the ``scheduler_leases`` and ``job_locks`` tables through a
    SQLAlchemy session factory from a ``SqlAlchemyJobStore``.

    Example usage::

        lock_mgr = JobLockManager(job_store)

        # Acquire scheduler lease
        if lock_mgr.acquire_scheduler_lease("scheduler", ttl_seconds=90):
            # This replica holds the scheduler lease.
            ...
    """

    def __init__(self, job_store: SqlAlchemyJobStore):
        self._session_maker = job_store.ManagedSessionMaker

    def acquire_scheduler_lease(self, lease_key: str, ttl_seconds: int) -> bool:
        """
        Acquire the scheduler lease for this replica.

        The method uses ``lease_key`` to identify the lease in the database.

        If no lease exists, or the existing lease has expired, this method acquires
        a new lease and returns True.

        If a valid lease already exists, this method returns False without acquiring
        a lease.

        Args:
            lease_key: The key that identifies this scheduler lease in the database.
            ttl_seconds: The duration of the lease in seconds. Must be greater than zero.

        Returns:
            True if this replica now holds the lease. False otherwise.

        Raises:
            MlflowException: Invalid ttl_seconds value.
        """

        if ttl_seconds <= 0:
            raise MlflowException.invalid_parameter_value(
                f"ttl_seconds must be greater than zero, got {ttl_seconds}"
            )

        try:
            with self._session_maker(read_only=False) as session:
                existing = (
                    session
                    .query(SqlSchedulerLease)
                    .filter(SqlSchedulerLease.lease_key == lease_key)
                    .with_for_update()
                    .one_or_none()
                )
                now = get_current_time_millis()

                if existing is not None:
                    if now < existing.acquired_at + (existing.ttl_seconds * 1000):
                        _logger.debug("Scheduler lease denied, a valid lease exists.")
                        return False

                    existing.acquired_at = now
                    existing.ttl_seconds = ttl_seconds
                else:
                    session.add(
                        SqlSchedulerLease(
                            lease_key=lease_key, acquired_at=now, ttl_seconds=ttl_seconds
                        )
                    )

        except MlflowException as e:
            # ManagedSessionMaker wraps all SQLAlchemy exceptions in MlflowException.
            # An IntegrityError can occur at startup when two replicas both find no
            # existing row and race to insert the same lease key.
            if isinstance(e.__cause__, IntegrityError):
                _logger.debug("Lease acquisition denied. A concurrent insert conflict occurred.")
                return False

            raise

        return True
