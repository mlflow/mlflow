from pathlib import Path
from unittest import mock

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Query
from sqlalchemy.orm.session import Session

from mlflow.exceptions import MlflowException
from mlflow.server.jobs.lock_manager import JobLockManager
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.tracking.dbmodels.models import SqlSchedulerLease


@pytest.fixture
def job_store() -> SqlAlchemyJobStore:
    return SqlAlchemyJobStore("sqlite:///:memory:")


@pytest.fixture
def lock_mgr(job_store: SqlAlchemyJobStore) -> JobLockManager:
    return JobLockManager(job_store)


@pytest.fixture
def threadsafe_job_store(tmp_path: Path) -> SqlAlchemyJobStore:
    db_path = tmp_path / "test_locks.db"
    return SqlAlchemyJobStore(f"sqlite:///{db_path}")


@pytest.fixture
def threadsafe_lock_mgr(threadsafe_job_store: SqlAlchemyJobStore) -> JobLockManager:
    return JobLockManager(threadsafe_job_store)


def test_acquire_scheduler_lease_succeeds_when_no_lease_exists(lock_mgr: JobLockManager) -> None:

    scheduler_lease = lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=60)
    assert scheduler_lease.lease_key == "scheduler-1"
    assert scheduler_lease.ttl_seconds == 60


@pytest.mark.parametrize("invalid_ttl", [-1, 0], ids=["negative", "zero"])
def test_acquire_scheduler_lease_invalid_ttl(lock_mgr: JobLockManager, invalid_ttl: int) -> None:

    match = f"ttl_seconds must be greater than zero, got {invalid_ttl}"
    with pytest.raises(MlflowException, match=match):
        _ = lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=invalid_ttl)


def test_acquire_scheduler_lease_fails_when_lease_held(lock_mgr: JobLockManager) -> None:

    # First call acquires
    scheduler_lease = lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=60)
    assert scheduler_lease.lease_key == "scheduler-1"
    assert scheduler_lease.ttl_seconds == 60

    # Second call should fail
    assert lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=60) is None


def test_multiple_scheduler_leases_can_coexist(lock_mgr: JobLockManager) -> None:

    scheduler_lease = lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=60)
    assert scheduler_lease.lease_key == "scheduler-1"
    assert scheduler_lease.ttl_seconds == 60

    scheduler_lease = lock_mgr.acquire_scheduler_lease("scheduler-2", ttl_seconds=60)
    assert scheduler_lease.lease_key == "scheduler-2"
    assert scheduler_lease.ttl_seconds == 60

    # Both leases should be held
    assert lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=60) is None
    assert lock_mgr.acquire_scheduler_lease("scheduler-2", ttl_seconds=60) is None


@pytest.mark.parametrize(
    ("attempt_delta_seconds", "expected"),
    [(5, False), (10, True), (11, True)],
    ids=["within_ttl", "at_ttl", "past_ttl"],
)
def test_acquire_scheduler_lease_when_a_lease_exists(
    lock_mgr: JobLockManager,
    attempt_delta_seconds: int,
    expected: bool,
) -> None:
    """Acquiring a job scheduler lock with an existing lease should fail when lease is valid and
    succeed when lease is expired.
    """

    lease_key = "scheduler-1"
    base_time = 1_000_000_000_000
    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis"

    with mock.patch(patch_time_lock_mgr, return_value=base_time) as mock_time:
        scheduler_lease = lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=10)
        assert scheduler_lease.lease_key == "scheduler-1"
        assert scheduler_lease.acquired_at == base_time
        assert scheduler_lease.ttl_seconds == 10
        mock_time.assert_called_once()

    new_time = base_time + (attempt_delta_seconds * 1000)
    with mock.patch(patch_time_lock_mgr, return_value=new_time) as mock_time:
        scheduler_lease = lock_mgr.acquire_scheduler_lease(lease_key, ttl_seconds=10)
        mock_time.assert_called_once()

    if expected:
        assert scheduler_lease.lease_key == "scheduler-1"
        assert scheduler_lease.acquired_at == new_time
        assert scheduler_lease.ttl_seconds == 10

        # Validate lease expiry updated when expected
        with lock_mgr._session_maker() as session:
            lock = (
                session
                .query(SqlSchedulerLease)
                .filter(SqlSchedulerLease.lease_key == lease_key)
                .one()
            )
            assert lock.acquired_at == new_time
            assert lock.ttl_seconds == 10

        # Validate new lease window when expected
        with mock.patch(patch_time_lock_mgr, return_value=(new_time + 1000)) as mock_time:
            assert lock_mgr.acquire_scheduler_lease(lease_key, ttl_seconds=10) is None
        mock_time.assert_called_once()
    else:
        assert scheduler_lease is None


def test_acquire_scheduler_lease_returns_none_on_concurrent_insert_race(
    lock_mgr: JobLockManager,
) -> None:

    integrity_error = IntegrityError(None, None, None)
    with mock.patch.object(Session, "add", side_effect=integrity_error) as mock_add:
        assert lock_mgr.acquire_scheduler_lease("scheduler", ttl_seconds=60) is None
        mock_add.assert_called_once()


def test_acquire_scheduler_lease_raises_non_integrity_errors(lock_mgr: JobLockManager) -> None:
    """Mock the session manager to raise a ValueError and ensure it is wrapped in an
    MlflowException
    """

    # Patching the get_current_time_millis call inside of the session context manager is the
    # easiest way to raise an arbitrary exception though it may not be the source.
    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis"
    exception = ValueError("Non IntegrityError exception")

    with mock.patch(patch_time_lock_mgr, side_effect=exception) as mock_time:
        with pytest.raises(MlflowException, match="Non IntegrityError exception"):
            _ = lock_mgr.acquire_scheduler_lease("scheduler", ttl_seconds=60)
        mock_time.assert_called_once()


def test_acquire_scheduler_lease_returns_none_on_concurrent_insert_race_mock(
    lock_mgr: JobLockManager,
) -> None:

    # Insert the row so it exists in the DB
    lock_mgr.acquire_scheduler_lease("scheduler", ttl_seconds=60)

    # Mock only one_or_none to return None, simulating the race gap where
    # another replica committed after our SELECT but before our INSERT
    with mock.patch.object(Query, "one_or_none", return_value=None) as mock_one_or_none:
        assert lock_mgr.acquire_scheduler_lease("scheduler", ttl_seconds=60) is None
        mock_one_or_none.assert_called_once()
