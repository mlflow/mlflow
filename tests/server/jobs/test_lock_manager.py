import threading
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

    assert lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=60) is True


@pytest.mark.parametrize("invalid_ttl", [-1, 0], ids=["negative", "zero"])
def test_acquire_scheduler_lease_invalid_ttl(lock_mgr: JobLockManager, invalid_ttl: int) -> None:

    match = f"ttl_seconds must be greater than zero, got {invalid_ttl}"
    with pytest.raises(MlflowException, match=match):
        lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=invalid_ttl)


def test_acquire_scheduler_lease_fails_when_lease_held(lock_mgr: JobLockManager) -> None:

    # First call acquires
    assert lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=60) is True

    # Second call should fail
    assert lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=60) is False


def test_multiple_scheduler_leases_can_coexist(lock_mgr: JobLockManager) -> None:

    assert lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=60) is True
    assert lock_mgr.acquire_scheduler_lease("scheduler-2", ttl_seconds=60) is True

    # Both leases should be held
    assert lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=60) is False
    assert lock_mgr.acquire_scheduler_lease("scheduler-2", ttl_seconds=60) is False


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
        assert lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=10) is True
        mock_time.assert_called_once()

    new_ttl = base_time + (attempt_delta_seconds * 1000)
    with mock.patch(patch_time_lock_mgr, return_value=new_ttl) as mock_time:
        assert lock_mgr.acquire_scheduler_lease(lease_key, ttl_seconds=10) is expected
        mock_time.assert_called_once()

    if expected:
        # Validate lease expiry updated when expected
        with lock_mgr._session_maker() as session:
            lock = (
                session
                .query(SqlSchedulerLease)
                .filter(SqlSchedulerLease.lease_key == lease_key)
                .one()
            )
            assert lock.acquired_at == new_ttl
            assert lock.ttl_seconds == 10

        # Validate new lease window when expected
        with mock.patch(patch_time_lock_mgr, return_value=(new_ttl + 1000)) as mock_time:
            assert lock_mgr.acquire_scheduler_lease(lease_key, ttl_seconds=10) is False
        mock_time.assert_called_once()


def test_concurrent_acquire_scheduler_lease(threadsafe_lock_mgr: JobLockManager) -> None:
    """Two threads racing to acquire the same scheduler lease, only one should succeed.
    SQLite does not support SELECT ... FOR UPDATE; it serializes via file-level write locking.
    This test verifies the concurrent behavior but does not exercise the FOR UPDATE path
    that protects existing rows on PostgreSQL/MySQL in production. True row-level locking
    semantics require a transactional backend.
    """

    barrier = threading.Barrier(2)
    results = []

    # Mock only one_or_none to return None, simulating the race gap where
    # another replica committed after our SELECT but before our INSERT
    with mock.patch.object(Query, "one_or_none", return_value=None) as mock_one_or_none:

        def try_acquire():
            barrier.wait()
            result = threadsafe_lock_mgr.acquire_scheduler_lease("scheduler-1", ttl_seconds=60)
            results.append(result)

        t1 = threading.Thread(target=try_acquire, name="mock-replica-1")
        t2 = threading.Thread(target=try_acquire, name="mock-replica-2")
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        mock_one_or_none.assert_called()
        assert mock_one_or_none.call_count == 2

    # Exactly one should succeed
    assert results.count(True) == 1
    assert results.count(False) == 1


def test_acquire_scheduler_lease_returns_false_on_concurrent_insert_race(
    lock_mgr: JobLockManager,
) -> None:

    integrity_error = IntegrityError(None, None, None)
    with mock.patch.object(Session, "add", side_effect=integrity_error) as mock_add:
        assert lock_mgr.acquire_scheduler_lease("scheduler", ttl_seconds=60) is False
        mock_add.assert_called_once()


def test_acquire_scheduler_lease_raises_non_integriy_errors(lock_mgr: JobLockManager) -> None:
    """Mock the session manager to raise a ValueError and ensure it is wrapped in an
    MlflowException
    """

    # Patching the get_current_time_millis call inside of the session context manager is the
    # easiest way to raise an arbitrary exception though it may not be the source.
    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis"
    exception = ValueError("Non IntegriryError exception")

    with mock.patch(patch_time_lock_mgr, side_effect=exception) as mock_time:
        with pytest.raises(MlflowException, match="Non IntegriryError exception"):
            lock_mgr.acquire_scheduler_lease("scheduler", ttl_seconds=60)
        mock_time.assert_called_once()


def test_acquire_scheduler_lease_returns_false_on_concurrent_insert_race_mock(
    lock_mgr: JobLockManager,
) -> None:

    # Insert the row so it exists in the DB
    lock_mgr.acquire_scheduler_lease("scheduler", ttl_seconds=60)

    # Mock only one_or_none to return None, simulating the race gap where
    # another replica committed after our SELECT but before our INSERT
    with mock.patch.object(Query, "one_or_none", return_value=None) as mock_one_or_none:
        assert lock_mgr.acquire_scheduler_lease("scheduler", ttl_seconds=60) is False
        mock_one_or_none.assert_called_once()
