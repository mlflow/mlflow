import uuid
from logging import Logger
from unittest import mock

import pytest
import sqlalchemy
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Query
from sqlalchemy.orm.session import Session

from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException
from mlflow.server.jobs.lock_manager import JobLockManager, SchedulerLease
from mlflow.store.jobs.abstract_store import JobUpdateStatus
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.tracking.dbmodels.models import SqlJob, SqlJobLock, SqlSchedulerLease
from mlflow.utils.time import get_current_time_millis


@pytest.fixture
def job_store() -> SqlAlchemyJobStore:
    return SqlAlchemyJobStore("sqlite:///:memory:")


@pytest.fixture
def lock_mgr(job_store: SqlAlchemyJobStore) -> JobLockManager:
    return JobLockManager(job_store)


@pytest.fixture
def sql_job() -> SqlJob:

    job_id = str(uuid.uuid4())
    creation_time = get_current_time_millis()
    return SqlJob(
        id=job_id,
        creation_time=creation_time,
        job_name="job-name",
        params="params",
        status=JobStatus.PENDING.to_int(),
        last_update_time=creation_time,
    )


@pytest.fixture
def sql_job_lock(sql_job: SqlJob) -> SqlJobLock:
    return SqlJobLock(lock_key="lock-key", job_id=sql_job.id, acquired_at=get_current_time_millis())


@pytest.mark.parametrize("invalid_ttl", [-1, 0], ids=["negative", "zero"])
def test_renew_scheduler_lease_raises_when_ttl_invalid(
    lock_mgr: JobLockManager, invalid_ttl: int
) -> None:

    missing_scheduler_lease = SchedulerLease(lease_key="doesnt-exist", acquired_at=0, ttl_seconds=0)
    match = f"ttl_seconds must be greater than zero, got {invalid_ttl}"
    with pytest.raises(MlflowException, match=match):
        _ = lock_mgr.renew_scheduler_lease(missing_scheduler_lease, ttl_seconds=invalid_ttl)


def test_renew_scheduler_lease_fails_when_no_lease_exists(lock_mgr: JobLockManager) -> None:

    missing_scheduler_lease = SchedulerLease(lease_key="doesnt-exist", acquired_at=0, ttl_seconds=1)
    assert lock_mgr.renew_scheduler_lease(missing_scheduler_lease, ttl_seconds=1) is None


@pytest.mark.parametrize(
    "caller_lease",
    [
        SchedulerLease("wrong-key", 0, 10),
        SchedulerLease("lease-key", 1, 10),
        SchedulerLease("lease-key", 0, 11),
        SchedulerLease("lease-key", 1, 11),
    ],
    ids=[
        "different_key",
        "different_acquired_at",
        "different_ttl",
        "different_acquired_at_and_ttl",
    ],
)
def test_renew_scheduler_lease_refused(
    lock_mgr: JobLockManager, caller_lease: SchedulerLease
) -> None:

    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis"
    existing_lease = SchedulerLease("lease-key", 0, 10)

    # stage existing lease
    with mock.patch(patch_time_lock_mgr, return_value=existing_lease.acquired_at) as mock_time:
        assert (
            lock_mgr.acquire_scheduler_lease(existing_lease.lease_key, existing_lease.ttl_seconds)
            == existing_lease
        )
        mock_time.assert_called_once()

    assert lock_mgr.renew_scheduler_lease(caller_lease, caller_lease.ttl_seconds) is None


def test_renew_scheduler_lease_granted_when_existing_lease_valid(lock_mgr: JobLockManager) -> None:

    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis"
    existing_lease = SchedulerLease("lease-key", 0, 10)

    # stage existing lease
    with mock.patch(patch_time_lock_mgr, return_value=existing_lease.acquired_at) as mock_time:
        assert (
            lock_mgr.acquire_scheduler_lease(existing_lease.lease_key, existing_lease.ttl_seconds)
            == existing_lease
        )
        mock_time.assert_called_once()

    with mock.patch(patch_time_lock_mgr, return_value=5) as renew_time:
        renewed_lease = lock_mgr.renew_scheduler_lease(existing_lease, 20)
        assert renewed_lease.lease_key == existing_lease.lease_key
        assert renewed_lease.acquired_at == 5
        assert renewed_lease.ttl_seconds == 20
        renew_time.assert_called_once()

    with lock_mgr._session_maker(read_only=True) as session:
        updated_row = (
            session
            .query(SqlSchedulerLease)
            .filter(SqlSchedulerLease.lease_key == existing_lease.lease_key)
            .one()
        )

        assert updated_row.acquired_at == 5
        assert updated_row.ttl_seconds == 20


def test_renew_scheduler_lease_granted_when_existing_lease_expired(
    lock_mgr: JobLockManager,
) -> None:
    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis"
    existing_lease = SchedulerLease("lease-key", 0, 10)

    # stage existing lease
    with mock.patch(patch_time_lock_mgr, return_value=existing_lease.acquired_at) as mock_time:
        assert (
            lock_mgr.acquire_scheduler_lease(existing_lease.lease_key, existing_lease.ttl_seconds)
            == existing_lease
        )
        mock_time.assert_called_once()

    with mock.patch(patch_time_lock_mgr, return_value=10_000) as renew_time:
        renewed_lease = lock_mgr.renew_scheduler_lease(existing_lease, 20)
        assert renewed_lease.lease_key == existing_lease.lease_key
        assert renewed_lease.acquired_at == 10_000
        assert renewed_lease.ttl_seconds == 20
        renew_time.assert_called_once()

    with lock_mgr._session_maker(read_only=True) as session:
        updated_row = (
            session
            .query(SqlSchedulerLease)
            .filter(SqlSchedulerLease.lease_key == existing_lease.lease_key)
            .one()
        )

        assert updated_row.acquired_at == 10_000
        assert updated_row.ttl_seconds == 20


def test_renew_scheduler_lease_works_once_with_original_lease(lock_mgr: JobLockManager) -> None:

    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis"
    existing_lease = SchedulerLease("lease-key", 0, 10)

    # stage existing lease
    with mock.patch(patch_time_lock_mgr, return_value=existing_lease.acquired_at) as mock_time:
        assert (
            lock_mgr.acquire_scheduler_lease(existing_lease.lease_key, existing_lease.ttl_seconds)
            == existing_lease
        )
        mock_time.assert_called_once()

    with mock.patch(patch_time_lock_mgr, return_value=10_000) as renew_time:
        renewed_lease = lock_mgr.renew_scheduler_lease(existing_lease, 20)
        assert renewed_lease.lease_key == existing_lease.lease_key
        assert renewed_lease.acquired_at == 10_000
        assert renewed_lease.ttl_seconds == 20
        renew_time.assert_called_once()

    assert lock_mgr.renew_scheduler_lease(existing_lease, 20) is None


def test_scheduler_lease_full_lifecycle(lock_mgr: JobLockManager) -> None:

    lease_key = "scheduler"
    ttl = 10
    patch_time = "mlflow.server.jobs.lock_manager.get_current_time_millis"

    # 1. Replica A acquires the lease at T=0.
    with mock.patch(patch_time, return_value=0) as mock_t:
        lease_a = lock_mgr.acquire_scheduler_lease(lease_key, ttl_seconds=ttl)
        assert lease_a is not None
        assert lease_a.lease_key == lease_key
        mock_t.assert_called_once()

    # 2. Replica A renews within TTL at T=5s.
    with mock.patch(patch_time, return_value=5_000) as mock_t:
        lease_a = lock_mgr.renew_scheduler_lease(lease_a, ttl_seconds=ttl)
        assert lease_a is not None
        assert lease_a.acquired_at == 5_000
        mock_t.assert_called_once()

    # 3. Lease expires Replica B acquires the expired lease at T=16s.
    with mock.patch(patch_time, return_value=16_000) as mock_t:
        lease_b = lock_mgr.acquire_scheduler_lease(lease_key, ttl_seconds=ttl)
        assert lease_b is not None
        assert lease_b.acquired_at == 16_000
        mock_t.assert_called_once()

    # 4. Replica A attempts renewal with its stale lease and fails
    assert lock_mgr.renew_scheduler_lease(lease_a, ttl_seconds=ttl) is None

    # 5. Replica B can still renew with its current lease.
    with mock.patch(patch_time, return_value=20_000) as mock_t:
        lease_b = lock_mgr.renew_scheduler_lease(lease_b, ttl_seconds=ttl)
        assert lease_b is not None
        assert lease_b.acquired_at == 20_000
        mock_t.assert_called_once()

    # 6. Verify final DB state matches Replica B's lease.
    with lock_mgr._session_maker() as session:
        row = (
            session.query(SqlSchedulerLease).filter(SqlSchedulerLease.lease_key == lease_key).one()
        )
        assert row.acquired_at == 20_000
        assert row.ttl_seconds == ttl


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


def test_acquire_exclusive_lock_succeeds_when_no_lock_exists(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:

    job = job_store.create_job(job_name="test_job", params="{}", timeout=60.0)
    update_status = job_store.claim_job(job.job_id)
    assert update_status == JobUpdateStatus.APPLIED

    assert lock_mgr.acquire_exclusive_lock("test-lock-key", job.job_id) is True


def test_acquire_exclusive_lock_fails_when_live_job_holds_lock(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:

    job1 = job_store.create_job(job_name="job1", params="{}", timeout=60.0)
    job2 = job_store.create_job(job_name="job2", params="{}", timeout=60.0)

    assert job_store.claim_job(job1.job_id) == JobUpdateStatus.APPLIED
    assert job_store.claim_job(job2.job_id) == JobUpdateStatus.APPLIED

    assert lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id) is True
    assert lock_mgr.acquire_exclusive_lock("shared-key", job2.job_id) is False


def test_acquire_exclusive_lock_evicts_stale_lock_terminal_job(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:

    job1 = job_store.create_job(job_name="job1", params="{}", timeout=60.0)
    assert job_store.claim_job(job1.job_id) == JobUpdateStatus.APPLIED
    assert lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id) is True
    job_store.report_job_result(job1.job_id, status=JobStatus.SUCCEEDED, result="done")

    job2 = job_store.create_job(job_name="job2", params="{}", timeout=60.0)
    assert job_store.claim_job(job2.job_id) == JobUpdateStatus.APPLIED
    assert lock_mgr.acquire_exclusive_lock("shared-key", job2.job_id) is True


def test_acquire_exclusive_lock_is_not_reentrant(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:
    """A second attempt to acquire a valid exclusive lock raises an exception.
    A simple refusal could cause the caller to mark the active job as CANCELED.
    The same job_id can acquire the lock after the lock becomes stale.
    See: test_acquire_exclusive_lock_evicts_stale_lock_terminal_job_for_same_job
    """

    job = job_store.create_job(job_name="job", params="{}", timeout=60.0)
    job_store.claim_job(job.job_id)

    assert lock_mgr.acquire_exclusive_lock("test-key", job.job_id) is True
    with pytest.raises(MlflowException, match="A valid lock already exists for this job_id"):
        _ = lock_mgr.acquire_exclusive_lock("test-key", job.job_id)


def test_acquire_exclusive_lock_evicts_stale_lock_terminal_job_for_same_job(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:

    job1 = job_store.create_job(job_name="job1", params="{}", timeout=60.0)
    assert job_store.claim_job(job1.job_id) == JobUpdateStatus.APPLIED
    assert lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id) is True
    job_store.report_job_result(job1.job_id, status=JobStatus.TIMEOUT, result=None)

    assert lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id) is True


def test_multiple_unique_locks_can_coexist(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:

    job1 = job_store.create_job(job_name="job1", params="{}", timeout=60.0)
    job2 = job_store.create_job(job_name="job2", params="{}", timeout=60.0)

    assert job_store.claim_job(job1.job_id) == JobUpdateStatus.APPLIED
    assert job_store.claim_job(job2.job_id) == JobUpdateStatus.APPLIED

    assert lock_mgr.acquire_exclusive_lock("key-1", job1.job_id) is True
    assert lock_mgr.acquire_exclusive_lock("key-2", job2.job_id) is True

    assert lock_mgr.acquire_exclusive_lock("key-1", job2.job_id) is False
    assert lock_mgr.acquire_exclusive_lock("key-2", job1.job_id) is False


def test_acquire_exclusive_lock_not_evicted_for_needs_recovery_job(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:

    job1 = job_store.create_job(job_name="job1", params="{}", timeout=60.0)
    assert job_store.claim_job(job1.job_id) == JobUpdateStatus.APPLIED

    assert lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id) is True
    job_store.mark_job_needs_recovery(job1.job_id)

    job2 = job_store.create_job(job_name="job2", params="{}", timeout=60.0)
    assert job_store.claim_job(job2.job_id) == JobUpdateStatus.APPLIED
    assert lock_mgr.acquire_exclusive_lock("shared-key", job2.job_id) is False


def test_acquire_exclusive_lock_held_by_deleted_job_raises_if_requesting_job_doesnt_exist(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:
    """The logic in the lock acquisition method assumes the job_id belongs
    to an existion job. If the job does not exist the method will fail
    to remove an orphaned lock and an mlflow exception will be raised.
    """

    lock_key = "orphan-key"
    job_id = "nonexistent-job-id"

    # Insert an orphaned lock row with FK enforcement off to simulate a cascade failure
    with job_store.engine.connect() as conn:
        conn.execute(sqlalchemy.text("PRAGMA foreign_keys = OFF"))
        conn.execute(
            sqlalchemy.text(
                "INSERT INTO job_locks (lock_key, job_id, acquired_at) VALUES (:key, :id, :at)"
            ),
            {"key": lock_key, "id": job_id, "at": get_current_time_millis()},
        )
        conn.commit()

    # Check exception message
    def check_exception(e: MlflowException) -> bool:

        if not isinstance(e.__cause__, IntegrityError):
            return False

        return "FOREIGN KEY constraint failed" in e.message

    # Call with missing job_id
    with pytest.raises(MlflowException, check=check_exception):
        _ = lock_mgr.acquire_exclusive_lock(lock_key, job_id)

    # Validate the orphaned lock still exists
    with lock_mgr._session_maker() as session:
        orphaned_lock = session.query(SqlJobLock).filter(SqlJobLock.lock_key == lock_key).one()
        assert orphaned_lock.lock_key == "orphan-key"
        assert orphaned_lock.job_id == "nonexistent-job-id"


def test_acquire_exclusive_lock_evicts_lock_held_by_deleted_job_and_issues_new_lock(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:
    """Evicts the lock when the holding job row was deleted but the lock row
    remains. Issues a new lock to the requesting job.
    """

    shared_job_lock = "shared-job-lock"

    # Insert an orphaned lock row with FK enforcement off to simulate a cascade failure
    with job_store.engine.connect() as conn:
        conn.execute(sqlalchemy.text("PRAGMA foreign_keys = OFF"))
        conn.execute(
            sqlalchemy.text(
                "INSERT INTO job_locks (lock_key, job_id, acquired_at) VALUES (:key, :id, :at)"
            ),
            {"key": shared_job_lock, "id": "nonexistent-job-id", "at": get_current_time_millis()},
        )
        conn.commit()

    job = job_store.create_job(job_name="new-job", params="{}", timeout=60.0)
    assert job_store.claim_job(job.job_id) == JobUpdateStatus.APPLIED

    assert lock_mgr.acquire_exclusive_lock(shared_job_lock, job.job_id) is True

    with lock_mgr._session_maker() as session:
        new_lock = (
            session.query(SqlJobLock).filter(SqlJobLock.lock_key == shared_job_lock).one_or_none()
        )

        assert new_lock is not None
        assert new_lock.lock_key == shared_job_lock
        assert new_lock.job_id == job.job_id


def test_acquire_exclusive_lock_with_no_timeout_and_active_lease_fails(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:

    job1 = job_store.create_job(job_name="job1", params="{}", timeout=None)
    assert job_store.claim_job(job1.job_id, lease_duration=3600.0) == JobUpdateStatus.APPLIED
    assert lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id) is True

    job2 = job_store.create_job(job_name="job2", params="{}", timeout=None)
    assert job_store.claim_job(job2.job_id) == JobUpdateStatus.APPLIED
    assert lock_mgr.acquire_exclusive_lock("shared-key", job2.job_id) is False


def test_acquire_exclusive_lock_fails_when_no_timeout_and_no_lease_and_non_terminal_job(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:

    job1 = job_store.create_job(job_name="job1", params="{}", timeout=None)
    assert job_store.claim_job(job1.job_id) == JobUpdateStatus.APPLIED
    assert lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id) is True

    job2 = job_store.create_job(job_name="job2", params="{}", timeout=None)
    assert job_store.claim_job(job2.job_id) == JobUpdateStatus.APPLIED
    assert lock_mgr.acquire_exclusive_lock("shared-key", job2.job_id) is False


def test_acquire_exclusive_lock_granted_when_no_timeout_and_no_lease_and_holding_job_terminal(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:

    job1 = job_store.create_job(job_name="job1", params="{}", timeout=None)
    assert job_store.claim_job(job1.job_id) == JobUpdateStatus.APPLIED
    assert lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id) is True

    job_store.finish_job(job1.job_id, "Job 1 finished")

    job2 = job_store.create_job(job_name="job2", params="{}", timeout=None)
    assert job_store.claim_job(job2.job_id) == JobUpdateStatus.APPLIED
    assert lock_mgr.acquire_exclusive_lock("shared-key", job2.job_id) is True


def test_acquire_exclusive_lock_integrity_error_unique_job_ids_returns_false(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:
    """
    Two different jobs race to acquire the same lock. The loser receives
    False so it can mark the job as CANCELED.
    """

    job1 = job_store.create_job(job_name="job1", params="{}", timeout=None)
    assert job_store.claim_job(job1.job_id) == JobUpdateStatus.APPLIED
    assert lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id) is True

    job2 = job_store.create_job(job_name="job2", params="{}", timeout=None)
    assert job_store.claim_job(job2.job_id) == JobUpdateStatus.APPLIED

    original_one_or_none = Query.one_or_none
    call_count = 0

    def patched_one_or_none(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None
        return original_one_or_none(self, *args, **kwargs)

    # Patch one_or_none so the first call returns None to simulate a race condition
    with mock.patch.object(
        Query, "one_or_none", side_effect=patched_one_or_none, autospec=True
    ) as mock_one_or_none:
        assert lock_mgr.acquire_exclusive_lock("shared-key", job2.job_id) is False
        assert mock_one_or_none.call_count == 2


def test_acquire_exclusive_lock_caught_integrity_error_raises_when_no_holding_job(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:
    """
    The method re-raises the IntegrityError when the recovery path cannot
    find the lock row or the holding job's ID.
    """

    job1 = job_store.create_job(job_name="job1", params="{}", timeout=None)
    assert job_store.claim_job(job1.job_id) == JobUpdateStatus.APPLIED

    integrity_error = IntegrityError("re-raise", None, None)
    original_one_or_none = Query.one_or_none
    call_count = 0

    def patched_one_or_none(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return original_one_or_none(self, *args, **kwargs)
        return None

    with (
        mock.patch.object(
            Query, "one_or_none", side_effect=patched_one_or_none, autospec=True
        ) as mock_one_or_none,
        mock.patch.object(Session, "add", side_effect=integrity_error) as mock_add,
    ):
        with pytest.raises(MlflowException, match="re-raise"):
            _ = lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id)
        mock_add.assert_called_once()
        assert mock_one_or_none.call_count == 2


def test_acquire_exclusive_lock_caught_integrity_error_raises_when_job_ids_match(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:
    """
    Simulates a race condition that must not occur in production. Two replicas
    try to acquire a lock for the same job row. The job store prevents this.
    If it occurs, the method must raise an exception, not return False.
    """

    job1 = job_store.create_job(job_name="job1", params="{}", timeout=None)
    assert job_store.claim_job(job1.job_id) == JobUpdateStatus.APPLIED
    assert lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id) is True

    original_one_or_none = Query.one_or_none
    call_count = 0

    def patched_one_or_none(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None
        return original_one_or_none(self, *args, **kwargs)

    def check_exception(e: MlflowException) -> bool:

        if not isinstance(e.__cause__, IntegrityError):
            return False

        return "UNIQUE constraint failed: job_locks.lock_key" in e.message

    # Patch one_or_none so the first call returns None to simulate a race condition
    with (
        mock.patch.object(
            Query, "one_or_none", side_effect=patched_one_or_none, autospec=True
        ) as mock_one_or_none,
        mock.patch.object(Logger, "error") as mock_logger_error,
    ):
        with pytest.raises(MlflowException, check=check_exception):
            _ = lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id)

        assert mock_one_or_none.call_count == 2
        mock_logger_error.assert_called_once_with(
            "An unexpected IntegrityError occurred during job lock acquisition"
        )


def test_acquire_exclusive_lock_reraises_non_integrity_mlflow_exception(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:

    job1 = job_store.create_job(job_name="job1", params="{}", timeout=None)
    assert job_store.claim_job(job1.job_id) == JobUpdateStatus.APPLIED

    non_integrity_error = RuntimeError("not integrity")
    with mock.patch.object(Session, "add", side_effect=non_integrity_error) as mock_add:
        with pytest.raises(MlflowException, match="not integrity"):
            lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id)
        mock_add.assert_called_once()


def test_acquire_exclusive_lock_evicts_stale_lock_timeout_exceeded(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:
    base_time = 1_000_000_000_000
    patch_time_jobstore = "mlflow.store.jobs.sqlalchemy_store.get_current_time_millis"
    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis"
    timeout = 60.0
    shared_job_lock = "shared-key"

    with mock.patch(patch_time_jobstore, return_value=base_time):
        job1 = job_store.create_job(job_name="job1", params="{}", timeout=timeout)
        assert job_store.claim_job(job1.job_id) == JobUpdateStatus.APPLIED

    with mock.patch(patch_time_lock_mgr, return_value=base_time):
        assert lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id) is True

    # 115% of timeout + 1 ms
    time_after_timeout = base_time + int(timeout * 1.15 * 1_000) + 1

    with mock.patch(patch_time_jobstore, return_value=time_after_timeout):
        job2 = job_store.create_job(job_name="job2", params="{}", timeout=timeout)
        assert job_store.claim_job(job2.job_id) == JobUpdateStatus.APPLIED

    with mock.patch(patch_time_lock_mgr, return_value=time_after_timeout):
        assert lock_mgr.acquire_exclusive_lock("shared-key", job2.job_id) is True

    with lock_mgr._session_maker() as session:
        new_lock = (
            session.query(SqlJobLock).filter(SqlJobLock.lock_key == shared_job_lock).one_or_none()
        )

        assert new_lock is not None
        assert new_lock.lock_key == shared_job_lock
        assert new_lock.job_id == job2.job_id


def test_acquire_exclusive_lock_fails_when_job_not_timed_out_but_lease_has_expired(
    job_store: SqlAlchemyJobStore, lock_mgr: JobLockManager
) -> None:
    base_time = 1_000_000_000_000
    patch_time_jobstore = "mlflow.store.jobs.sqlalchemy_store.get_current_time_millis"
    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis"

    with mock.patch(patch_time_jobstore, return_value=base_time):
        job1 = job_store.create_job(job_name="job1", params="{}", timeout=60.0)
        assert job_store.claim_job(job1.job_id, lease_duration=10.0) == JobUpdateStatus.APPLIED

    with mock.patch(patch_time_lock_mgr, return_value=base_time):
        assert lock_mgr.acquire_exclusive_lock("shared-key", job1.job_id) is True

    # Advance past lease expiry but still within timeout
    after_lease = base_time + 11_000

    with mock.patch(patch_time_jobstore, return_value=after_lease):
        job2 = job_store.create_job(job_name="job2", params="{}", timeout=60.0)
        assert job_store.claim_job(job2.job_id) == JobUpdateStatus.APPLIED

    with mock.patch(patch_time_lock_mgr, return_value=after_lease):
        assert lock_mgr.acquire_exclusive_lock("shared-key", job2.job_id) is False


@pytest.mark.parametrize(
    ("lease_duration", "timeout", "time_delta", "expected"),
    [
        (None, None, 0, True),
        (11.5, None, 11_500, True),
        (None, 10.0, 11_500, True),
        (11.5, 10.0, 11_500, True),
        (11.5, None, 12_000, True),
        (None, 10.0, 12_000, False),
        (11.5, 10.0, 12_000, False),
        (11.5, 20.0, 12_000, True),
        (20.0, 10.0, 12_000, False),
    ],
    ids=[
        "no_lease_or_timeout_is_valid",
        "valid_lease_no_timeout_is_valid",
        "no_lease_valid_timeout_is_valid",
        "valid_lease_valid_timeout_is_valid",
        "expired_lease_no_timeout_is_valid",
        "no_lease_expired_timeout_is_not_valid",
        "expired_lease_expired_timeout_is_not_valid",
        "expired_lease_valid_timeout_is_valid",
        "valid_lease_expired_timeout_is_not_valid",
    ],
)
def test_is_job_lock_valid_lease_and_timeout_interleaving(
    sql_job: SqlJob,
    sql_job_lock: SqlJobLock,
    lease_duration: float | None,
    timeout: float | None,
    time_delta: int,
    expected: bool,
) -> None:

    base_time = 1_000_000_000_000
    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis"

    sql_job_lock.acquired_at = base_time
    sql_job.creation_time = base_time
    sql_job.timeout = timeout
    sql_job.lease_expires_at = lease_duration
    if lease_duration is not None:
        sql_job.lease_expires_at = base_time + int(lease_duration * 1_000)

    check_time = base_time + time_delta
    with mock.patch(patch_time_lock_mgr, return_value=check_time):
        assert JobLockManager._is_job_lock_valid(sql_job_lock, sql_job) is expected


def test_is_job_lock_valid_false_when_holding_job_none(sql_job_lock: SqlJobLock) -> None:
    assert JobLockManager._is_job_lock_valid(sql_job_lock, None) is False


def test_is_job_lock_valid_raises_when_lock_and_holding_dont_match(
    sql_job: SqlJob,
    sql_job_lock: SqlJobLock,
) -> None:

    sql_job.id = str(uuid.uuid4())
    match = (
        f"Lock is not held by SqlJob lock.job_id='{sql_job_lock.job_id}'"
        f" != holding_job.id='{sql_job.id}'"
    )
    with pytest.raises(MlflowException, match=match):
        _ = JobLockManager._is_job_lock_valid(sql_job_lock, sql_job)


@pytest.mark.parametrize(
    "job_status",
    [
        JobStatus.PENDING,
        JobStatus.RUNNING,
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.TIMEOUT,
        JobStatus.CANCELED,
        JobStatus.NEEDS_RECOVERY,
    ],
)
def test_is_job_lock_valid_status_values(
    sql_job: SqlJob, sql_job_lock: SqlJobLock, job_status: JobStatus
) -> None:

    # Verify unmodified fixtures are valid
    assert JobLockManager._is_job_lock_valid(sql_job_lock, sql_job) is True

    # Update status
    sql_job.status = JobStatus.to_int(job_status)
    is_finalized = JobStatus.is_finalized(job_status)
    assert JobLockManager._is_job_lock_valid(sql_job_lock, sql_job) is not is_finalized
