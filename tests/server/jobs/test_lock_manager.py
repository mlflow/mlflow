from logging import Logger
from unittest import mock
from unittest.mock import Mock

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Query
from sqlalchemy.orm.session import Session

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import BAD_REQUEST, TEMPORARILY_UNAVAILABLE
from mlflow.server.jobs.lock_manager import (
    JobLockManager,
    SchedulerLease,
    _check_time_drift_and_log,
)
from mlflow.store.db.db_types import MSSQL, MYSQL, POSTGRES
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.tracking.dbmodels.models import SqlSchedulerLease


@pytest.fixture
def job_store() -> SqlAlchemyJobStore:
    return SqlAlchemyJobStore("sqlite:///:memory:")


@pytest.fixture
def lock_mgr(job_store: SqlAlchemyJobStore) -> JobLockManager:
    return JobLockManager(job_store)


@pytest.mark.parametrize("invalid_ttl", [-1, 0], ids=["negative", "zero"])
def test_renew_scheduler_lease_raises_when_ttl_invalid(
    lock_mgr: JobLockManager, invalid_ttl: int
) -> None:

    missing_scheduler_lease = SchedulerLease(lease_key="doesnt-exist", acquired_at=0, ttl_seconds=0)
    match = f"ttl_seconds must be greater than zero, got {invalid_ttl}"
    with pytest.raises(MlflowException, match=match):
        _ = lock_mgr.renew_scheduler_lease(missing_scheduler_lease, ttl_seconds=invalid_ttl)


@pytest.mark.parametrize(
    ("app_now", "db_now", "drift"),
    [
        (0, 0, None),
        (0, 1000, None),
        (0, 1001, -1001),
        (1001, 0, 1001),
    ],
    ids=["no_drift", "exactly_1_second", "db_drift_ahead", "app_drift_ahead"],
)
def test_check_time_drift_and_log(app_now: int, db_now: int, drift: int | None) -> None:

    with mock.patch.object(Logger, "warning") as mock_logger_warning:
        _check_time_drift_and_log(app_now_millis=app_now, db_now_millis=db_now)

    if drift is not None:
        mock_logger_warning.assert_called_once_with(
            "Time drift > 1s detected APP_time - DB_time = %d ms", drift
        )
    else:
        mock_logger_warning.assert_not_called()


def test_is_unique_constraint_violation_sqlite(lock_mgr: JobLockManager) -> None:

    base = BaseException("UNIQUE constraint failed")
    integrity_error = IntegrityError(None, None, base)
    assert lock_mgr._is_unique_constraint_violation(integrity_error)

    base = BaseException("other constraint failed")
    integrity_error = IntegrityError(None, None, base)
    assert not lock_mgr._is_unique_constraint_violation(integrity_error)


def test_is_unique_constraint_violation_pgsql(lock_mgr: JobLockManager) -> None:

    lock_mgr.db_type = POSTGRES

    base = Mock()
    base.sqlstate = "23505"
    base.pgcode = None
    integrity_error = IntegrityError(None, None, base)
    assert lock_mgr._is_unique_constraint_violation(integrity_error)

    base = Mock()
    base.sqlstate = "23503"
    base.pgcode = None
    integrity_error = IntegrityError(None, None, base)
    assert not lock_mgr._is_unique_constraint_violation(integrity_error)

    base = Mock()
    base.sqlstate = None
    base.pgcode = "23505"
    integrity_error = IntegrityError(None, None, base)
    assert lock_mgr._is_unique_constraint_violation(integrity_error)

    base = Mock()
    base.sqlstate = None
    base.pgcode = "23503"
    integrity_error = IntegrityError(None, None, base)
    assert not lock_mgr._is_unique_constraint_violation(integrity_error)

    base = BaseException("UNIQUE constraint failed")
    integrity_error = IntegrityError(None, None, base)
    assert not lock_mgr._is_unique_constraint_violation(integrity_error)


def test_is_unique_constraint_violation_mssql(lock_mgr: JobLockManager) -> None:

    lock_mgr.db_type = MSSQL

    base = Mock()
    base.args = (2627, "other arg")
    integrity_error = IntegrityError(None, None, base)
    assert lock_mgr._is_unique_constraint_violation(integrity_error)

    base = Mock()
    base.args = (2601, "other arg")
    integrity_error = IntegrityError(None, None, base)
    assert lock_mgr._is_unique_constraint_violation(integrity_error)

    base = Mock()
    base.args = ("other arg", 2601)
    integrity_error = IntegrityError(None, None, base)
    assert not lock_mgr._is_unique_constraint_violation(integrity_error)

    base = Mock()
    base.args = ("other arg", "(2601, error)")
    integrity_error = IntegrityError(None, None, base)
    assert not lock_mgr._is_unique_constraint_violation(integrity_error)

    base = Mock()
    base.args = ("other arg", "(2627, error)")
    integrity_error = IntegrityError(None, None, base)
    assert not lock_mgr._is_unique_constraint_violation(integrity_error)

    base = Mock()
    base.args = ("other arg", "(2600, error)")
    integrity_error = IntegrityError(None, None, base)
    assert not lock_mgr._is_unique_constraint_violation(integrity_error)

    base = BaseException("UNIQUE constraint failed")
    integrity_error = IntegrityError(None, None, base)
    assert not lock_mgr._is_unique_constraint_violation(integrity_error)


def test_is_unique_constraint_violation_mysql(lock_mgr: JobLockManager) -> None:

    lock_mgr.db_type = MYSQL

    base = Mock()
    base.args = (1062, "other arg")
    integrity_error = IntegrityError(None, None, base)
    assert lock_mgr._is_unique_constraint_violation(integrity_error)

    base = Mock()
    base.args = (1060, "other arg")
    integrity_error = IntegrityError(None, None, base)
    assert not lock_mgr._is_unique_constraint_violation(integrity_error)

    base = BaseException("UNIQUE constraint failed")
    integrity_error = IntegrityError(None, None, base)
    assert not lock_mgr._is_unique_constraint_violation(integrity_error)


def test_is_unique_constraint_violation_raises(lock_mgr: JobLockManager) -> None:

    lock_mgr.db_type = "not-supported"

    base = BaseException("UNIQUE constraint failed")
    integrity_error = IntegrityError(None, None, base)

    with pytest.raises(MlflowException, match="Unsupported db type: not-supported"):
        _ = lock_mgr._is_unique_constraint_violation(integrity_error)


def test_guard_insert_race_success(lock_mgr: JobLockManager) -> None:

    def fn(a: int) -> int:
        return a**2

    assert lock_mgr._guard_insert_race(fn, 2) == 4


def test_guard_insert_race_integrity_error_returns_null(lock_mgr: JobLockManager) -> None:

    def fn(a: int) -> int:
        base = BaseException("unique constraint failed")
        e = IntegrityError(None, None, base)
        raise MlflowException(message=e, error_code=BAD_REQUEST) from e

    assert lock_mgr._guard_insert_race(fn, 2) is None


def test_guard_insert_race_value_error_raises(lock_mgr: JobLockManager) -> None:

    def fn(a: int) -> int:
        raise ValueError("value error")

    with pytest.raises(ValueError, match="value error"):
        _ = lock_mgr._guard_insert_race(fn, 2)


def test_guard_insert_race_operational_error_non_deadlock_raises(lock_mgr: JobLockManager) -> None:

    def fn(a: int) -> int:
        e = OperationalError("mock operational error", None, None)
        raise MlflowException(message=e, error_code=TEMPORARILY_UNAVAILABLE) from e

    with pytest.raises(MlflowException, match="mock operational error"):
        _ = lock_mgr._guard_insert_race(fn, 2)


def test_guard_insert_race_operational_error_raises_after_retries(lock_mgr: JobLockManager) -> None:

    def fn(a: int) -> int:
        e = OperationalError("mock deadlock error", None, None)
        raise MlflowException(message=e, error_code=TEMPORARILY_UNAVAILABLE) from e

    with mock.patch("mlflow.server.jobs.lock_manager.time.sleep") as mock_sleep:
        with pytest.raises(MlflowException, match="mock deadlock error"):
            _ = lock_mgr._guard_insert_race(fn, 2)
        mock_sleep.assert_called()


def test_guard_insert_race_operational_error_retries_and_succeeds(lock_mgr: JobLockManager) -> None:

    call_count = 0

    def fn(a: int) -> int:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            e = OperationalError("mock deadlock error", None, None)
            raise MlflowException(message=e, error_code=TEMPORARILY_UNAVAILABLE) from e

        return a**2

    with mock.patch("mlflow.server.jobs.lock_manager.time.sleep") as mock_sleep:
        assert lock_mgr._guard_insert_race(fn, 2) == 4
        mock_sleep.assert_called()


def test_guard_insert_race_operational_error_retries_only_deadlock(
    lock_mgr: JobLockManager,
) -> None:

    call_count = 0

    def fn(a: int) -> int:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            e = OperationalError("mock deadlock error", None, None)
            raise MlflowException(message=e, error_code=TEMPORARILY_UNAVAILABLE) from e

        base = BaseException("unique constraint failed")
        e = IntegrityError(None, None, base)
        raise MlflowException(message=e, error_code=BAD_REQUEST) from e

    with mock.patch("mlflow.server.jobs.lock_manager.time.sleep") as mock_sleep:
        assert lock_mgr._guard_insert_race(fn, 2) is None
        mock_sleep.assert_called_once()


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

    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis_expression"
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

    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis_expression"
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
    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis_expression"
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

    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis_expression"
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
    patch_time = "mlflow.server.jobs.lock_manager.get_current_time_millis_expression"

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
    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis_expression"

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

    base = BaseException("unique constraint failed")
    integrity_error = IntegrityError(None, None, base)
    with mock.patch.object(Session, "execute", side_effect=integrity_error) as mock_add:
        assert lock_mgr.acquire_scheduler_lease("scheduler", ttl_seconds=60) is None
        mock_add.assert_called_once()


def test_acquire_scheduler_lease_raises_non_integrity_errors(lock_mgr: JobLockManager) -> None:
    """Mock the session manager to raise a ValueError and ensure it is wrapped in an
    MlflowException
    """

    # Patching the get_current_time_millis call inside of the session context manager is the
    # easiest way to raise an arbitrary exception though it may not be the source.
    patch_time_lock_mgr = "mlflow.server.jobs.lock_manager.get_current_time_millis_expression"
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
