import threading
from unittest import mock

from mlflow.server.jobs._scheduler_leadership import _SchedulerLeadershipCoordinator
from mlflow.server.jobs.lock_manager import JobLockManager, SchedulerLease
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore


def _coordinator(lock_manager, start_consumer, stop_event, **kwargs):
    return _SchedulerLeadershipCoordinator(
        lock_manager,
        start_consumer,
        stop_event=stop_event,
        lease_ttl_seconds=kwargs.get("lease_ttl_seconds", 30),
        renew_interval_seconds=kwargs.get("renew_interval_seconds", 60),
        follower_retry_seconds=0,
        follower_retry_jitter_seconds=0,
        child_poll_interval_seconds=0,
    )


def test_leader_starts_periodic_consumer():
    lease = SchedulerLease("periodic-tasks-scheduler", 1, 30)
    lock_manager = mock.Mock()
    lock_manager.acquire_scheduler_lease.return_value = lease
    stop_event = threading.Event()
    consumer = mock.Mock()
    consumer.poll.return_value = None

    def start_consumer():
        stop_event.set()
        return consumer

    _coordinator(lock_manager, start_consumer, stop_event).run()

    lock_manager.acquire_scheduler_lease.assert_called_once_with(
        "periodic-tasks-scheduler", ttl_seconds=30
    )
    consumer.kill.assert_called_once()
    consumer.wait.assert_called_once_with(timeout=5.0)


def test_follower_does_not_start_periodic_consumer_and_retries():
    lease = SchedulerLease("periodic-tasks-scheduler", 1, 30)
    lock_manager = mock.Mock()
    lock_manager.acquire_scheduler_lease.side_effect = [None, lease]
    stop_event = threading.Event()
    consumer = mock.Mock()
    consumer.poll.return_value = None

    def start_consumer():
        stop_event.set()
        return consumer

    start_consumer = mock.Mock(side_effect=start_consumer)
    _coordinator(lock_manager, start_consumer, stop_event).run()

    assert lock_manager.acquire_scheduler_lease.call_count == 2
    start_consumer.assert_called_once()


def test_leader_renews_lease():
    lease = SchedulerLease("periodic-tasks-scheduler", 1, 30)
    renewed_lease = SchedulerLease("periodic-tasks-scheduler", 2, 30)
    lock_manager = mock.Mock()
    stop_event = threading.Event()
    consumer = mock.Mock()
    consumer.poll.return_value = None

    def renew(*args, **kwargs):
        stop_event.set()
        return renewed_lease

    lock_manager.renew_scheduler_lease.side_effect = renew
    coordinator = _coordinator(lock_manager, lambda: consumer, stop_event, renew_interval_seconds=0)
    coordinator._consumer = consumer

    coordinator._run_as_leader(lease)

    lock_manager.renew_scheduler_lease.assert_called_once_with(lease, ttl_seconds=30)
    consumer.kill.assert_called_once()


def test_renewal_refusal_stops_periodic_consumer():
    lease = SchedulerLease("periodic-tasks-scheduler", 1, 30)
    lock_manager = mock.Mock()
    lock_manager.renew_scheduler_lease.return_value = None
    stop_event = threading.Event()
    consumer = mock.Mock()
    consumer.poll.return_value = None
    coordinator = _coordinator(lock_manager, lambda: consumer, stop_event, renew_interval_seconds=0)
    coordinator._consumer = consumer

    coordinator._run_as_leader(lease)

    lock_manager.renew_scheduler_lease.assert_called_once_with(lease, ttl_seconds=30)
    consumer.kill.assert_called_once()


def test_renewal_error_stops_periodic_consumer():
    lease = SchedulerLease("periodic-tasks-scheduler", 1, 30)
    lock_manager = mock.Mock()
    lock_manager.renew_scheduler_lease.side_effect = RuntimeError("database unavailable")
    stop_event = threading.Event()
    consumer = mock.Mock()
    consumer.poll.return_value = None
    coordinator = _coordinator(lock_manager, lambda: consumer, stop_event, renew_interval_seconds=0)
    coordinator._consumer = consumer

    coordinator._run_as_leader(lease)

    lock_manager.renew_scheduler_lease.assert_called_once_with(lease, ttl_seconds=30)
    consumer.kill.assert_called_once()


def test_blocked_renewal_stops_consumer_at_lease_deadline():
    lease = SchedulerLease("periodic-tasks-scheduler", 1, 1)
    renewed_lease = SchedulerLease("periodic-tasks-scheduler", 2, 1)
    lock_manager = mock.Mock()
    renewal_started = threading.Event()
    release_renewal = threading.Event()
    consumer_stopped = threading.Event()

    def renew(*args, **kwargs):
        renewal_started.set()
        release_renewal.wait(timeout=5)
        return renewed_lease

    lock_manager.renew_scheduler_lease.side_effect = renew
    consumer = mock.Mock()
    consumer.poll.return_value = None
    consumer.kill.side_effect = consumer_stopped.set
    start_consumer = mock.Mock()
    coordinator = _coordinator(
        lock_manager,
        start_consumer,
        threading.Event(),
        lease_ttl_seconds=1,
        renew_interval_seconds=0,
    )
    coordinator._consumer = consumer
    thread = threading.Thread(
        target=coordinator._run_as_leader,
        args=(lease,),
        name="test-blocked-scheduler-renewal",
    )
    thread.start()

    try:
        assert renewal_started.wait(timeout=5)
        assert consumer_stopped.wait(timeout=5)
        assert not release_renewal.is_set()
    finally:
        release_renewal.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    start_consumer.assert_not_called()
    consumer.kill.assert_called_once()


def test_expired_lease_does_not_restart_periodic_consumer():
    lease = SchedulerLease("periodic-tasks-scheduler", 1, 30)
    lock_manager = mock.Mock()
    lock_manager.renew_scheduler_lease.return_value = None
    exited_consumer = mock.Mock(returncode=1)
    exited_consumer.poll.return_value = 1
    start_consumer = mock.Mock()
    coordinator = _coordinator(
        lock_manager, start_consumer, threading.Event(), renew_interval_seconds=0
    )
    coordinator._consumer = exited_consumer

    coordinator._run_as_leader(lease, lease_deadline=0)

    lock_manager.renew_scheduler_lease.assert_not_called()
    start_consumer.assert_not_called()
    exited_consumer.wait.assert_called_once_with(timeout=5.0)


def test_consumer_is_reaped_when_it_exits_during_stop():
    consumer = mock.Mock()
    consumer.poll.return_value = None
    consumer.kill.side_effect = ProcessLookupError
    coordinator = _coordinator(mock.Mock(), mock.Mock(), threading.Event())
    coordinator._consumer = consumer

    coordinator._stop_consumer()

    consumer.kill.assert_called_once()
    consumer.wait.assert_called_once_with(timeout=5.0)


def test_acquisition_error_is_retried_as_follower():
    lease = SchedulerLease("periodic-tasks-scheduler", 1, 30)
    lock_manager = mock.Mock()
    lock_manager.acquire_scheduler_lease.side_effect = [RuntimeError("database unavailable"), lease]
    stop_event = threading.Event()
    consumer = mock.Mock()
    consumer.poll.return_value = None

    def start_consumer():
        stop_event.set()
        return consumer

    start_consumer = mock.Mock(side_effect=start_consumer)
    _coordinator(lock_manager, start_consumer, stop_event).run()

    assert lock_manager.acquire_scheduler_lease.call_count == 2
    start_consumer.assert_called_once()


def test_consumer_start_error_is_retried_while_leadership_is_held():
    lease = SchedulerLease("periodic-tasks-scheduler", 1, 30)
    lock_manager = mock.Mock()
    stop_event = threading.Event()
    consumer = mock.Mock()
    consumer.poll.return_value = None

    def start_consumer():
        if start_consumer.call_count == 1:
            raise RuntimeError("could not start consumer")
        stop_event.set()
        return consumer

    start_consumer = mock.Mock(side_effect=start_consumer)
    coordinator = _coordinator(
        lock_manager,
        start_consumer,
        stop_event,
        renew_interval_seconds=60,
    )

    coordinator._run_as_leader(lease)

    assert start_consumer.call_count == 2
    consumer.kill.assert_called_once()


def test_exited_periodic_consumer_restarts_while_leadership_is_held():
    lease = SchedulerLease("periodic-tasks-scheduler", 1, 30)
    lock_manager = mock.Mock()
    stop_event = threading.Event()
    exited_consumer = mock.Mock(returncode=1)
    exited_consumer.poll.return_value = 1
    running_consumer = mock.Mock()
    running_consumer.poll.return_value = None

    def start_consumer():
        if start_consumer.call_count == 2:
            stop_event.set()
        return [exited_consumer, running_consumer][start_consumer.call_count - 1]

    start_consumer = mock.Mock(side_effect=start_consumer)
    coordinator = _coordinator(
        lock_manager,
        start_consumer,
        stop_event,
        renew_interval_seconds=60,
    )

    coordinator._run_as_leader(lease)

    assert start_consumer.call_count == 2
    exited_consumer.kill.assert_not_called()
    running_consumer.kill.assert_called_once()


def test_only_one_coordinator_starts_periodic_consumer(db_uri):
    stop_events = [threading.Event(), threading.Event()]
    acquisition_attempted = [threading.Event(), threading.Event()]
    consumer_started = threading.Event()
    started_by = []
    coordinators = []

    for replica, (stop_event, attempted) in enumerate(
        zip(stop_events, acquisition_attempted, strict=True)
    ):
        real_lock_manager = JobLockManager(SqlAlchemyJobStore(db_uri))
        lock_manager = mock.Mock(wraps=real_lock_manager)

        def acquire(*args, _attempted=attempted, _manager=real_lock_manager, **kwargs):
            _attempted.set()
            return _manager.acquire_scheduler_lease(*args, **kwargs)

        lock_manager.acquire_scheduler_lease.side_effect = acquire

        def start_consumer(_replica=replica):
            started_by.append(_replica)
            consumer_started.set()
            consumer = mock.Mock()
            consumer.poll.return_value = None
            return consumer

        coordinators.append(_coordinator(lock_manager, start_consumer, stop_event))

    threads = [
        threading.Thread(target=coordinator.run, name=f"test-scheduler-coordinator-{index}")
        for index, coordinator in enumerate(coordinators)
    ]
    for thread in threads:
        thread.start()

    try:
        assert all(attempted.wait(timeout=5) for attempted in acquisition_attempted)
        assert consumer_started.wait(timeout=5)
        assert len(started_by) == 1
    finally:
        for stop_event in stop_events:
            stop_event.set()
        for thread in threads:
            thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)


def test_follower_takes_over_expired_lease_and_stale_leader_stops(db_uri):
    patch_time = "mlflow.server.jobs.lock_manager.get_current_time_millis_expression"
    leader_lock_manager = JobLockManager(SqlAlchemyJobStore(db_uri))
    follower_lock_manager = JobLockManager(SqlAlchemyJobStore(db_uri))

    with mock.patch(patch_time, return_value=0):
        leader_lease = leader_lock_manager.acquire_scheduler_lease(
            "periodic-tasks-scheduler", ttl_seconds=30
        )
    with mock.patch(patch_time, return_value=31_000):
        follower_lease = follower_lock_manager.acquire_scheduler_lease(
            "periodic-tasks-scheduler", ttl_seconds=30
        )

    assert leader_lease is not None
    assert follower_lease is not None

    leader_consumer = mock.Mock()
    leader_consumer.poll.return_value = None
    stale_leader = _coordinator(
        leader_lock_manager,
        lambda: leader_consumer,
        threading.Event(),
        renew_interval_seconds=0,
    )
    stale_leader._consumer = leader_consumer
    stale_leader._run_as_leader(leader_lease)

    leader_consumer.kill.assert_called_once()

    follower_stop = threading.Event()
    follower_consumer = mock.Mock()
    follower_consumer.poll.return_value = None

    def start_follower_consumer():
        follower_stop.set()
        return follower_consumer

    follower = _coordinator(
        follower_lock_manager,
        start_follower_consumer,
        follower_stop,
        renew_interval_seconds=60,
    )
    follower._run_as_leader(follower_lease)

    follower_consumer.kill.assert_called_once()
