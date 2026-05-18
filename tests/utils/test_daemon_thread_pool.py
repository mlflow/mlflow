import threading
import time

import pytest

from mlflow.utils.async_logging.daemon_thread_pool import DaemonThreadPool


def test_submit_returns_future_with_result():
    pool = DaemonThreadPool(max_workers=2, thread_name_prefix="TestSubmit")
    try:
        fut = pool.submit(lambda x: x + 1, 41)
        assert fut.result(timeout=5) == 42
    finally:
        pool.shutdown(wait=True, timeout=1)


def test_submit_propagates_exception():
    pool = DaemonThreadPool(max_workers=1, thread_name_prefix="TestRaise")
    try:
        fut = pool.submit(lambda: 1 / 0)
        with pytest.raises(ZeroDivisionError, match="division by zero"):
            fut.result(timeout=5)
    finally:
        pool.shutdown(wait=True, timeout=1)


def test_workers_are_daemons():
    pool = DaemonThreadPool(max_workers=3, thread_name_prefix="TestDaemon")
    try:
        assert all(t.daemon for t in pool._workers)
    finally:
        pool.shutdown(wait=True, timeout=1)


def test_submit_after_shutdown_raises():
    pool = DaemonThreadPool(max_workers=1, thread_name_prefix="TestPostShutdown")
    pool.shutdown(wait=True, timeout=1)
    with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
        pool.submit(lambda: None)


def test_shutdown_timeout_bounded_when_worker_blocked():
    pool = DaemonThreadPool(max_workers=1, thread_name_prefix="TestBlocked")
    started = threading.Event()

    def blocking():
        started.set()
        time.sleep(120)

    pool.submit(blocking)
    assert started.wait(timeout=5)
    t0 = time.monotonic()
    pool.shutdown(wait=True, timeout=0.5)
    elapsed = time.monotonic() - t0
    # `t.join(timeout=0.5)` returns after at most 0.5s even if the worker is stuck.
    assert elapsed < 2.0, f"shutdown took {elapsed:.2f}s, expected < 2.0s"


@pytest.mark.parametrize("max_workers", [0, -1])
def test_invalid_max_workers(max_workers):
    with pytest.raises(ValueError, match="max_workers must be positive"):
        DaemonThreadPool(max_workers=max_workers)


def test_shutdown_is_idempotent():
    pool = DaemonThreadPool(max_workers=2, thread_name_prefix="TestIdempotent")
    pool.shutdown(wait=True, timeout=1)
    pool.shutdown(wait=True, timeout=1)


def test_multiple_submits_in_flight():
    pool = DaemonThreadPool(max_workers=3, thread_name_prefix="TestParallel")
    try:
        futs = [pool.submit(lambda i=i: i * 2) for i in range(20)]
        results = sorted(f.result(timeout=5) for f in futs)
        assert results == [i * 2 for i in range(20)]
    finally:
        pool.shutdown(wait=True, timeout=2)
