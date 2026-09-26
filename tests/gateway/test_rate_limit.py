import itertools
import threading
from unittest import mock

import pytest
from fastapi import HTTPException

from mlflow.gateway.rate_limit import (
    RATE_LIMIT_WINDOW_SECONDS,
    check_rate_limit,
    reset_rate_limiters,
)
from mlflow.store.tracking.gateway.entities import GatewayEndpointConfig


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    fake = _FakeClock()
    reset_rate_limiters()
    with mock.patch("mlflow.gateway.rate_limit.time.monotonic", side_effect=fake):
        yield fake
    reset_rate_limiters()


def _config(name="my-endpoint", calls_per_minute=None):
    return GatewayEndpointConfig(
        endpoint_id=f"e-{name}",
        endpoint_name=name,
        models=[],
        calls_per_minute=calls_per_minute,
    )


@pytest.mark.parametrize("limit", [None, 0])
def test_no_limit_configured_is_a_noop(clock, limit):
    config = _config(calls_per_minute=limit)
    for _ in range(1000):
        check_rate_limit(config)


def test_calls_within_limit_are_allowed(clock):
    config = _config(calls_per_minute=3)
    for _ in range(3):
        check_rate_limit(config)


def test_call_over_limit_raises_429_with_retry_after(clock):
    config = _config(calls_per_minute=2)
    check_rate_limit(config)
    clock.advance(10)
    check_rate_limit(config)

    with pytest.raises(HTTPException, match="Rate limit exceeded") as exc_info:
        check_rate_limit(config)

    assert exc_info.value.status_code == 429
    # The oldest call happened 10s ago, so the window frees a slot 50s from now.
    assert exc_info.value.headers["Retry-After"] == "50"
    assert "my-endpoint" in exc_info.value.detail
    assert "2 calls per minute" in exc_info.value.detail


def test_rejected_calls_do_not_extend_the_window(clock):
    config = _config(calls_per_minute=1)
    check_rate_limit(config)

    clock.advance(30)
    with pytest.raises(HTTPException, match="Rate limit exceeded"):
        check_rate_limit(config)

    # The rejected call must not be recorded, so the window still clears 60s after
    # the single successful call rather than 60s after the rejection.
    clock.advance(RATE_LIMIT_WINDOW_SECONDS - 30)
    check_rate_limit(config)


def test_window_slides_rather_than_resetting(clock):
    config = _config(calls_per_minute=2)
    check_rate_limit(config)
    clock.advance(40)
    check_rate_limit(config)

    clock.advance(21)
    # The first call has aged out, the second has not: exactly one slot is free.
    check_rate_limit(config)
    with pytest.raises(HTTPException, match="Rate limit exceeded"):
        check_rate_limit(config)


@pytest.mark.parametrize("elapsed", [RATE_LIMIT_WINDOW_SECONDS, RATE_LIMIT_WINDOW_SECONDS + 1])
def test_limit_resets_after_full_window(clock, elapsed):
    config = _config(calls_per_minute=2)
    for _ in range(2):
        check_rate_limit(config)
    with pytest.raises(HTTPException, match="Rate limit exceeded"):
        check_rate_limit(config)

    clock.advance(elapsed)
    for _ in range(2):
        check_rate_limit(config)


def test_limit_not_reached_just_before_window_expiry(clock):
    config = _config(calls_per_minute=1)
    check_rate_limit(config)
    clock.advance(RATE_LIMIT_WINDOW_SECONDS - 1)
    with pytest.raises(HTTPException, match="Rate limit exceeded") as exc_info:
        check_rate_limit(config)
    assert exc_info.value.headers["Retry-After"] == "1"


def test_windows_are_isolated_per_endpoint(clock):
    first = _config("endpoint-a", calls_per_minute=1)
    second = _config("endpoint-b", calls_per_minute=1)

    check_rate_limit(first)
    check_rate_limit(second)

    with pytest.raises(HTTPException, match="Rate limit exceeded"):
        check_rate_limit(first)
    with pytest.raises(HTTPException, match="Rate limit exceeded"):
        check_rate_limit(second)


def test_concurrent_calls_admit_exactly_the_limit():
    reset_rate_limiters()
    limit = 20
    config = _config("concurrent-endpoint", calls_per_minute=limit)
    allowed = itertools.count()
    rejected = itertools.count()
    barrier = threading.Barrier(50)

    def _call():
        barrier.wait()
        try:
            check_rate_limit(config)
        except HTTPException:
            next(rejected)
        else:
            next(allowed)

    threads = [threading.Thread(target=_call) for _ in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert next(allowed) == limit
    assert next(rejected) == 50 - limit
    reset_rate_limiters()
