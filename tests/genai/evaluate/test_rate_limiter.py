import pytest

from mlflow.genai.evaluation.harness import _make_rate_limiter
from mlflow.genai.evaluation.rate_limiter import NoOpRateLimiter, RPSRateLimiter


class FakeClock:
    """Deterministic clock for testing. sleep() advances the clock by the requested amount.

    Thread safety is not needed here because RPSRateLimiter's internal lock serializes
    all calls to clock() and sleep() — they are never called concurrently for a given limiter.
    """

    def __init__(self):
        self._now = 0.0
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self._now += seconds


def test_invalid_rate_raises():
    with pytest.raises(ValueError, match="must be positive"):
        RPSRateLimiter(0)
    with pytest.raises(ValueError, match="must be positive"):
        RPSRateLimiter(-1)


def test_burst_tokens_consumed_without_sleeping():
    clock = FakeClock()
    limiter = RPSRateLimiter(5, clock=clock.monotonic, sleep=clock.sleep)

    for _ in range(5):
        limiter.acquire()

    assert clock.sleep_calls == []


def test_sleep_called_when_tokens_exhausted():
    clock = FakeClock()
    limiter = RPSRateLimiter(5, clock=clock.monotonic, sleep=clock.sleep)

    for _ in range(5):
        limiter.acquire()

    limiter.acquire()
    assert len(clock.sleep_calls) == 1
    assert clock.sleep_calls[0] == pytest.approx(0.2, abs=0.01)


def test_total_sleep_for_sustained_rate():
    clock = FakeClock()
    limiter = RPSRateLimiter(10, clock=clock.monotonic, sleep=clock.sleep)

    for _ in range(20):
        limiter.acquire()

    total_sleep = sum(clock.sleep_calls)
    assert total_sleep == pytest.approx(1.0, abs=0.01)


def test_tokens_refill_after_idle():
    clock = FakeClock()
    limiter = RPSRateLimiter(10, clock=clock.monotonic, sleep=clock.sleep)

    for _ in range(10):
        limiter.acquire()

    clock._now += 1.0

    sleep_before = len(clock.sleep_calls)
    for _ in range(10):
        limiter.acquire()

    assert clock.sleep_calls[sleep_before:] == []


def test_partial_refill():
    clock = FakeClock()
    limiter = RPSRateLimiter(10, clock=clock.monotonic, sleep=clock.sleep)

    for _ in range(10):
        limiter.acquire()

    clock._now += 0.5

    sleep_before = len(clock.sleep_calls)
    for _ in range(5):
        limiter.acquire()

    assert clock.sleep_calls[sleep_before:] == []

    limiter.acquire()
    assert len(clock.sleep_calls) == sleep_before + 1


def test_noop_acquire_does_nothing():
    limiter = NoOpRateLimiter()
    for _ in range(1000):
        limiter.acquire()


def test_make_rate_limiter_positive_rate():
    assert isinstance(_make_rate_limiter(10.0), RPSRateLimiter)


def test_make_rate_limiter_zero_returns_noop():
    assert isinstance(_make_rate_limiter(0.0), NoOpRateLimiter)


def test_make_rate_limiter_none_returns_noop():
    assert isinstance(_make_rate_limiter(None), NoOpRateLimiter)
