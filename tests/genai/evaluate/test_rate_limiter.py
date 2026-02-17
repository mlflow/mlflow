import pytest

from mlflow.genai.evaluation.rate_limiter import (
    NoOpRateLimiter,
    RPSRateLimiter,
    call_with_retry,
    eval_retry_context,
    is_rate_limit_error,
)
from mlflow.genai.judges.adapters.litellm_adapter import disable_litellm_rate_limit_retries
from mlflow.utils.rest_utils import is_429_retry_disabled


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


# ── Token bucket tests ──


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


# ── is_rate_limit_error tests ──


class _FakeRateLimitError(Exception):
    pass


_FakeRateLimitError.__name__ = "RateLimitError"


class _FakeStatusCodeError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


class _FakeResponseError(Exception):
    def __init__(self, status_code):
        self.response = type("R", (), {"status_code": status_code})()
        super().__init__(f"HTTP {status_code}")


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_FakeRateLimitError("rate limit"), True),
        (_FakeStatusCodeError(429), True),
        (_FakeResponseError(429), True),
        (Exception("Error 429: too many requests"), True),
        (Exception("rate limit exceeded"), True),
        (_FakeStatusCodeError(500), False),
        (_FakeResponseError(500), False),
        (Exception("something else entirely"), False),
        (ValueError("bad value"), False),
    ],
)
def test_is_rate_limit_error(exc, expected):
    assert is_rate_limit_error(exc) == expected


# ── AIMD tests ──


def test_throttle_halves_rate():
    clock = FakeClock()
    limiter = RPSRateLimiter(10.0, adaptive=True, clock=clock.monotonic, sleep=clock.sleep)

    limiter.report_throttle()
    assert limiter._rps == pytest.approx(5.0)


def test_throttle_respects_floor():
    clock = FakeClock()
    limiter = RPSRateLimiter(2.0, adaptive=True, clock=clock.monotonic, sleep=clock.sleep)

    # First throttle: 2.0 * 0.5 = 1.0
    limiter.report_throttle()
    assert limiter._rps == pytest.approx(1.0)

    # Second throttle (after cooldown): should stay at floor 1.0
    clock._now += 10.0
    limiter.report_throttle()
    assert limiter._rps == pytest.approx(1.0)


def test_throttle_cooldown_coalesces_rapid_signals():
    clock = FakeClock()
    limiter = RPSRateLimiter(10.0, adaptive=True, clock=clock.monotonic, sleep=clock.sleep)

    limiter.report_throttle()
    assert limiter._rps == pytest.approx(5.0)

    # Within cooldown window — should be ignored
    clock._now += 1.0
    limiter.report_throttle()
    assert limiter._rps == pytest.approx(5.0)

    # After cooldown — should take effect
    clock._now += 10.0
    limiter.report_throttle()
    assert limiter._rps == pytest.approx(2.5)


def test_success_restores_rate():
    clock = FakeClock()
    limiter = RPSRateLimiter(10.0, adaptive=True, clock=clock.monotonic, sleep=clock.sleep)

    limiter.report_throttle()
    assert limiter._rps == pytest.approx(5.0)

    # Repeatedly report success — rate should climb back past initial
    for _ in range(100):
        limiter.report_success()

    assert limiter._rps > 10.0


@pytest.mark.parametrize(
    ("multiplier", "expected_ceiling"),
    [(5.0, 50.0), (3.0, 30.0)],
)
def test_success_climbs_to_multiplier_ceiling(multiplier, expected_ceiling):
    clock = FakeClock()
    limiter = RPSRateLimiter(
        10.0,
        adaptive=True,
        max_rps_multiplier=multiplier,
        clock=clock.monotonic,
        sleep=clock.sleep,
    )
    for _ in range(10000):
        limiter.report_success()
    assert limiter._rps == pytest.approx(expected_ceiling)


def test_adaptive_false_ignores_throttle_and_success():
    clock = FakeClock()
    limiter = RPSRateLimiter(10.0, adaptive=False, clock=clock.monotonic, sleep=clock.sleep)

    limiter.report_throttle()
    assert limiter._rps == pytest.approx(10.0)

    limiter.report_success()
    assert limiter._rps == pytest.approx(10.0)


# ── call_with_retry tests ──


def test_call_with_retry_success():
    sleep_calls = []
    limiter = NoOpRateLimiter()
    result = call_with_retry(lambda: 42, limiter, max_retries=3, sleep=sleep_calls.append)
    assert result == 42
    assert sleep_calls == []


def test_call_with_retry_retries_on_429_then_succeeds():
    sleep_calls = []
    limiter = NoOpRateLimiter()
    attempts = []

    def flaky_fn():
        attempts.append(1)
        if len(attempts) < 3:
            raise _FakeRateLimitError("rate limited")
        return "ok"

    result = call_with_retry(flaky_fn, limiter, max_retries=3, sleep=sleep_calls.append)
    assert result == "ok"
    assert len(attempts) == 3
    # Two retries with exponential backoff: 2^0=1, 2^1=2
    assert sleep_calls == [1, 2]


def test_call_with_retry_non_429_propagates_immediately():
    sleep_calls = []
    limiter = NoOpRateLimiter()

    def always_raises():
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        call_with_retry(always_raises, limiter, max_retries=3, sleep=sleep_calls.append)
    assert sleep_calls == []


def test_call_with_retry_exhausted_retries():
    sleep_calls = []
    limiter = NoOpRateLimiter()

    def always_rate_limited():
        raise _FakeRateLimitError("rate limited")

    with pytest.raises(_FakeRateLimitError, match="rate limited"):
        call_with_retry(always_rate_limited, limiter, max_retries=2, sleep=sleep_calls.append)
    # 3 attempts total (initial + 2 retries), 2 sleeps
    assert len(sleep_calls) == 2


def test_call_with_retry_reports_throttle_and_success():
    clock = FakeClock()
    limiter = RPSRateLimiter(10.0, adaptive=True, clock=clock.monotonic, sleep=clock.sleep)
    attempts = []

    def flaky_fn():
        attempts.append(1)
        if len(attempts) == 1:
            raise _FakeRateLimitError("rate limited")
        return "ok"

    result = call_with_retry(flaky_fn, limiter, max_retries=3, sleep=clock.sleep)
    assert result == "ok"
    # After throttle: 10.0 * 0.5 = 5.0, then success bumps it back up slightly
    assert limiter._rps < 10.0


# ── eval_retry_context tests ──


def _retry_flags_active():
    """Check that both downstream retry-suppression flags are set."""
    litellm_flag = disable_litellm_rate_limit_retries()
    return litellm_flag.get() and is_429_retry_disabled()


def test_eval_retry_context_sets_and_resets():
    assert not _retry_flags_active()

    with eval_retry_context():
        assert _retry_flags_active()

    assert not _retry_flags_active()


def test_eval_retry_context_nests():
    assert not _retry_flags_active()

    with eval_retry_context():
        assert _retry_flags_active()
        with eval_retry_context():
            assert _retry_flags_active()
        assert _retry_flags_active()

    assert not _retry_flags_active()
