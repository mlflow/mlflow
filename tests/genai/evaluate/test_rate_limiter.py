import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest

import mlflow.genai.judges.adapters.rate_limit_retry_adapters  # noqa: F401
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.harness import (
    AUTO_INITIAL_RPS,
    _compute_eval_scores,
    _make_rate_limiter,
    _parse_rate_limit,
)
from mlflow.genai.evaluation.rate_limiter import (
    NoOpRateLimiter,
    RPSRateLimiter,
    call_with_retry,
    eval_retry_context,
    is_rate_limit_error,
)
from mlflow.genai.judges.adapters.litellm_adapter import (
    _get_litellm_retry_policy,
    disable_litellm_rate_limit_retries,
    is_litellm_rate_limit_retries_disabled,
)
from mlflow.genai.judges.adapters.rate_limit_retry_adapters import RateLimitRetryAdapter
from mlflow.genai.scorers.base import scorer
from mlflow.utils.rest_utils import disable_429_retry, is_429_retry_disabled


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


def test_sub_one_rps_can_acquire():
    # rps < 1.0 was broken: _max_tokens was set to rps, so the bucket could never
    # accumulate a full token and acquire() would loop forever.
    clock = FakeClock()
    limiter = RPSRateLimiter(0.5, clock=clock.monotonic, sleep=clock.sleep)  # 1 req / 2s

    # First acquire: initial tokens=0.5, sleeps 1s to reach 1.0, then succeeds.
    # Second acquire: tokens=0, sleeps 2s to reach 1.0, then succeeds.
    limiter.acquire()
    limiter.acquire()

    total_sleep = sum(clock.sleep_calls)
    assert total_sleep == pytest.approx(3.0, abs=0.1)


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


# ── _make_rate_limiter / _parse_rate_limit tests ──


def test_make_rate_limiter_positive_rate():
    assert isinstance(_make_rate_limiter(10.0), RPSRateLimiter)


def test_make_rate_limiter_zero_returns_noop():
    assert isinstance(_make_rate_limiter(0.0), NoOpRateLimiter)


def test_make_rate_limiter_none_returns_noop():
    assert isinstance(_make_rate_limiter(None), NoOpRateLimiter)


def test_make_rate_limiter_adaptive():
    limiter = _make_rate_limiter(10.0, adaptive=True)
    assert isinstance(limiter, RPSRateLimiter)
    assert limiter._adaptive is True


@pytest.mark.parametrize(
    ("raw", "expected_rps", "expected_adaptive"),
    [
        ("auto", AUTO_INITIAL_RPS, True),
        ("AUTO", AUTO_INITIAL_RPS, True),
        (" Auto ", AUTO_INITIAL_RPS, True),
        ("25", 25.0, False),
        ("0", None, False),
        (None, None, False),
    ],
)
def test_parse_rate_limit(raw, expected_rps, expected_adaptive):
    rps, adaptive = _parse_rate_limit(raw)
    assert rps == expected_rps
    assert adaptive == expected_adaptive


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

# Both built-in adapters forced active so tests don't depend on litellm
# being installed or a Databricks tracking URI being configured.
_BOTH_ADAPTERS_ACTIVE = [
    RateLimitRetryAdapter(
        name="litellm",
        is_adapter_active=lambda: True,
        disable_internal_retries=disable_litellm_rate_limit_retries,
    ),
    RateLimitRetryAdapter(
        name="databricks-sdk",
        is_adapter_active=lambda: True,
        disable_internal_retries=disable_429_retry,
    ),
]


def _retry_flags_active():
    """Check that both downstream retry-suppression flags are set."""
    return is_litellm_rate_limit_retries_disabled() and is_429_retry_disabled()


def test_eval_retry_context_sets_and_resets():
    assert not _retry_flags_active()

    with patch(
        "mlflow.genai.judges.adapters.rate_limit_retry_adapters._RETRY_ADAPTER_REGISTRY",
        _BOTH_ADAPTERS_ACTIVE,
    ):
        with eval_retry_context():
            assert _retry_flags_active()

    assert not _retry_flags_active()


def test_eval_retry_context_nests():
    assert not _retry_flags_active()

    with patch(
        "mlflow.genai.judges.adapters.rate_limit_retry_adapters._RETRY_ADAPTER_REGISTRY",
        _BOTH_ADAPTERS_ACTIVE,
    ):
        with eval_retry_context():
            assert _retry_flags_active()
            with eval_retry_context():
                assert _retry_flags_active()
            assert _retry_flags_active()

    assert not _retry_flags_active()


def test_litellm_retry_policy_disables_rate_limit_retries_when_flag_set():
    with eval_retry_context():
        policy = _get_litellm_retry_policy(3)
    assert policy.RateLimitErrorRetries == 0
    assert policy.TimeoutErrorRetries == 3


# ── contextvar propagation into the scorer thread pool ──


def _flags_in_current_thread() -> tuple[str, bool, bool]:
    return (
        threading.current_thread().name,
        is_litellm_rate_limit_retries_disabled(),
        is_429_retry_disabled(),
    )


def _run_scores_capturing_flags(num_scorers):
    """Run scorers through _compute_eval_scores inside eval_retry_context().

    Returns the (thread_name, litellm_disabled, http_429_disabled) tuple each
    scorer observed from inside its worker thread. A barrier forces every scorer
    to run concurrently so the parallel path is genuinely exercised.
    """
    captured: list[tuple[str, bool, bool]] = []
    lock = threading.Lock()
    barrier = threading.Barrier(num_scorers)

    def make_probe(name):
        @scorer(name=name)
        def probe(outputs):
            barrier.wait()
            observed = _flags_in_current_thread()
            with lock:
                captured.append(observed)
            return 1.0

        return probe

    scorer_objs = [make_probe(f"probe_{i}") for i in range(num_scorers)]
    item = EvalItem(request_id="r1", inputs={}, outputs="x", expectations={})

    # Force both adapters active so the flags are set regardless of litellm being
    # installed or the tracking URI, isolating the test to context propagation.
    with patch(
        "mlflow.genai.judges.adapters.rate_limit_retry_adapters._RETRY_ADAPTER_REGISTRY",
        _BOTH_ADAPTERS_ACTIVE,
    ):
        with eval_retry_context():
            _compute_eval_scores(eval_item=item, scorers=scorer_objs, max_retries=0)
    return captured


def test_scorer_thread_sees_retry_flags_single():
    captured = _run_scores_capturing_flags(num_scorers=1)

    assert len(captured) == 1
    thread_name, litellm_disabled, http_disabled = captured[0]
    assert thread_name != threading.current_thread().name
    assert thread_name.startswith("MlflowGenAIEvalScorer")
    assert litellm_disabled is True
    assert http_disabled is True


def test_scorer_threads_see_retry_flags_under_concurrency():
    captured = _run_scores_capturing_flags(num_scorers=5)

    assert len(captured) == 5
    worker_threads = {name for name, _, _ in captured}
    assert len(worker_threads) > 1
    assert all(name.startswith("MlflowGenAIEvalScorer") for name, _, _ in captured)
    assert all(litellm and http for _, litellm, http in captured)


def test_retry_flags_reset_in_caller_thread_after_scoring():
    assert not _retry_flags_active()

    _run_scores_capturing_flags(num_scorers=2)

    # The caller's thread context is unchanged after the with-block exits.
    assert not _retry_flags_active()


# The following three tests pin the raw contextvars semantics the fix relies on,
# independent of MLflow, so a future refactor that reintroduces the bug fails loudly
# with an explanatory test name rather than a silent behavioral regression.

_probe_cv: contextvars.ContextVar[str] = contextvars.ContextVar("_probe_cv", default="DEFAULT")


def test_threadpool_does_not_inherit_context_by_default():
    _probe_cv.set("SET")
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cv-test") as ex:
        observed = list(ex.map(lambda _: _probe_cv.get(), range(2)))
    # This is the bug: plain submit loses the caller's contextvars.
    assert observed == ["DEFAULT", "DEFAULT"]


def test_single_shared_context_cannot_be_entered_concurrently():
    _probe_cv.set("SET")
    shared = contextvars.copy_context()
    hold = threading.Event()

    def work(_):
        # Hold the context open so the other submission collides trying to enter
        # it. No barrier: the colliding worker never runs work(), so a rendezvous
        # would deadlock.
        hold.wait(timeout=2)
        return _probe_cv.get()

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cv-test") as ex:
        futures = [ex.submit(shared.run, work, i) for i in range(2)]
        errors = []
        try:
            for f in as_completed(futures):
                try:
                    f.result()
                except RuntimeError as e:
                    errors.append(str(e))
                    hold.set()  # release the other worker so the pool can drain
        finally:
            hold.set()
    assert any("already entered" in e for e in errors)


def test_fresh_context_copy_per_submit_propagates_value():
    _probe_cv.set("SET")
    started = threading.Barrier(3)

    def work(_):
        started.wait()
        return _probe_cv.get()

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="cv-test") as ex:
        futures = [ex.submit(contextvars.copy_context().run, work, i) for i in range(3)]
        observed = [f.result() for f in as_completed(futures)]
    # A fresh copy per submit both carries the value and tolerates concurrency.
    assert observed == ["SET", "SET", "SET"]
