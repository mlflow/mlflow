"""Thread-safe rate limiters for evaluation harness."""

from __future__ import annotations

import abc
import contextlib
import logging
import threading
import time
from typing import Callable

_logger = logging.getLogger(__name__)


@contextlib.contextmanager
def eval_retry_context():
    """Disable downstream 429 retries so errors bubble up to call_with_retry().

    Sets flags on both the HTTP-layer retry (rest_utils) and the litellm
    adapter so that rate-limit errors propagate to the evaluate pipeline's
    own retry/AIMD logic.
    """
    from mlflow.genai.judges.adapters.litellm_adapter import disable_litellm_rate_limit_retries
    from mlflow.utils.rest_utils import disable_429_retry

    litellm_flag = disable_litellm_rate_limit_retries()
    litellm_token = litellm_flag.set(True)
    try:
        with disable_429_retry():
            yield
    finally:
        litellm_flag.reset(litellm_token)


def is_rate_limit_error(exc: BaseException) -> bool:
    # 1. Known exception types from popular LLM libraries
    type_name = type(exc).__name__
    if type_name == "RateLimitError":
        return True

    # 2. Check .response.status_code or .status_code (httpx / requests style)
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
    if status == 429:
        return True

    # 3. Fallback: string matching
    exc_str = str(exc).lower()
    if "429" in exc_str or "rate limit" in exc_str:
        return True

    return False


class RateLimiter(abc.ABC):
    @abc.abstractmethod
    def acquire(self) -> None: ...

    def report_throttle(self) -> None:
        """Called when a 429 / rate-limit error is observed. No-op by default."""

    def report_success(self) -> None:
        """Called on a successful request. No-op by default."""

    @property
    def current_rps(self) -> float | None:
        """Current requests-per-second, or None if not rate-limited."""
        return None


class RPSRateLimiter(RateLimiter):
    """Thread-safe token-bucket rate limiter with optional AIMD adaptation.

    Each acquire() consumes one token and blocks until a token is available.
    Tokens refill at the configured rate, with a burst capacity of one second.

    When ``adaptive=True``, report_throttle() multiplicatively decreases the
    rate and report_success() additively increases it (AIMD).
    """

    def __init__(
        self,
        requests_per_second: float,
        adaptive: bool = False,
        max_rps_multiplier: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._rps = requests_per_second
        self._initial_rps = requests_per_second
        self._max_tokens = requests_per_second
        self._tokens = requests_per_second
        self._clock = clock
        self._sleep = sleep
        self._last_refill = clock()
        self._lock = threading.Lock()

        # AIMD parameters
        self._adaptive = adaptive
        self._beta = 0.5  # multiplicative decrease factor
        self._alpha = 1.0  # additive increase per success
        self._min_rps = 1.0  # floor for rate reduction
        self._max_rps = max_rps_multiplier * requests_per_second  # ceiling for additive increase
        self._throttle_cooldown = 5.0  # seconds between consecutive decreases
        self._last_throttle_time = -float("inf")

    def acquire(self) -> None:
        thread = threading.current_thread().name
        _logger.debug(
            f"[{thread}] acquire() called — rps={self._rps:.1f} tokens={self._tokens:.2f}"
        )
        while True:
            with self._lock:
                now = self._clock()
                elapsed = now - self._last_refill
                self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rps)
                self._last_refill = now

                # Epsilon avoids floating-point drift (e.g. 0.1*10 = 0.9999...98)
                if self._tokens >= 1.0 - 1e-9:
                    self._tokens -= 1.0
                    return

                wait_time = (1.0 - self._tokens) / self._rps

            _logger.debug(f"[{thread}] acquire() sleeping {wait_time:.3f}s")
            self._sleep(wait_time)

    def report_throttle(self) -> None:
        if not self._adaptive:
            return
        with self._lock:
            now = self._clock()
            if now - self._last_throttle_time < self._throttle_cooldown:
                return
            self._last_throttle_time = now
            old_rps = self._rps
            self._rps = max(self._min_rps, self._rps * self._beta)
            self._max_tokens = self._rps
            # Clamp current tokens to new burst size
            self._tokens = min(self._tokens, self._max_tokens)
            _logger.info(
                f"Rate limit hit — reducing rate from {old_rps:.1f} to {self._rps:.1f} rps"
            )

    def report_success(self) -> None:
        if not self._adaptive:
            return
        with self._lock:
            old_rps = self._rps
            self._rps = min(self._max_rps, self._rps + self._alpha / self._rps)
            self._max_tokens = self._rps
            if self._rps > old_rps + 0.5:
                _logger.debug(f"Rate increased to {self._rps:.1f} rps")

    @property
    def current_rps(self) -> float | None:
        return self._rps


class NoOpRateLimiter(RateLimiter):
    def acquire(self) -> None:
        pass


def call_with_retry(
    fn: Callable[[], object],
    rate_limiter: RateLimiter,
    max_retries: int,
    sleep: Callable[[float], None] = time.sleep,
) -> object:
    last_exc = None
    for attempt in range(max_retries + 1):
        rate_limiter.acquire()
        try:
            result = fn()
        except Exception as e:
            if not is_rate_limit_error(e):
                raise
            last_exc = e
            rate_limiter.report_throttle()
            if attempt < max_retries:
                delay = min(2**attempt, 60)
                _logger.info(
                    f"Rate-limited (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay}s"
                )
                sleep(delay)
        else:
            rate_limiter.report_success()
            return result
    raise last_exc
