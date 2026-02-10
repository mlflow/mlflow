"""Thread-safe rate limiters for evaluation harness."""

from __future__ import annotations

import abc
import threading
import time
from typing import Callable


class RateLimiter(abc.ABC):
    @abc.abstractmethod
    def acquire(self) -> None: ...


class RPSRateLimiter(RateLimiter):
    """Thread-safe token-bucket rate limiter.

    Each acquire() consumes one token and blocks until a token is available.
    Tokens refill at the configured rate, with a burst capacity of one second.
    """

    def __init__(
        self,
        requests_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._rps = requests_per_second
        self._max_tokens = requests_per_second
        self._tokens = requests_per_second
        self._clock = clock
        self._sleep = sleep
        self._last_refill = clock()
        self._lock = threading.Lock()

    def acquire(self) -> None:
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

            self._sleep(wait_time)


class NoOpRateLimiter(RateLimiter):
    def acquire(self) -> None:
        pass
