from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow.genai.evaluation.rate_limiter import RateLimiter


class RequestLimitedScorerExecution:
    """Per-invocation deadline and cancellation, sharing an evaluation's request budget."""

    def __init__(self, rate_limiter: RateLimiter, clock=time.monotonic):
        self._rate_limiter = rate_limiter
        self._clock = clock
        self._condition = threading.Condition()
        self._cancelled = threading.Event()
        self._running = 1  # The scorer thread; fanout workers register separately.
        self._admission_waiters = 0
        self._elapsed = 0.0
        self._updated_at = clock()

    def _account_time(self):
        now = self._clock()
        # Exclude only wall time during which *all* execution is waiting for capacity.
        # Summing individual workers' waits would extend the timeout multiple times.
        if self._running or not self._admission_waiters:
            self._elapsed += now - self._updated_at
        self._updated_at = now

    @contextmanager
    def _activity(self, running: int, admission_waiters: int = 0):
        with self._condition:
            self._account_time()
            self._running += running
            self._admission_waiters += admission_waiters
            self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._account_time()
                self._running -= running
                self._admission_waiters -= admission_waiters
                self._condition.notify_all()

    @contextmanager
    def worker(self):
        with self._activity(running=1):
            self.check_cancelled()
            yield

    def waiting_for_workers(self):
        return self._activity(running=-1)

    def acquire(self):
        self.check_cancelled()
        with self._activity(running=-1, admission_waiters=1):
            self._rate_limiter.acquire(cancel_event=self._cancelled)
        self.check_cancelled()

    def report_success(self):
        self._rate_limiter.report_success()

    def report_throttle(self):
        self._rate_limiter.report_throttle()

    def check_cancelled(self):
        if self._cancelled.is_set():
            raise CancelledError("Scorer execution was cancelled")

    def sleep(self, seconds: float):
        # Retry backoff still consumes execution time, but can be interrupted on timeout.
        self._cancelled.wait(seconds)
        self.check_cancelled()

    def cancel(self):
        self._cancelled.set()
        self.notify_completion()

    def start(self):
        with self._condition:
            self._elapsed = 0.0
            self._updated_at = self._clock()

    def notify_completion(self):
        with self._condition:
            self._condition.notify_all()

    def wait(self, done: threading.Event, timeout: float) -> bool:
        with self._condition:
            while not done.is_set():
                self._account_time()
                remaining = timeout - self._elapsed
                if remaining <= 0 or self._cancelled.is_set():
                    self.cancel()
                    return False
                paused = self._running == 0 and self._admission_waiters > 0
                self._condition.wait(None if paused else min(remaining, threading.TIMEOUT_MAX))
            return True


_scorer_execution: ContextVar[RequestLimitedScorerExecution | None] = ContextVar(
    "request_limited_scorer_execution", default=None
)


def get_scorer_execution() -> RequestLimitedScorerExecution | None:
    return _scorer_execution.get()


@contextmanager
def scorer_execution_context(execution: RequestLimitedScorerExecution):
    token = _scorer_execution.set(execution)
    try:
        yield
    finally:
        execution.cancel()
        _scorer_execution.reset(token)
