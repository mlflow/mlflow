import contextvars
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor

import pytest

from mlflow.genai.evaluation.rate_limiter import NoOpRateLimiter, RPSRateLimiter
from mlflow.genai.scorers._scorer_execution import (
    RequestLimitedScorerExecution,
    get_scorer_execution,
    scorer_execution_context,
)
from mlflow.genai.scorers.base import scorer
from mlflow.utils.timeout import MlflowTimeoutError


def test_overlapping_admission_waits_do_not_extend_time_for_active_judge():
    now = 0.0
    permits = threading.Semaphore(0)
    waiting = threading.Barrier(3)
    active = threading.Event()
    finish = threading.Event()
    finished = threading.Event()

    class Budget(NoOpRateLimiter):
        def acquire(self, *, cancel_event=None):
            waiting.wait(timeout=5)
            assert permits.acquire(timeout=5)

    execution = RequestLimitedScorerExecution(Budget(), clock=lambda: now)

    @scorer(timeout=1)
    def judge():
        def chunk():
            with execution.worker():
                execution.acquire()
                active.set()
                assert finish.wait(5)

        try:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="TestChunk") as pool:
                futures = [pool.submit(contextvars.copy_context().run, chunk) for _ in range(2)]
                with execution.waiting_for_workers():
                    for future in futures:
                        future.result()
            return True
        finally:
            finished.set()

    def invoke():
        with scorer_execution_context(execution):
            return judge.run()

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="TestScorer") as pool:
        future = pool.submit(invoke)
        try:
            waiting.wait(timeout=5)
            # Ensure the parent has joined its workers before advancing virtual time.
            with execution._condition:
                assert execution._condition.wait_for(lambda: execution._running == 0, timeout=5)
            now = 100
            permits.release()
            assert active.wait(5)
            assert not future.done()
            now = 101.1
            execution.notify_completion()
            with pytest.raises(MlflowTimeoutError, match="timed out"):
                future.result(timeout=2)
        finally:
            permits.release(2)
            finish.set()
            assert finished.wait(5)


def test_cancellation_interrupts_admission_without_consuming_capacity():
    entered_wait = threading.Event()

    class Cancellation(threading.Event):
        def wait(self, timeout=None):
            entered_wait.set()
            return super().wait(timeout)

    cancelled = Cancellation()
    limiter = RPSRateLimiter(0.01)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="TestAdmission") as pool:
        future = pool.submit(limiter.acquire, cancel_event=cancelled)
        assert entered_wait.wait(2)
        cancelled.set()
        with pytest.raises(CancelledError, match="admission was cancelled"):
            future.result(timeout=2)

    limiter = RPSRateLimiter(1)
    with pytest.raises(CancelledError, match="admission was cancelled"):
        limiter.acquire(cancel_event=cancelled)
    # Cancellation before admission leaves the initial token available.
    limiter.acquire()


def test_each_scorer_attempt_gets_a_fresh_execution_deadline():
    execution = RequestLimitedScorerExecution(NoOpRateLimiter())

    @scorer(timeout=1)
    def judge():
        time.sleep(0.6)
        return True

    with scorer_execution_context(execution):
        assert judge.run() is True
        assert judge.run() is True
    assert get_scorer_execution() is None
    with pytest.raises(CancelledError, match="execution was cancelled"):
        execution.acquire()


def test_cancellation_interrupts_retry_backoff():
    execution = RequestLimitedScorerExecution(NoOpRateLimiter())
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="TestRetry") as pool:
        future = pool.submit(execution.sleep, 600)
        execution.cancel()
        with pytest.raises(CancelledError, match="execution was cancelled"):
            future.result(timeout=2)
