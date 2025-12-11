# Based ons: https://github.com/openai/openai-cookbook/blob/6df6ceff470eeba26a56de131254e775292eac22/examples/api_request_parallel_processor.py
# Several changes were made to make it work with MLflow.

"""
API REQUEST PARALLEL PROCESSOR

Using the OpenAI API to process lots of text quickly takes some care.
If you trickle in a million API requests one by one, they'll take days to complete.
If you flood a million API requests in parallel, they'll exceed the rate limits and fail with
errors. To maximize throughput, parallel requests need to be throttled to stay under rate limits.

This script parallelizes requests to the OpenAI API

Features:
- Makes requests concurrently, to maximize throughput
- Retries failed requests up to {max_attempts} times, to avoid missing data
- Logs errors, to diagnose problems with requests
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable

import mlflow

_logger = logging.getLogger(__name__)


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    lock: threading.Lock = threading.Lock()
    error = None

    def start_task(self):
        with self.lock:
            self.num_tasks_started += 1
            self.num_tasks_in_progress += 1

    def complete_task(self, *, success: bool):
        with self.lock:
            self.num_tasks_in_progress -= 1
            if success:
                self.num_tasks_succeeded += 1
            else:
                self.num_tasks_failed += 1

    def increment_num_rate_limit_errors(self):
        with self.lock:
            self.num_rate_limit_errors += 1


def call_api(
    index: int,
    results: list[tuple[int, Any]],
    task: Callable[[], Any],
    status_tracker: StatusTracker,
):
    import openai

    status_tracker.start_task()
    try:
        result = task()
        _logger.debug(f"Request #{index} succeeded")
        status_tracker.complete_task(success=True)
        results.append((index, result))
    except openai.RateLimitError as e:
        status_tracker.complete_task(success=False)
        _logger.debug(f"Request #{index} failed with: {e}")
        status_tracker.increment_num_rate_limit_errors()
        status_tracker.error = mlflow.MlflowException(
            f"Request #{index} failed with rate limit: {e}."
        )
    except Exception as e:
        status_tracker.complete_task(success=False)
        _logger.debug(f"Request #{index} failed with: {e}")
        status_tracker.error = mlflow.MlflowException(
            f"Request #{index} failed with: {e.__cause__}"
        )


def process_api_requests(
    request_tasks: list[Callable[[], Any]],
    max_workers: int = 10,
):
    """Processes API requests in parallel"""
    # initialize trackers
    status_tracker = StatusTracker()  # single instance to track a collection of variables

    results: list[tuple[int, Any]] = []
    request_tasks_iter = enumerate(request_tasks)
    _logger.debug(f"Request pool executor will run {len(request_tasks)} requests")
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="MlflowOpenAiApi"
    ) as executor:
        futures = [
            executor.submit(
                call_api,
                index=index,
                task=task,
                results=results,
                status_tracker=status_tracker,
            )
            for index, task in request_tasks_iter
        ]
        wait(futures, return_when=FIRST_EXCEPTION)

    # after finishing, log final status
    if status_tracker.num_tasks_failed > 0:
        if status_tracker.num_tasks_failed == 1:
            raise status_tracker.error
        raise mlflow.MlflowException(
            f"{status_tracker.num_tasks_failed} tasks failed. See logs for details."
        )
    if status_tracker.num_rate_limit_errors > 0:
        _logger.debug(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. "
            "Consider running at a lower rate."
        )

    return [res for _, res in sorted(results)]
