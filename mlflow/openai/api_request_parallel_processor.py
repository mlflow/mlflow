# Based ons: https://github.com/openai/openai-cookbook/blob/6df6ceff470eeba26a56de131254e775292eac22/examples/api_request_parallel_processor.py
# Several changes were made to make it work with MLflow.

"""
API REQUEST PARALLEL PROCESSOR

Using the OpenAI API to process lots of text quickly takes some care.
If you trickle in a million API requests one by one, they'll take days to complete.
If you flood a million API requests in parallel, they'll exceed the rate limits and fail with
errors. To maximize throughput, parallel requests need to be throttled to stay under rate limits.

This script parallelizes requests to the OpenAI API while throttling to stay under rate limits.

Features:
- Streams requests from file, to avoid running out of memory for giant jobs
- Makes requests concurrently, to maximize throughput
- Throttles request and token usage, to stay under rate limits
- Retries failed requests up to {max_attempts} times, to avoid missing data
- Logs errors, to diagnose problems with requests
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict

import requests
import tiktoken

import mlflow
from mlflow.protos.databricks_pb2 import UNAUTHENTICATED
from mlflow.utils.openai_utils import _OAITokenHolder

_logger = logging.getLogger(__name__)


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits
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

    def increment_num_api_errors(self):
        with self.lock:
            self.num_api_errors += 1


@dataclass
class APIRequest:
    """
    Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API
    call.
    """

    index: int
    request_url: str
    request_json: dict
    token_consumption: int
    attempts_left: int
    results: list[tuple[int, any]]
    start_time: int
    last_log_time: int
    timeout: int = 60

    def call_api(
        self,
        retry_queue: queue.Queue,
        status_tracker: StatusTracker,
        headers: Dict[str, str],
    ):
        """Calls the OpenAI API and stores results."""
        _logger.debug(f"Request #{self.index} started")
        try:
            response = requests.post(
                url=self.request_url,
                headers=headers,
                json=self.request_json,
                timeout=self.timeout,
            )
            response_json = response.json()

            # Refer to error codes: https://platform.openai.com/docs/guides/error-codes/api-errors
            if "error" in response_json:
                error = response_json["error"]
                logging.warning(f"Request #{self.index} failed with error {error}")
                # Rate limit error
                if "Rate limit" in error.get("message", ""):
                    _logger.debug(f"Request #{self.index} failed with {error!r}")
                    current_time = time.time()
                    status_tracker.time_of_last_rate_limit_error = current_time
                    status_tracker.increment_num_rate_limit_errors()
                    # check time since first request, fail at 10 minutes
                    if current_time - self.start_time < 600:
                        if current_time - self.last_log_time > 60:
                            _logger.warning("Retrying for request failed with rate limit.")
                            self.last_log_time = current_time
                        retry_queue.put_nowait(self)
                    else:
                        _logger.warning("Request failed after retrying for 10 minutes.")
                        status_tracker.complete_task(success=False)
                        status_tracker.error = mlflow.MlflowException(
                            "Request failed after retrying for 10 minutes."
                        )
                # Other retryable errors
                elif response.status_code in [500, 503]:
                    _logger.debug(f"Request #{self.index} failed with {error!r}")
                    status_tracker.increment_num_api_errors()
                    if self.attempts_left > 0:
                        retry_queue.put_nowait(self)
                    else:
                        status_tracker.complete_task(success=False)
                        status_tracker.error = mlflow.MlflowException(
                            f"Request #{self.index} failed with {error!r}"
                        )
                # Unretryable errors
                else:
                    _logger.warning(f"Request #{self.index} failed with {error!r}")
                    status_tracker.increment_num_api_errors()
                    status_tracker.complete_task(success=False)
                    if response.status_code == 401:
                        status_tracker.error = mlflow.MlflowException(
                            f"Authentication Error for OpenAI. Error response:\n {error!r}",
                            error_code=UNAUTHENTICATED,
                        )
                    else:
                        status_tracker.error = mlflow.MlflowException(
                            f"Request #{self.index} to OpenAI failed. Error response: {error}"
                        )
            else:
                _logger.debug(f"Request #{self.index} succeeded")
                status_tracker.complete_task(success=True)
                self.results.append((self.index, response_json))
        except Exception as e:
            _logger.debug(f"Request #{self.index} failed with {e!r}")
            status_tracker.increment_num_api_errors()
            status_tracker.complete_task(success=False)
            status_tracker.error = e


def num_tokens_consumed_from_request(
    request_json: dict, request_url: str, token_encoding_name: str
):
    """
    Count the number of tokens in the request. Only supports completion and embedding requests.
    """
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if "completions" in request_url:
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if "chat/completions" in request_url:
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                return prompt_tokens + completion_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                return prompt_tokens + completion_tokens * len(prompt)
            else:
                raise TypeError(
                    "Expecting either string or list of strings for 'prompt' field in completion "
                    "request"
                )
    # if embeddings request, tokens = input tokens
    elif "embeddings" in request_url:
        inp = request_json["input"]
        if isinstance(inp, str):  # single input
            return len(encoding.encode(inp))
        elif isinstance(inp, list):  # multiple inputs
            return sum([len(encoding.encode(i)) for i in inp])
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'Support for "{request_url}" not implemented in this script')


def process_api_requests(
    requests: list[dict[str, any]],
    request_url: str,
    api_token: _OAITokenHolder,
    # Reference: https://platform.openai.com/docs/guides/rate-limits/overview
    max_requests_per_minute: float = 3_500,
    max_tokens_per_minute: float = 90_000,
    token_encoding_name: str = "cl100k_base",
    max_attempts: int = 5,
    max_workers: int = 10,
    throw_original_error=False,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15

    # initialize trackers
    retry_queue = queue.Queue()
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()
    results: list[tuple[int, any]] = []
    requests_iter = enumerate(requests)
    last_index = len(requests) - 1
    requests_exhausted = False
    _logger.debug(f"Request pool executor will run {len(requests)} requests")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            # get next request (if one is not already waiting for capacity)
            if next_request is None:
                if not retry_queue.empty():
                    next_request = retry_queue.get_nowait()
                    _logger.debug(f"Retrying request {next_request.index}: {next_request}")
                elif req := next(requests_iter, None):
                    # get new request
                    index, request_json = req
                    current_time = time.time()
                    next_request = APIRequest(
                        request_url=request_url,
                        index=index,
                        request_json=request_json,
                        token_consumption=num_tokens_consumed_from_request(
                            request_json, request_url, token_encoding_name
                        ),
                        attempts_left=max_attempts,
                        results=results,
                        start_time=current_time,
                        last_log_time=current_time,
                    )
                    status_tracker.start_task()
                    requests_exhausted = index == last_index

            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            available_request_capacity = min(
                available_request_capacity
                + int(max_requests_per_minute * seconds_since_update / 60.0),
                max_requests_per_minute,
            )
            available_token_capacity = min(
                available_token_capacity + int(max_tokens_per_minute * seconds_since_update / 60.0),
                max_tokens_per_minute,
            )
            last_update_time = current_time

            if next_request:
                _logger.debug(f"Available request capacity: {available_request_capacity}")
                _logger.debug(f"Available token capacity: {available_token_capacity}")
                next_request_tokens = next_request.token_consumption
                if (
                    available_request_capacity >= 1
                    and available_token_capacity >= next_request_tokens
                ):
                    # update counters
                    available_request_capacity -= 1
                    available_token_capacity -= next_request_tokens
                    next_request.attempts_left -= 1
                    # call API
                    api_token.refresh(logger=_logger)
                    executor.submit(
                        next_request.call_api,
                        retry_queue=retry_queue,
                        status_tracker=status_tracker,
                        headers=api_token.auth_headers(),
                    )
                    next_request = None  # reset next_request to empty
                else:
                    next_request = None
                    status_tracker.complete_task(success=False)
                    status_tracker.error = mlflow.MlflowException(
                        "Request size exceeded max tokens."
                    )

            # if all tasks are finished, break
            if requests_exhausted and status_tracker.num_tasks_in_progress == 0:
                break

            # if a rate limit error was hit recently, pause to cool down
            seconds_since_rate_limit_error = (
                time.time() - status_tracker.time_of_last_rate_limit_error
            )
            if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                remaining_seconds_to_pause = (
                    seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
                )
                _logger.debug(
                    "Encountered rate limit error. Pausing to cool down for "
                    f"{remaining_seconds_to_pause} seconds..."
                )
                time.sleep(remaining_seconds_to_pause)
                # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago

            time.sleep(0.001)  # avoid busy waiting

    # after finishing, log final status
    if status_tracker.num_tasks_failed > 0:
        if throw_original_error and len(requests) == 1:
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
