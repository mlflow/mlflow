# Source: https://github.com/openai/openai-cookbook/blob/6df6ceff470eeba26a56de131254e775292eac22/examples/api_request_parallel_processor.py
# A couple changes were made to make it work with MLflow.
# Only chat completion is supported for now.

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
import argparse
import json
import pathlib
import sys
import subprocess
import tempfile
import asyncio  # for running API calls concurrently
import logging  # for logging rate limit warnings and other messages
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass  # for storing API inputs, outputs, and metadata

import tiktoken  # for counting tokens
import openai
import openai.error

import mlflow
from mlflow.environment_variables import _MLFLOW_OPENAI_TESTING

_logger = logging.getLogger(__name__)


@dataclass
class StatusTracker:
    """
    Stores metadata about the script's progress. Only one instance is created.
    """

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits

    def start_task(self):
        self.num_tasks_started += 1
        self.num_tasks_in_progress += 1

    def complete_task(self, *, success: bool):
        self.num_tasks_in_progress -= 1
        if success:
            self.num_tasks_succeeded += 1
        else:
            self.num_tasks_failed += 1


@dataclass
class APIRequest:
    """
    Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API
    call.
    """

    index: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    output_file: str

    async def call_api(self, retry_queue: asyncio.Queue, status_tracker: StatusTracker):
        """
        Calls the OpenAI API and stores results.
        """
        _logger.info(f"Starting request #{self.index}")
        try:
            response = await openai.ChatCompletion.acreate(**self.request_json)
            status_tracker.complete_task(success=True)
            # Save the index (will be used for sorting) and response to the output file
            with open(self.output_file, "a") as f:
                f.write(f"{self.index} ")
                json.dump(response, f)
                f.write("\n")
        except openai.error.RateLimitError as e:
            _logger.warning(f"Request {self.index} failed with error {e!r}")
            status_tracker.time_of_last_rate_limit_error = time.time()
            status_tracker.num_rate_limit_errors += 1
        # Other retryable errors
        except (
            openai.error.Timeout,
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.ServiceUnavailableError,
        ) as e:
            _logger.warning(f"Request {self.index} failed with Exception {e!r}")
            status_tracker.num_api_errors += 1
            if self.attempts_left > 0:
                retry_queue.put_nowait(self)
            else:
                status_tracker.complete_task(success=False)
        # Unretryable errors
        except Exception as e:
            _logger.warning(f"Request {self.index} failed with Exception {e!r}")
            status_tracker.num_api_errors += 1
            status_tracker.complete_task(success=False)


def num_tokens_consumed_from_request(
    request_json: dict, api_endpoint: str, token_encoding_name: str
):
    """
    Count the number of tokens in the request. Only supports completion and embedding requests.
    """
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
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
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError(
                    "Expecting either string or list of strings for 'prompt' field in completion "
                    "request"
                )
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        inp = request_json["input"]
        if isinstance(inp, str):  # single input
            num_tokens = len(encoding.encode(inp))
            return num_tokens
        elif isinstance(inp, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in inp])
            return num_tokens
        else:
            raise TypeError(
                'Expecting either string or list of strings for "inputs" field in embedding request'
            )
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


async def process_api_requests(
    input_file: str,
    output_file: str,
    max_requests_per_minute: float = 1_500,
    max_tokens_per_minute: float = 125_000,
    token_encoding_name: str = "cl100k_base",
    max_attempts: int = 5,
):
    """
    Processes API requests in parallel, throttling to stay under rate limits.
    """
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # `requests` will provide requests one at a time
    with open(input_file) as f:
        requests_iter = enumerate(f)
        while True:
            # get next request (if one is not already waiting for capacity)
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                    _logger.warning(f"Retrying request {next_request.index}: {next_request}")
                elif req := next(requests_iter, None):
                    # get new request
                    index, request_json = req
                    request_json = json.loads(request_json)
                    next_request = APIRequest(
                        index=index,
                        request_json=request_json,
                        token_consumption=num_tokens_consumed_from_request(
                            request_json, "chat/completions", token_encoding_name
                        ),
                        attempts_left=max_attempts,
                        output_file=output_file,
                    )
                    status_tracker.start_task()

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

            # if enough capacity available, call API
            if next_request:
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
                    asyncio.create_task(
                        next_request.call_api(
                            retry_queue=queue_of_requests_to_retry, status_tracker=status_tracker
                        )
                    )
                    next_request = None  # reset next_request to empty

            # if all tasks are finished, break
            if status_tracker.num_tasks_in_progress == 0:
                break

            # main loop sleeps briefly so concurrent tasks can run
            await asyncio.sleep(seconds_to_sleep_each_loop)

            # if a rate limit error was hit recently, pause to cool down
            seconds_since_rate_limit_error = (
                time.time() - status_tracker.time_of_last_rate_limit_error
            )
            if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                remaining_seconds_to_pause = (
                    seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
                )
                await asyncio.sleep(remaining_seconds_to_pause)
                # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                cool_down_time = time.ctime(
                    status_tracker.time_of_last_rate_limit_error
                    + seconds_to_pause_after_rate_limit_error
                )
                _logger.warning(f"Pausing to cool down until {cool_down_time}")

        # after finishing, log final status
        if status_tracker.num_tasks_failed > 0:
            raise mlflow.MlflowException(
                f"{status_tracker.num_tasks_failed} tasks failed. See logs for details."
            )
        if status_tracker.num_rate_limit_errors > 0:
            _logger.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. "
                "Consider running at a lower rate."
            )


def run_as_subprocess(requests):
    _logger.info(f"Running {len(requests)} requests in parallel...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = pathlib.Path(temp_dir)
        input_file = temp_dir / "input.jsonl"
        output_file = temp_dir / "output.jsonl"
        with open(input_file, "w") as f:
            for request in requests:
                json.dump(request, f)
                f.write("\n")
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--input-file",
                input_file,
                "--output-file",
                output_file,
            ],
            check=True,
        )
        with open(output_file) as f:
            responses = []
            for line in f:
                index, response = line.split(maxsplit=1)
                index = int(index)
                response = json.loads(response)
                responses.append((index, response))
            return [r[1] for r in sorted(responses, key=lambda x: x[0])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    if _MLFLOW_OPENAI_TESTING.get():
        from mlflow.openai.utils import _mock_async_request, _mock_async_chat_completion_response

        with _mock_async_request(return_value=_mock_async_chat_completion_response()):
            asyncio.run(process_api_requests(args.input_file, args.output_file))
    else:
        asyncio.run(process_api_requests(args.input_file, args.output_file))


if __name__ == "__main__":
    main()
