# Based ons: https://github.com/openai/openai-cookbook/blob/6df6ceff470eeba26a56de131254e775292eac22/examples/api_request_parallel_processor.py
# Several changes were made to make it work with MLflow.
# Currently, only chat completion is supported.

"""
API REQUEST PARALLEL PROCESSOR

Using the LangChain API to process lots of text quickly takes some care.
If you trickle in a million API requests one by one, they'll take days to complete.
This script parallelizes requests using LangChain API.

Features:
- Streams requests from file, to avoid running out of memory for giant jobs
- Makes requests concurrently, to maximize throughput
- Logs errors, to diagnose problems with requests
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import langchain

import mlflow

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
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    lock: threading.Lock = threading.Lock()

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
    lc_model: langchain.chains.base.Chain
    request_json: dict
    results: list[tuple[int, str]]

    def call_api(self, status_tracker: StatusTracker):
        """
        Calls the LangChain API and stores results.
        """
        from langchain.schema import BaseRetriever

        _logger.debug(f"Request #{self.index} started")
        try:
            if isinstance(self.lc_model, BaseRetriever):
                # Retrievers are invoked differently than Chains
                docs = self.lc_model.get_relevant_documents(**self.request_json)
                list_of_str_page_content = [doc.page_content for doc in docs]
                response = json.dumps(list_of_str_page_content)
            else:
                response = self.lc_model.run(**self.request_json)
            _logger.debug(f"Request #{self.index} succeeded")
            status_tracker.complete_task(success=True)
            self.results.append((self.index, response))
        except Exception as e:
            _logger.warning(f"Request #{self.index} failed with {e!r}")
            status_tracker.increment_num_api_errors()
            status_tracker.complete_task(success=False)


def process_api_requests(
    lc_model,
    requests: List[Union[str, Dict[str, Any]]] = None,
    max_workers: int = 10,
):
    """
    Processes API requests in parallel.
    """

    # initialize trackers
    retry_queue = queue.Queue()
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    results: list[tuple[int, str]] = []
    requests_iter = enumerate(requests)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            # get next request (if one is not already waiting for capacity)
            if next_request is None:
                if not retry_queue.empty():
                    next_request = retry_queue.get_nowait()
                    _logger.warning(f"Retrying request {next_request.index}: {next_request}")
                elif req := next(requests_iter, None):
                    # get new request
                    index, request_json = req
                    next_request = APIRequest(
                        index=index, lc_model=lc_model, request_json=request_json, results=results
                    )
                    status_tracker.start_task()

            # if enough capacity available, call API
            if next_request:
                # call API
                executor.submit(
                    next_request.call_api,
                    status_tracker=status_tracker,
                )
                next_request = None  # reset next_request to empty

            # if all tasks are finished, break
            if status_tracker.num_tasks_in_progress == 0:
                break

            time.sleep(0.001)  # avoid busy waiting

        # after finishing, log final status
        if status_tracker.num_tasks_failed > 0:
            raise mlflow.MlflowException(
                f"{status_tracker.num_tasks_failed} tasks failed. See logs for details."
            )

        return [res for _, res in sorted(results)]
