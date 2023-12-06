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

import logging
import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import langchain.chains
from langchain.schema import AgentAction

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
    errors: dict

    def _prepare_to_serialize(self, response: dict):
        """
        Converts LangChain objects to JSON-serializable formats.
        """
        from langchain.load.dump import dumps

        if "intermediate_steps" in response:
            steps = response["intermediate_steps"]
            if (
                isinstance(steps, tuple)
                and len(steps) == 2
                and isinstance(steps[0], AgentAction)
                and isinstance(steps[1], str)
            ):
                response["intermediate_steps"] = [
                    {
                        "tool": agent.tool,
                        "tool_input": agent.tool_input,
                        "log": agent.log,
                        "result": result,
                    }
                    for agent, result in response["intermediate_steps"]
                ]
            else:
                try:
                    # `AgentAction` objects are not yet implemented for serialization in `dumps`
                    # https://github.com/langchain-ai/langchain/issues/8815#issuecomment-1666763710
                    response["intermediate_steps"] = dumps(steps)
                except Exception as e:
                    _logger.warning(f"Failed to serialize intermediate steps: {e!r}")
        # The `dumps` format for `Document` objects is noisy, so we will still have custom logic
        if "source_documents" in response:
            response["source_documents"] = [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in response["source_documents"]
            ]

    def call_api(self, status_tracker: StatusTracker):
        """
        Calls the LangChain API and stores results.
        """
        import numpy as np
        from langchain.schema import BaseRetriever

        from mlflow.langchain.utils import lc_runnables_types, runnables_supports_batch_types

        _logger.debug(f"Request #{self.index} started")
        try:
            if isinstance(self.lc_model, BaseRetriever):
                # Retrievers are invoked differently than Chains
                docs = self.lc_model.get_relevant_documents(**self.request_json)
                response = [
                    {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
                ]
            elif isinstance(self.lc_model, lc_runnables_types()):
                if isinstance(self.request_json, np.ndarray):
                    # numpy array is not json serializable, so we convert it to list
                    self.request_json = self.request_json.tolist()
                if isinstance(self.request_json, dict):
                    # This is a temporary fix for the case when spark_udf converts
                    # input into pandas dataframe with column name, while the model
                    # does not accept dictionaries as input, it leads to erros like
                    # Expected Scalar value for String field \'query_text\'\\n
                    try:
                        response = self.lc_model.invoke(self.request_json)
                    except Exception:
                        _logger.warning(
                            f"Failed to invoke {self.lc_model.__class__.__name__} "
                            "with {self.request_json}. Error: {e!r}. Trying to "
                            "invoke with the first value of the dictionary."
                        )
                        self.request_json = next(iter(self.request_json.values()))
                        if isinstance(self.request_json, np.ndarray):
                            self.request_json = self.request_json.tolist()
                        response = self.lc_model.invoke(self.request_json)
                elif isinstance(self.request_json, list) and isinstance(
                    self.lc_model, runnables_supports_batch_types()
                ):
                    response = self.lc_model.batch(self.request_json)
                else:
                    response = self.lc_model.invoke(self.request_json)
            else:
                response = self.lc_model(self.request_json, return_only_outputs=True)

                # to maintain existing code, single output chains will still return only the result
                if len(response) == 1:
                    response = response.popitem()[1]
                else:
                    self._prepare_to_serialize(response)

            _logger.debug(f"Request #{self.index} succeeded")
            status_tracker.complete_task(success=True)
            self.results.append((self.index, response))
        except Exception as e:
            self.errors[
                self.index
            ] = f"error: {e!r} {traceback.format_exc()}\n request payload: {self.request_json}"
            status_tracker.increment_num_api_errors()
            status_tracker.complete_task(success=False)


def process_api_requests(
    lc_model,
    requests: Optional[List[Union[Any, Dict[str, Any]]]] = None,
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
    errors: dict = {}
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
                        index=index,
                        lc_model=lc_model,
                        request_json=request_json,
                        results=results,
                        errors=errors,
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
                f"{status_tracker.num_tasks_failed} tasks failed. Errors: {errors}"
            )

        return [res for _, res in sorted(results)]
