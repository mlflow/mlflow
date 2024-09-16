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
from langchain.callbacks.base import BaseCallbackHandler

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.langchain.utils.chat import (
    transform_request_json_for_chat_if_necessary,
    try_transform_response_iter_to_chat_format,
    try_transform_response_to_chat_format,
)
from mlflow.langchain.utils.serialization import convert_to_serializable
from mlflow.pyfunc.context import (
    Context,
    get_prediction_context,
    maybe_set_prediction_context,
)

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

    Args:
        index: The request's index in the tasks list
        lc_model: The LangChain model to call
        request_json: The request's input data
        results: The list to append the request's output data to, it's a list of tuples
            (index, response)
        errors: A dictionary to store any errors that occur
        convert_chat_responses: Whether to convert the model's responses to chat format
        did_perform_chat_conversion: Whether the input data was converted to chat format
            based on the model's type and input data.
        stream: Whether the request is a stream request
        prediction_context: The prediction context to use for the request
    """

    index: int
    lc_model: langchain.chains.base.Chain
    request_json: dict
    results: list[tuple[int, str]]
    errors: dict
    convert_chat_responses: bool
    did_perform_chat_conversion: bool
    stream: bool
    params: Dict[str, Any]
    prediction_context: Optional[Context] = None

    def _predict_single_input(self, single_input, callback_handlers, **kwargs):
        config = kwargs.pop("config", {})
        config["callbacks"] = config.get("callbacks", []) + (callback_handlers or [])
        if self.stream:
            return self.lc_model.stream(single_input, config=config, **kwargs)
        if hasattr(self.lc_model, "invoke"):
            return self.lc_model.invoke(single_input, config=config, **kwargs)
        else:
            # for backwards compatibility, __call__ is deprecated and will be removed in 0.3.0
            # kwargs shouldn't have config field if invoking with __call__
            return self.lc_model(single_input, callbacks=callback_handlers, **kwargs)

    def _try_convert_response(self, response):
        if self.stream:
            return try_transform_response_iter_to_chat_format(response)
        else:
            return try_transform_response_to_chat_format(response)

    def single_call_api(self, callback_handlers: Optional[List[BaseCallbackHandler]]):
        from langchain.schema import BaseRetriever

        from mlflow.langchain.utils import langgraph_types, lc_runnables_types

        if isinstance(self.lc_model, BaseRetriever):
            # Retrievers are invoked differently than Chains
            response = self.lc_model.get_relevant_documents(
                **self.request_json, callbacks=callback_handlers, **self.params
            )
        elif isinstance(self.lc_model, lc_runnables_types() + langgraph_types()):
            if isinstance(self.request_json, dict):
                # This is a temporary fix for the case when spark_udf converts
                # input into pandas dataframe with column name, while the model
                # does not accept dictionaries as input, it leads to errors like
                # Expected Scalar value for String field 'query_text'
                try:
                    response = self._predict_single_input(
                        self.request_json, callback_handlers, **self.params
                    )
                except TypeError as e:
                    _logger.warning(
                        f"Failed to invoke {self.lc_model.__class__.__name__} "
                        f"with {self.request_json}. Error: {e!r}. Trying to "
                        "invoke with the first value of the dictionary."
                    )
                    self.request_json = next(iter(self.request_json.values()))
                    (
                        prepared_request_json,
                        did_perform_chat_conversion,
                    ) = transform_request_json_for_chat_if_necessary(
                        self.request_json, self.lc_model
                    )
                    self.did_perform_chat_conversion = did_perform_chat_conversion

                    response = self._predict_single_input(
                        prepared_request_json, callback_handlers, **self.params
                    )
            else:
                response = self._predict_single_input(
                    self.request_json, callback_handlers, **self.params
                )

            if self.did_perform_chat_conversion or self.convert_chat_responses:
                response = self._try_convert_response(response)
        else:
            # return_only_outputs is invalid for stream call
            if isinstance(self.lc_model, langchain.chains.base.Chain) and not self.stream:
                kwargs = {"return_only_outputs": True}
            else:
                kwargs = {}
            kwargs.update(**self.params)
            response = self._predict_single_input(self.request_json, callback_handlers, **kwargs)

            if self.did_perform_chat_conversion or self.convert_chat_responses:
                response = self._try_convert_response(response)
            elif isinstance(response, dict) and len(response) == 1:
                # to maintain existing code, single output chains will still return
                # only the result
                response = response.popitem()[1]

        return convert_to_serializable(response)

    def call_api(
        self, status_tracker: StatusTracker, callback_handlers: Optional[List[BaseCallbackHandler]]
    ):
        """
        Calls the LangChain API and stores results.
        """
        _logger.debug(f"Request #{self.index} started with payload: {self.request_json}")

        try:
            with maybe_set_prediction_context(self.prediction_context):
                response = self.single_call_api(callback_handlers)
            _logger.debug(f"Request #{self.index} succeeded with response: {response}")
            self.results.append((self.index, response))
            status_tracker.complete_task(success=True)
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
    callback_handlers: Optional[List[BaseCallbackHandler]] = None,
    convert_chat_responses: bool = False,
    params: Optional[Dict[str, Any]] = None,
):
    """
    Processes API requests in parallel.
    """

    # initialize trackers
    retry_queue = queue.Queue()
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    results = []
    errors = {}

    # Note: we should call `transform_request_json_for_chat_if_necessary`
    # for the whole batch data, because the conversion should obey the rule
    # that if any record in the batch can't be converted, then all the record
    # in this batch can't be converted.
    (
        converted_chat_requests,
        did_perform_chat_conversion,
    ) = transform_request_json_for_chat_if_necessary(requests, lc_model)

    requests_iter = enumerate(converted_chat_requests)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            # get next request (if one is not already waiting for capacity)
            if not retry_queue.empty():
                next_request = retry_queue.get_nowait()
                _logger.warning(f"Retrying request {next_request.index}: {next_request}")
            elif req := next(requests_iter, None):
                # get new request
                index, converted_chat_request_json = req
                next_request = APIRequest(
                    index=index,
                    lc_model=lc_model,
                    request_json=converted_chat_request_json,
                    results=results,
                    errors=errors,
                    convert_chat_responses=convert_chat_responses,
                    did_perform_chat_conversion=did_perform_chat_conversion,
                    stream=False,
                    prediction_context=get_prediction_context(),
                    params=params,
                )
                status_tracker.start_task()
            else:
                next_request = None

            # if enough capacity available, call API
            if next_request:
                # call API
                executor.submit(
                    next_request.call_api,
                    status_tracker=status_tracker,
                    callback_handlers=callback_handlers,
                )

            # if all tasks are finished, break
            # check next_request to avoid terminating the process
            # before extra requests need to be processed
            if status_tracker.num_tasks_in_progress == 0 and next_request is None:
                break

            time.sleep(0.001)  # avoid busy waiting

        # after finishing, log final status
        if status_tracker.num_tasks_failed > 0:
            raise mlflow.MlflowException(
                f"{status_tracker.num_tasks_failed} tasks failed. Errors: {errors}"
            )

        return [res for _, res in sorted(results)]


def process_stream_request(
    lc_model,
    request_json: Union[Any, Dict[str, Any]],
    callback_handlers: Optional[List[BaseCallbackHandler]] = None,
    convert_chat_responses: bool = False,
    params: Optional[Dict[str, Any]] = None,
):
    """
    Process single stream request.
    """
    if not hasattr(lc_model, "stream"):
        raise MlflowException(
            f"Model {lc_model.__class__.__name__} does not support streaming prediction output. "
            "No `stream` method found."
        )

    (
        converted_chat_requests,
        did_perform_chat_conversion,
    ) = transform_request_json_for_chat_if_necessary(request_json, lc_model)

    api_request = APIRequest(
        index=0,
        lc_model=lc_model,
        request_json=converted_chat_requests,
        results=None,
        errors=None,
        convert_chat_responses=convert_chat_responses,
        did_perform_chat_conversion=did_perform_chat_conversion,
        stream=True,
        prediction_context=get_prediction_context(),
        params=params,
    )
    with maybe_set_prediction_context(api_request.prediction_context):
        return api_request.single_call_api(callback_handlers)
