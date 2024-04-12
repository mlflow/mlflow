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
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Union

import langchain.chains
import pydantic
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AIMessage, HumanMessage, SystemMessage
from langchain.schema import ChatMessage as LangChainChatMessage
from packaging.version import Version

import mlflow
from mlflow.exceptions import MlflowException

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


# NB: Even though _ChatMessage is only referenced in one method within this module
# (as of 12/27/2023), it must be defined at the module level for compatibility with
# pydantic < 2
class _ChatMessage(pydantic.BaseModel, extra="forbid"):
    role: str
    content: str

    def to_langchain_message(self) -> LangChainChatMessage:
        if self.role == "system":
            return SystemMessage(content=self.content)
        elif self.role == "assistant":
            return AIMessage(content=self.content)
        elif self.role == "user":
            return HumanMessage(content=self.content)
        else:
            raise MlflowException.invalid_parameter_value(
                f"Unrecognized chat message role: {self.role}"
            )


class _ChatDeltaMessage(pydantic.BaseModel):
    role: str
    content: str


# NB: Even though _ChatRequest is only referenced in one method within this module
# (as of 12/27/2023), it must be defined at the module level for compatibility with
# pydantic < 2
class _ChatRequest(pydantic.BaseModel, extra="forbid"):
    messages: List[_ChatMessage]


class _ChatChoice(pydantic.BaseModel, extra="forbid"):
    index: int
    message: _ChatMessage = None
    finish_reason: Optional[str] = None


class _ChatChoiceDelta(pydantic.BaseModel):
    index: int
    finish_reason: Optional[str] = None
    delta: _ChatDeltaMessage


class _ChatUsage(pydantic.BaseModel, extra="forbid"):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class _ChatResponse(pydantic.BaseModel, extra="forbid"):
    id: Optional[str] = None
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    # Make the model field optional since we may not be able to get a stable model identifier
    # for an arbitrary LangChain model
    model: Optional[str] = None
    choices: List[_ChatChoice]
    usage: _ChatUsage


class _ChatChunkResponse(pydantic.BaseModel):
    id: Optional[str] = None
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    # Make the model field optional since we may not be able to get a stable model identifier
    # for an arbitrary LangChain model
    model: Optional[str] = None
    choices: List[_ChatChoiceDelta]


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
    convert_chat_responses: bool
    did_perform_chat_conversion: bool
    stream: bool

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

    def single_call_api(self, callback_handlers: Optional[List[BaseCallbackHandler]]):
        from langchain.schema import BaseRetriever

        from mlflow.langchain.utils import lc_runnables_types

        if isinstance(self.lc_model, BaseRetriever):
            # Retrievers are invoked differently than Chains
            docs = self.lc_model.get_relevant_documents(**self.request_json)
            response = [
                {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
            ]
        elif isinstance(self.lc_model, lc_runnables_types()):

            def _predict_single_input(single_input):
                if self.stream:
                    return self.lc_model.stream(
                        single_input, config={"callbacks": callback_handlers}
                    )
                return self.lc_model.invoke(single_input, config={"callbacks": callback_handlers})

            if isinstance(self.request_json, dict):
                # This is a temporary fix for the case when spark_udf converts
                # input into pandas dataframe with column name, while the model
                # does not accept dictionaries as input, it leads to errors like
                # Expected Scalar value for String field 'query_text'
                try:
                    response = _predict_single_input(self.request_json)
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
                    ) = APIRequest._transform_request_json_for_chat_if_necessary(
                        self.request_json, self.lc_model
                    )
                    self.did_perform_chat_conversion = did_perform_chat_conversion

                    response = _predict_single_input(prepared_request_json)
            else:
                response = _predict_single_input(self.request_json)

            if self.did_perform_chat_conversion or self.convert_chat_responses:
                if self.stream:
                    response = APIRequest._try_transform_response_iter_to_chat_format(response)
                else:
                    response = APIRequest._try_transform_response_to_chat_format(response)
        else:
            response = self.lc_model(
                self.request_json,
                return_only_outputs=True,
                callbacks=callback_handlers,
            )

            if self.did_perform_chat_conversion or self.convert_chat_responses:
                response = APIRequest._try_transform_response_to_chat_format(response)
            elif len(response) == 1:
                # to maintain existing code, single output chains will still return
                # only the result
                response = response.popitem()[1]
            else:
                self._prepare_to_serialize(response)

        return response

    def call_api(
        self, status_tracker: StatusTracker, callback_handlers: Optional[List[BaseCallbackHandler]]
    ):
        """
        Calls the LangChain API and stores results.
        """
        _logger.debug(f"Request #{self.index} started with payload: {self.request_json}")

        try:
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

    @staticmethod
    def _transform_request_json_for_chat_if_necessary(request_json, lc_model):
        """
        Returns:
            A 2-element tuple containing:

                1. The new request.
                2. A boolean indicating whether or not the request was transformed from the OpenAI
                chat format.
        """
        input_fields = APIRequest._get_lc_model_input_fields(lc_model)
        if "messages" in input_fields:
            # If the chain accepts a "messages" field directly, don't attempt to convert
            # the request to LangChain's Message format automatically. Assume that the chain
            # is handling the "messages" field by itself
            return request_json, False

        def json_dict_might_be_chat_request(json_message: Dict):
            return (
                isinstance(json_message, dict)
                and "messages" in json_message
                and
                # Additional keys can't be specified when calling LangChain invoke() / batch()
                # with chat messages
                len(json_message) == 1
            )

        if isinstance(request_json, dict) and json_dict_might_be_chat_request(request_json):
            try:
                return APIRequest._convert_chat_request_or_throw(request_json), True
            except pydantic.ValidationError:
                return request_json, False
        elif isinstance(request_json, list) and all(
            json_dict_might_be_chat_request(json) for json in request_json
        ):
            try:
                return (
                    [
                        APIRequest._convert_chat_request_or_throw(json_dict)
                        for json_dict in request_json
                    ],
                    True,
                )
            except pydantic.ValidationError:
                return request_json, False
        else:
            return request_json, False

    @staticmethod
    def _try_transform_response_to_chat_format(response):
        if isinstance(response, str):
            message_content = response
            message_id = None
        elif isinstance(response, AIMessage):
            message_content = response.content
            message_id = getattr(response, "id", None)
        else:
            return response

        transformed_response = _ChatResponse(
            id=message_id,
            created=int(time.time()),
            model=None,
            choices=[
                _ChatChoice(
                    index=0,
                    message=_ChatMessage(
                        role="assistant",
                        content=message_content,
                    ),
                    finish_reason=None,
                )
            ],
            usage=_ChatUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

        if Version(pydantic.__version__) < Version("2.0"):
            return json.loads(transformed_response.json())
        else:
            return transformed_response.model_dump(mode="json")

    @staticmethod
    def _try_transform_response_iter_to_chat_format(chunk_iter):
        from langchain_core.messages.ai import AIMessageChunk

        is_pydantic_v1 = Version(pydantic.__version__) < Version("2.0")

        def _gen_converted_chunk(message_content, message_id, finish_reason):
            transformed_response = _ChatChunkResponse(
                id=message_id,
                created=int(time.time()),
                model=None,
                choices=[
                    _ChatChoiceDelta(
                        index=0,
                        delta=_ChatDeltaMessage(
                            role="assistant",
                            content=message_content,
                        ),
                        finish_reason=finish_reason,
                    )
                ],
            )

            if is_pydantic_v1:
                return json.loads(transformed_response.json())
            else:
                return transformed_response.model_dump(mode="json")

        def _convert(chunk):
            if isinstance(chunk, str):
                message_content = chunk
                message_id = None
                finish_reason = None
            elif isinstance(chunk, AIMessageChunk):
                message_content = chunk.content
                message_id = getattr(chunk, "id", None)

                if response_metadata := getattr(chunk, "response_metadata", None):
                    finish_reason = response_metadata.get("finish_reason")
                else:
                    finish_reason = None
            elif isinstance(chunk, AIMessage):
                # The langchain chat model does not support stream
                # so `model.stream` returns the whole result.
                message_content = chunk.content
                message_id = getattr(chunk, "id", None)
                finish_reason = "stop"
            else:
                return chunk
            return _gen_converted_chunk(
                message_content,
                message_id=message_id,
                finish_reason=finish_reason,
            )

        return map(_convert, chunk_iter)

    @staticmethod
    def _get_lc_model_input_fields(lc_model) -> Set:
        try:
            if hasattr(lc_model, "input_schema") and callable(lc_model.input_schema):
                return set(lc_model.input_schema().__fields__)
        except Exception as e:
            _logger.debug(
                f"Unexpected exception while checking LangChain input schema for"
                f" request transformation: {e}"
            )

        return set()

    @staticmethod
    def _convert_chat_request_or_throw(chat_request: Dict):
        if Version(pydantic.__version__) < Version("2.0"):
            model = _ChatRequest.parse_obj(chat_request)
        else:
            model = _ChatRequest.model_validate(chat_request)

        return [message.to_langchain_message() for message in model.messages]


def process_api_requests(
    lc_model,
    requests: Optional[List[Union[Any, Dict[str, Any]]]] = None,
    max_workers: int = 10,
    callback_handlers: Optional[List[BaseCallbackHandler]] = None,
    convert_chat_responses: bool = False,
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

    # Note: we should call `_transform_request_json_for_chat_if_necessary`
    # for the whole batch data, because the conversion should obey the rule
    # that if any record in the batch can't be converted, then all the record
    # in this batch can't be converted.
    (
        converted_chat_requests,
        did_perform_chat_conversion,
    ) = APIRequest._transform_request_json_for_chat_if_necessary(requests, lc_model)

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
):
    from mlflow.langchain.utils import lc_runnables_types

    if not isinstance(lc_model, lc_runnables_types()):
        raise MlflowException(
            f"Model {lc_model.__class__.__name__} does not support streaming prediction output."
        )

    (
        converted_chat_requests,
        did_perform_chat_conversion,
    ) = APIRequest._transform_request_json_for_chat_if_necessary(request_json, lc_model)

    api_request = APIRequest(
        index=0,
        lc_model=lc_model,
        request_json=converted_chat_requests,
        results=None,
        errors=None,
        convert_chat_responses=convert_chat_responses,
        did_perform_chat_conversion=did_perform_chat_conversion,
        stream=True,
    )

    return api_request.single_call_api(callback_handlers)
