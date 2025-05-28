import json
import logging
import time
from collections import defaultdict
from typing import Any, Optional, Union

import pydantic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.outputs.generation import Generation

from mlflow.environment_variables import MLFLOW_CONVERT_MESSAGES_DICT_FOR_LANGCHAIN
from mlflow.exceptions import MlflowException
from mlflow.tracing.constant import TokenUsageKey
from mlflow.types.chat import (
    ChatChoice,
    ChatChoiceDelta,
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
)
from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER

_logger = logging.getLogger(__name__)


_TOKEN_USAGE_KEY_MAPPING = {
    # OpenAI
    "prompt_tokens": TokenUsageKey.INPUT_TOKENS,
    "completion_tokens": TokenUsageKey.OUTPUT_TOKENS,
    "total_tokens": TokenUsageKey.TOTAL_TOKENS,
    # OpenAI Streaming, Anthropic, etc.
    "input_tokens": TokenUsageKey.INPUT_TOKENS,
    "output_tokens": TokenUsageKey.OUTPUT_TOKENS,
}


def convert_lc_message_to_chat_message(lc_message: Union[BaseMessage]) -> ChatMessage:
    """
    Convert LangChain's message format to the MLflow's standard chat message format.
    """
    if isinstance(lc_message, AIMessage):
        if tool_calls := _get_tool_calls_from_ai_message(lc_message):
            return ChatMessage(
                role="assistant",
                # If tool calls present, content null value should be None not empty string
                # according to the OpenAI spec, which ChatMessage is following
                # Ref: https://github.com/langchain-ai/langchain/blob/32917a0b98cb8edcfb8d0e84f0878434e1c3f192/libs/partners/openai/langchain_openai/chat_models/base.py#L116-L117
                content=lc_message.content or None,
                tool_calls=tool_calls,
            )
        else:
            return ChatMessage(role="assistant", content=lc_message.content)
    elif isinstance(lc_message, ChatMessage):
        return ChatMessage(role=lc_message.role, content=lc_message.content)
    elif isinstance(lc_message, FunctionMessage):
        return ChatMessage(role="function", content=lc_message.content)
    elif isinstance(lc_message, ToolMessage):
        return ChatMessage(
            role="tool",
            content=lc_message.content,
            tool_call_id=lc_message.tool_call_id,
        )
    elif isinstance(lc_message, HumanMessage):
        return ChatMessage(role="user", content=lc_message.content)
    elif isinstance(lc_message, SystemMessage):
        return ChatMessage(role="system", content=lc_message.content)
    else:
        raise MlflowException.invalid_parameter_value(
            f"Unexpected message type. Expected a BaseMessage subclass, but got: {type(lc_message)}"
        )


def _chat_model_to_langchain_message(message: ChatMessage) -> BaseMessage:
    """
    Convert the MLflow's standard chat message format to LangChain's message format.
    """
    if message.role == "system":
        return SystemMessage(content=message.content)
    elif message.role == "assistant":
        return AIMessage(content=message.content)
    elif message.role == "user":
        return HumanMessage(content=message.content)
    elif message.role == "tool":
        return ToolMessage(content=message.content, tool_call_id=message.tool_call_id)
    elif message.role == "function":
        return FunctionMessage(content=message.content)
    else:
        raise MlflowException.invalid_parameter_value(
            f"Unrecognized chat message role: {message.role}"
        )


def _get_tool_calls_from_ai_message(message: AIMessage) -> list[dict]:
    # AIMessage does not have tool_calls field in LangChain < 0.1.0.
    if not hasattr(message, "tool_calls"):
        return []

    tool_calls = [
        {
            "type": "function",
            "id": tc["id"],
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["args"]),
            },
        }
        for tc in message.tool_calls
    ]

    invalid_tool_calls = [
        {
            "type": "function",
            "id": tc["id"],
            "function": {
                "name": tc["name"],
                "arguments": tc["args"],
            },
        }
        for tc in message.invalid_tool_calls
    ]

    if tool_calls or invalid_tool_calls:
        return tool_calls + invalid_tool_calls

    # Get tool calls from additional kwargs if present.
    return [
        {
            k: v
            for k, v in tool_call.items()  # type: ignore[union-attr]
            if k in {"id", "type", "function"}
        }
        for tool_call in message.additional_kwargs.get("tool_calls", [])
    ]


def convert_lc_generation_to_chat_message(lc_gen: Generation) -> ChatMessage:
    """
    Convert LangChain's generation format to the MLflow's standard chat message format.
    """
    if isinstance(lc_gen, ChatGeneration):
        try:
            return convert_lc_message_to_chat_message(lc_gen.message)
        except Exception as e:
            # When failed to convert the message, return as assistant message
            _logger.debug(
                f"Failed to convert the message from ChatGeneration to ResponseMessage: {e}",
                exc_info=True,
            )

    return ChatMessage(role="assistant", content=lc_gen.text)


def try_transform_response_to_chat_format(response: Any) -> dict:
    """
    Try to convert the response to the standard chat format and return its dict representation.

    If the response is not one of the supported types, return the response as-is.
    """
    if isinstance(response, (str, AIMessage)):
        if isinstance(response, str):
            message_id = None
            message = ChatMessage(role="assistant", content=response)
        else:
            message_id = getattr(response, "id", None)
            message = convert_lc_message_to_chat_message(response)

        transformed_response = ChatCompletionResponse(
            id=message_id,
            created=int(time.time()),
            model="",
            object="chat.completion",
            choices=[
                ChatChoice(
                    index=0,
                    message=message,
                    finish_reason=None,
                )
            ],
            usage=ChatUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )
        if IS_PYDANTIC_V2_OR_NEWER:
            return transformed_response.model_dump(mode="json", exclude_unset=True)
        else:
            return json.loads(transformed_response.json(exclude_unset=True))
    else:
        return response


def try_transform_response_iter_to_chat_format(chunk_iter):
    from langchain_core.messages.ai import AIMessageChunk

    def _gen_converted_chunk(message_content, message_id, finish_reason):
        transformed_response = ChatCompletionChunk(
            id=message_id,
            created=int(time.time()),
            model="",
            choices=[
                ChatChunkChoice(
                    index=0,
                    delta=ChatChoiceDelta(
                        role="assistant",
                        content=message_content,
                    ),
                    finish_reason=finish_reason,
                )
            ],
        )

        if IS_PYDANTIC_V2_OR_NEWER:
            return transformed_response.model_dump(mode="json")
        else:
            return json.loads(transformed_response.json())

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


def _convert_chat_request_or_throw(chat_request: dict[str, Any]) -> list[Union[BaseMessage]]:
    model = ChatCompletionRequest.validate_compat(chat_request)
    return [_chat_model_to_langchain_message(message) for message in model.messages]


def _convert_chat_request(chat_request: Union[dict, list[dict]]):
    if isinstance(chat_request, list):
        return [_convert_chat_request_or_throw(request) for request in chat_request]
    else:
        return _convert_chat_request_or_throw(chat_request)


def _get_lc_model_input_fields(lc_model) -> set[str]:
    try:
        if hasattr(lc_model, "input_schema"):
            return set(lc_model.input_schema.__fields__)
    except Exception as e:
        _logger.debug(
            f"Unexpected exception while checking LangChain input schema for"
            f" request transformation: {e}"
        )

    return set()


def _should_transform_request_json_for_chat(lc_model):
    # Avoid converting the request to LangChain's Message format if the chain
    # is an AgentExecutor, as LangChainChatMessage might not be accepted by the chain
    from langchain.agents import AgentExecutor

    if isinstance(lc_model, AgentExecutor):
        return False

    input_fields = _get_lc_model_input_fields(lc_model)
    if "messages" in input_fields:
        # If the chain accepts a "messages" field directly, don't attempt to convert
        # the request to LangChain's Message format automatically. Assume that the chain
        # is handling the "messages" field by itself
        return False

    return True


def transform_request_json_for_chat_if_necessary(request_json, lc_model):
    """
    Convert the input request JSON to LangChain's Message format if the LangChain model
    accepts ChatMessage objects (e.g. AIMessage, HumanMessage, SystemMessage) as input.

    Args:
        request_json: The input request JSON.
        lc_model: The LangChain model.

    Returns:
        A 2-element tuple containing:

            1. The new request.
            2. A boolean indicating whether or not the request was transformed from the OpenAI
            chat format.
    """

    def json_dict_might_be_chat_request(json_message):
        return (
            isinstance(json_message, dict)
            and "messages" in json_message
            and
            # Additional keys can't be specified when calling LangChain invoke() / batch()
            # with chat messages
            len(json_message) == 1
            # messages field should be a list
            and isinstance(json_message["messages"], list)
        )

    def is_list_of_chat_messages(json_message: list[dict]):
        return isinstance(json_message, list) and all(
            json_dict_might_be_chat_request(message) for message in json_message
        )

    should_convert = MLFLOW_CONVERT_MESSAGES_DICT_FOR_LANGCHAIN.get()
    if should_convert is None:
        should_convert = _should_transform_request_json_for_chat(lc_model) and (
            json_dict_might_be_chat_request(request_json) or is_list_of_chat_messages(request_json)
        )
        if should_convert:
            _logger.debug(
                "Converting the request JSON to LangChain's Message format. "
                "To disable this conversion, set the environment variable "
                f"`{MLFLOW_CONVERT_MESSAGES_DICT_FOR_LANGCHAIN}` to 'false'."
            )

    if should_convert:
        try:
            return _convert_chat_request(request_json), True
        except pydantic.ValidationError:
            _logger.debug(
                "Failed to convert the request JSON to LangChain's Message format. "
                "The request will be passed to the LangChain model as-is. ",
                exc_info=True,
            )
            return request_json, False
    else:
        return request_json, False


def parse_token_usage(
    lc_generations: list[Generation],
) -> Optional[dict[str, int]]:
    """Parse the token usage from the LangChain generations."""
    aggregated = defaultdict(int)
    for generation in lc_generations:
        if token_usage := _parse_token_usage_from_generation(generation):
            for key in token_usage:
                aggregated[key] += token_usage[key]

    return dict(aggregated) if aggregated else None


def _parse_token_usage_from_generation(generation: Generation) -> Optional[dict[str, int]]:
    message = getattr(generation, "message", None)
    if not message:
        return None

    metadata = (
        message.usage_metadata
        or message.response_metadata.get("usage")
        or message.response_metadata.get("token_usage")
    )
    return _parse_token_counts(metadata) if metadata else None


def _parse_token_counts(usage_metadata: dict[str, Any]) -> dict[str, int]:
    """Standardize token usage metadata keys to MLflow's token usage keys."""
    usage = {}
    for key, value in usage_metadata.items():
        if usage_key := _TOKEN_USAGE_KEY_MAPPING.get(key):
            usage[usage_key] = value

    # If the total tokens are not present, calculate it from the input and output tokens
    if usage and usage.get(TokenUsageKey.TOTAL_TOKENS) is None:
        usage[TokenUsageKey.TOTAL_TOKENS] = usage.get(TokenUsageKey.INPUT_TOKENS, 0) + usage.get(
            TokenUsageKey.OUTPUT_TOKENS, 0
        )

    return usage
