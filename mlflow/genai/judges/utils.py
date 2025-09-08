from __future__ import annotations

import json
import logging
import re
import threading
from contextlib import ContextDecorator
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlflow.genai.judges.base import JudgeField
    from mlflow.types.llm import ChatMessage

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.utils.enum_utils import StrEnum
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE
from mlflow.utils.uri import is_databricks_uri

_logger = logging.getLogger(__name__)

# "endpoints" is a special case for Databricks model serving endpoints.
_NATIVE_PROVIDERS = ["openai", "anthropic", "bedrock", "mistral", "endpoints"]

# Global cache to track model capabilities across function calls
# Key: model URI (e.g., "openai/gpt-4"), Value: boolean indicating response_format support
_MODEL_RESPONSE_FORMAT_CAPABILITIES: dict[str, bool] = {}


def get_default_model() -> str:
    if is_databricks_uri(mlflow.get_tracking_uri()):
        return _DATABRICKS_DEFAULT_JUDGE_MODEL
    else:
        return "openai:/gpt-4.1-mini"


def format_prompt(prompt: str, **values) -> str:
    """Format double-curly variables in the prompt template."""
    for key, value in values.items():
        prompt = re.sub(r"\{\{\s*" + key + r"\s*\}\}", str(value), prompt)
    return prompt


def add_output_format_instructions(prompt: str, output_fields: list["JudgeField"]) -> str:
    """
    Add structured output format instructions to a judge prompt.

    This ensures the LLM returns a JSON response with the expected fields,
    matching the expected format for the invoke_judge_model function.

    Args:
        prompt: The formatted prompt with template variables filled in
        output_fields: List of JudgeField objects defining output fields.

    Returns:
        The prompt with output format instructions appended
    """
    json_format_lines = []
    for field in output_fields:
        json_format_lines.append(f'    "{field.name}": "{field.description}"')

    json_format = "{\n" + ",\n".join(json_format_lines) + "\n}"

    output_format_instructions = f"""

Please provide your assessment in the following JSON format only (no markdown):

{json_format}"""
    return prompt + output_format_instructions


def _sanitize_justification(justification: str) -> str:
    # Some judge prompts instruct the model to think step by step.
    return justification.replace("Let's think step by step. ", "")


def invoke_judge_model(
    model_uri: str,
    prompt: str | list["ChatMessage"],
    assessment_name: str,
    trace: Trace | None = None,
    num_retries: int = 10,
) -> Feedback:
    """
    Invoke the judge model.

    First, try to invoke the judge model via litellm. If litellm is not installed,
    fallback to native parsing using the AI Gateway adapters.

    Args:
        model_uri: The model URI.
        prompt: The prompt to evaluate. Can be a string (single prompt) or
                a list of ChatMessage objects.
        assessment_name: The name of the assessment.
        trace: Optional trace object for context.
        num_retries: Number of retries on transient failures when using litellm.
    """
    from mlflow.metrics.genai.model_utils import (
        _parse_model_uri,
        get_endpoint_type,
        score_model_on_payload,
    )

    provider, model_name = _parse_model_uri(model_uri)

    # Convert to uniform ChatMessage format for internal processing
    if isinstance(prompt, str):
        from mlflow.types.llm import ChatMessage

        messages = [ChatMessage(role="user", content=prompt)]
    else:
        # Already ChatMessage objects
        messages = prompt

    # Try litellm first for better performance.
    if _is_litellm_available():
        response = _invoke_litellm(provider, model_name, messages, trace, num_retries)
    elif trace is not None:
        raise MlflowException(
            "LiteLLM is required for using traces with judges. "
            "Please install it with `pip install litellm`.",
            error_code=BAD_REQUEST,
        )
    elif provider in _NATIVE_PROVIDERS:
        # Convert ChatMessage objects to dicts for native providers
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]
        response = score_model_on_payload(
            model_uri=model_uri,
            payload=messages_dict,
            endpoint_type=get_endpoint_type(model_uri) or "llm/v1/chat",
        )
    else:
        raise MlflowException(
            f"LiteLLM is required for using '{provider}' LLM. Please install it with "
            "`pip install litellm`.",
            error_code=BAD_REQUEST,
        )

    try:
        response_dict = json.loads(response)
        feedback = Feedback(
            name=assessment_name,
            value=response_dict["result"],
            rationale=_sanitize_justification(response_dict.get("rationale", "")),
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=model_uri,
            ),
        )
    except json.JSONDecodeError as e:
        raise MlflowException(
            f"Failed to parse the response from the judge. Response: {response}",
            error_code=INVALID_PARAMETER_VALUE,
        ) from e

    return feedback


def _is_litellm_available() -> bool:
    try:
        import litellm  # noqa: F401

        return True
    except ImportError:
        return False


class _SuppressLiteLLMNonfatalErrors(ContextDecorator):
    """
    Thread-safe context manager and decorator to suppress LiteLLM's "Give Feedback" and
    "Provider List" messages. These messages indicate nonfatal bugs in the LiteLLM library;
    they are often noisy and can be safely ignored.

    Uses reference counting to ensure suppression remains active while any thread is running,
    preventing race conditions in parallel execution.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.count = 0
        self.original_litellm_settings = {}

    def __enter__(self) -> "_SuppressLiteLLMNonfatalErrors":
        try:
            import litellm
        except ImportError:
            return self

        with self.lock:
            if self.count == 0:
                # First caller - store original settings and enable suppression
                self.original_litellm_settings = {
                    "set_verbose": getattr(litellm, "set_verbose", None),
                    "suppress_debug_info": getattr(litellm, "suppress_debug_info", None),
                }
                litellm.set_verbose = False
                litellm.suppress_debug_info = True
            self.count += 1

        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any | None,
    ) -> bool:
        try:
            import litellm
        except ImportError:
            return False

        with self.lock:
            self.count -= 1
            if self.count == 0:
                # Last caller - restore original settings
                if (
                    original_verbose := self.original_litellm_settings.get("set_verbose")
                ) is not None:
                    litellm.set_verbose = original_verbose
                if (
                    original_suppress := self.original_litellm_settings.get("suppress_debug_info")
                ) is not None:
                    litellm.suppress_debug_info = original_suppress
                self.original_litellm_settings.clear()

        return False


# Global instance for use as threadsafe decorator
_suppress_litellm_nonfatal_errors = _SuppressLiteLLMNonfatalErrors()


@_suppress_litellm_nonfatal_errors
def _invoke_litellm(
    provider: str,
    model_name: str,
    messages: list["ChatMessage"],
    trace: Trace | None,
    num_retries: int,
) -> str:
    """
    Invoke the judge via litellm with retry support.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic').
        model_name: The model name.
        messages: List of ChatMessage objects.
        trace: Optional trace object for context with tool calling support.
        num_retries: Number of retries with exponential backoff on transient failures.

    Returns:
        The model's response content.

    Raises:
        MlflowException: If the request fails after all retries.
    """
    import litellm

    # Import at function level to avoid circular imports
    # (tools.registry imports from utils for invoke_judge_model)
    from mlflow.genai.judges.tools import list_judge_tools
    from mlflow.genai.judges.tools.registry import _judge_tool_registry

    # Convert ChatMessage objects to dicts for litellm
    messages_dict = [{"role": msg.role, "content": msg.content} for msg in messages]

    litellm_model_uri = f"{provider}/{model_name}"
    tools = []

    if trace is not None:
        judge_tools = list_judge_tools()
        tools = [tool.get_definition().to_dict() for tool in judge_tools]

    def _make_completion_request(messages: list[litellm.Message], include_response_format: bool):
        """Helper to make litellm completion request with optional response_format."""
        kwargs = {
            "model": litellm_model_uri,
            "messages": messages,
            "tools": tools if tools else None,
            "tool_choice": "auto" if tools else None,
            "retry_policy": _get_litellm_retry_policy(num_retries),
            "retry_strategy": "exponential_backoff_retry",
            # In LiteLLM version 1.55.3+, max_retries is stacked on top of retry_policy.
            # To avoid double-retry, we set max_retries=0
            "max_retries": 0,
            # Drop any parameters that are known to be unsupported by the LLM.
            # This is important for compatibility with certain models that don't support
            # certain call parameters (e.g. GPT-4 doesn't support 'response_format')
            "drop_params": True,
        }
        if include_response_format:
            kwargs["response_format"] = _get_judge_response_format()
        return litellm.completion(**kwargs)

    include_response_format = _MODEL_RESPONSE_FORMAT_CAPABILITIES.get(litellm_model_uri, True)
    while True:
        try:
            messages_dict = _prune_messages_over_context_length(
                messages=messages_dict, model_name=litellm_model_uri, max_tokens=100000
            )
            try:
                response = _make_completion_request(
                    messages_dict, include_response_format=include_response_format
                )
            except (litellm.BadRequestError, litellm.UnsupportedParamsError) as e:
                # Check whether the request attempted to use structured outputs, rather than
                # checking whether the model supports structured outputs in the capabilities cache,
                # since the capabilities cache may have been updated between the time that
                # include_response_format was set and the request was made
                if include_response_format:
                    # Retry without response_format if the request failed due to unsupported params.
                    # Some models don't support structured outputs (response_format) at all,
                    # and some models don't support both tool calling and structured outputs.
                    _logger.debug(
                        f"Model {litellm_model_uri} may not support structured outputs or combined "
                        f"tool calling + structured outputs. Error: {e}. "
                        f"Falling back to unstructured response."
                    )
                    # Cache the capability for future calls
                    _MODEL_RESPONSE_FORMAT_CAPABILITIES[litellm_model_uri] = False
                    include_response_format = False
                    response = _make_completion_request(
                        messages_dict, include_response_format=False
                    )
                else:
                    # Already tried without response_format and still got error
                    raise

            message = response.choices[0].message
            if not message.tool_calls:
                return message.content

            messages_dict.append(message.model_dump())
            # TODO: Consider making tool calls concurrent for better performance.
            # Currently sequential for simplicity and to maintain order of results.
            for tool_call in message.tool_calls:
                try:
                    mlflow_tool_call = _create_mlflow_tool_call_from_litellm(
                        litellm_tool_call=tool_call
                    )
                    result = _judge_tool_registry.invoke(tool_call=mlflow_tool_call, trace=trace)
                except Exception as e:
                    messages_dict.append(
                        _create_litellm_tool_response_message(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.function.name,
                            content=f"Error: {e!s}",
                        )
                    )
                else:
                    # Convert dataclass results to dict if needed
                    # The tool result is either a dict, string, or dataclass
                    if is_dataclass(result):
                        result = asdict(result)
                    result_json = (
                        json.dumps(result, default=str) if not isinstance(result, str) else result
                    )
                    messages_dict.append(
                        _create_litellm_tool_response_message(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.function.name,
                            content=result_json,
                        )
                    )
        except Exception as e:
            raise MlflowException(f"Failed to invoke the judge via litellm: {e}") from e


def _create_mlflow_tool_call_from_litellm(litellm_tool_call) -> Any:
    """
    Create an MLflow ToolCall from a LiteLLM tool call.

    Args:
        litellm_tool_call: The LiteLLM ChatCompletionMessageToolCall object.

    Returns:
        An MLflow ToolCall object.
    """
    from mlflow.types.llm import ToolCall

    return ToolCall(
        id=litellm_tool_call.id,
        function={
            "name": litellm_tool_call.function.name,
            "arguments": litellm_tool_call.function.arguments,
        },
    )


def _create_litellm_tool_response_message(
    tool_call_id: str, tool_name: str, content: str
) -> dict[str, str]:
    """
    Create a tool response message for LiteLLM.

    Args:
        tool_call_id: The ID of the tool call being responded to.
        tool_name: The name of the tool that was invoked.
        content: The content to include in the response.

    Returns:
        A dictionary representing the tool response message.
    """
    return {
        "tool_call_id": tool_call_id,
        "role": "tool",
        "name": tool_name,
        "content": content,
    }


def _get_judge_response_format() -> dict[str, Any]:
    """
    Get the response format for judge evaluations.

    Returns:
        A dictionary containing the JSON schema for structured outputs.
    """
    # Import here to avoid circular imports
    from mlflow.genai.judges.base import Judge

    output_fields = Judge.get_output_fields()

    properties = {}
    required_fields = []

    for field in output_fields:
        properties[field.name] = {
            "type": "string",
            "description": field.description,
        }
        required_fields.append(field.name)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "judge_evaluation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required_fields,
                "additionalProperties": False,
            },
        },
    }


def _prune_messages_over_context_length(
    messages: list["litellm.Message"],  # noqa: F821
    model_name: str,
    max_tokens: int = 100000,
) -> list["litellm.Message"]:  # noqa: F821
    """
    Prune tool call messages from history to stay under token limit.

    Removes oldest tool call + tool response message pairs until under limit.

    Args:
        messages: List of LiteLLM message objects
        model_name: Model name for token counting
        max_tokens: Maximum token limit (default: 100,000)

    Returns:
        Pruned list of LiteLLM message objects under the token limit
    """
    import litellm

    initial_tokens = litellm.token_counter(model=model_name, messages=messages)
    if initial_tokens <= max_tokens:
        return messages
    pruned_messages = messages[:]
    # Remove tool call pairs until we're under limit
    while litellm.token_counter(model=model_name, messages=pruned_messages) > max_tokens:
        # Find first assistant message with tool calls
        assistant_msg = None
        assistant_idx = None
        for i, msg in enumerate(pruned_messages):
            if msg.role == "assistant" and msg.tool_calls:
                assistant_msg = msg
                assistant_idx = i
                break
        if assistant_msg is None:
            break  # No more tool calls to remove
        pruned_messages.pop(assistant_idx)
        # Remove corresponding tool response messages
        tool_call_ids = {tc.id for tc in assistant_msg.tool_calls}
        pruned_messages = [
            msg
            for msg in pruned_messages
            if not (msg.role == "tool" and msg.tool_call_id in tool_call_ids)
        ]

    final_tokens = litellm.token_counter(model=model_name, messages=pruned_messages)
    _logger.info(f"Pruned message history from {initial_tokens} to {final_tokens} tokens")
    return pruned_messages


def _get_litellm_retry_policy(num_retries: int):
    """
    Get a LiteLLM retry policy for retrying requests when transient API errors occur.

    Args:
        num_retries: The number of times to retry a request if it fails transiently due to
                     network error, rate limiting, etc. Requests are retried with exponential
                     backoff.

    Returns:
        A LiteLLM RetryPolicy instance.
    """
    from litellm import RetryPolicy

    return RetryPolicy(
        TimeoutErrorRetries=num_retries,
        RateLimitErrorRetries=num_retries,
        InternalServerErrorRetries=num_retries,
        ContentPolicyViolationErrorRetries=num_retries,
        # We don't retry on errors that are unlikely to be transient
        # (e.g. bad request, invalid auth credentials)
        BadRequestErrorRetries=0,
        AuthenticationErrorRetries=0,
    )


class CategoricalRating(StrEnum):
    """
    A categorical rating for an assessment.

    Example:
        .. code-block:: python

            from mlflow.genai.judges import CategoricalRating
            from mlflow.entities import Feedback

            # Create feedback with categorical rating
            feedback = Feedback(
                name="my_metric", value=CategoricalRating.YES, rationale="The metric is passing."
            )
    """

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"

    @classmethod
    def _missing_(cls, value: str):
        value = value.lower()
        for member in cls:
            if member == value:
                return member
        return cls.UNKNOWN
