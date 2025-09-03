from __future__ import annotations

import json
import logging
import re
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

    # Convert to uniform dict format for internal processing
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        # Convert ChatMessage objects to dicts
        messages = [{"role": msg.role, "content": msg.content} for msg in prompt]

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
        response = score_model_on_payload(
            model_uri=model_uri,
            payload=messages,
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


def _invoke_litellm(
    provider: str,
    model_name: str,
    messages: list[dict[str, Any]],
    trace: Trace | None,
    num_retries: int,
) -> str:
    """
    Invoke the judge via litellm with retry support.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic').
        model_name: The model name.
        messages: List of message dicts with 'role' and 'content' keys.
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

    litellm_model_uri = f"{provider}/{model_name}"
    tools = []
    response_format = _get_judge_response_format()

    if trace is not None:
        judge_tools = list_judge_tools()
        tools = [tool.get_definition().to_dict() for tool in judge_tools]

    def _make_completion_request(include_response_format: bool):
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
        }
        if include_response_format:
            kwargs["response_format"] = response_format
        return litellm.completion(**kwargs)

    while True:
        try:
            try:
                response = _make_completion_request(
                    include_response_format=_MODEL_RESPONSE_FORMAT_CAPABILITIES.get(
                        litellm_model_uri, True
                    )
                )
            except litellm.BadRequestError as e:
                if _MODEL_RESPONSE_FORMAT_CAPABILITIES.get(litellm_model_uri, True):
                    # Retry without response_format if the request failed due to bad request.
                    # Some models don't support structured outputs (response_format) at all,
                    # and some models don't support both tool calling and structured outputs.
                    _logger.debug(
                        f"Model {litellm_model_uri} may not support structured outputs or combined "
                        f"tool calling + structured outputs. BadRequestError: {e}. "
                        f"Falling back to unstructured response."
                    )
                    # Cache the capability for future calls
                    _MODEL_RESPONSE_FORMAT_CAPABILITIES[litellm_model_uri] = False
                    response = _make_completion_request(include_response_format=False)
                else:
                    # Already tried without response_format and still got BadRequestError
                    raise

            message = response.choices[0].message
            if not message.tool_calls:
                return message.content

            messages.append(message.model_dump())
            # TODO: Consider making tool calls concurrent for better performance.
            # Currently sequential for simplicity and to maintain order of results.
            for tool_call in message.tool_calls:
                try:
                    mlflow_tool_call = _create_mlflow_tool_call_from_litellm(
                        litellm_tool_call=tool_call
                    )
                    result = _judge_tool_registry.invoke(tool_call=mlflow_tool_call, trace=trace)
                except Exception as e:
                    messages.append(
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
                    messages.append(
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
