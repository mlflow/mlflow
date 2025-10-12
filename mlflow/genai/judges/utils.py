from __future__ import annotations

import json
import logging
import re
import threading
import time
import traceback
import warnings
from contextlib import ContextDecorator
from dataclasses import asdict, dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import pydantic
import requests

if TYPE_CHECKING:
    import litellm

    from mlflow.genai.judges.base import AlignmentOptimizer, JudgeField
    from mlflow.types.llm import ChatMessage, ToolCall

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.environment_variables import MLFLOW_JUDGE_MAX_ITERATIONS
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.utils.enum_utils import StrEnum
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INVALID_PARAMETER_VALUE,
    REQUEST_LIMIT_EXCEEDED,
)
from mlflow.telemetry.events import InvokeCustomJudgeModelEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.telemetry.utils import _is_in_databricks
from mlflow.utils.uri import is_databricks_uri
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)

# "endpoints" is a special case for Databricks model serving endpoints.
_NATIVE_PROVIDERS = ["openai", "anthropic", "bedrock", "mistral", "endpoints"]

# Global cache to track model capabilities across function calls
# Key: model URI (e.g., "openai/gpt-4"), Value: boolean indicating response_format support
_MODEL_RESPONSE_FORMAT_CAPABILITIES: dict[str, bool] = {}


class DatabricksLLMJudgePrompts(NamedTuple):
    """Result of splitting ChatMessage list for Databricks API."""

    system_prompt: str | None
    user_prompt: str


def _check_databricks_agents_installed() -> None:
    """Check if databricks-agents is installed for databricks judge functionality.

    Raises:
        MlflowException: If databricks-agents is not installed.
    """
    try:
        import databricks.agents.evals  # noqa: F401
    except ImportError:
        raise MlflowException(
            f"To use '{_DATABRICKS_DEFAULT_JUDGE_MODEL}' as the judge model, the Databricks "
            "agents library must be installed. Please install it with: "
            "`pip install databricks-agents`",
            error_code=BAD_REQUEST,
        )


def get_default_model() -> str:
    if is_databricks_uri(mlflow.get_tracking_uri()):
        return _DATABRICKS_DEFAULT_JUDGE_MODEL
    else:
        return "openai:/gpt-4.1-mini"


def get_default_optimizer() -> AlignmentOptimizer:
    """
    Get the default alignment optimizer.

    Returns:
        A SIMBA alignment optimizer with no model specified (uses default model).
    """
    from mlflow.genai.judges.optimizers.simba import SIMBAAlignmentOptimizer

    return SIMBAAlignmentOptimizer()


def _is_litellm_available() -> bool:
    """Check if LiteLLM is available for import."""
    try:
        import litellm  # noqa: F401

        return True
    except ImportError:
        return False


def validate_judge_model(model_uri: str) -> None:
    """
    Validate that a judge model URI is valid and has required dependencies.

    This function performs early validation at judge construction time to provide
    fast feedback about configuration issues.

    Args:
        model_uri: The model URI to validate (e.g., "databricks", "openai:/gpt-4")

    Raises:
        MlflowException: If the model URI is invalid or required dependencies are missing.
    """
    from mlflow.metrics.genai.model_utils import _parse_model_uri

    # Special handling for Databricks default model
    if model_uri == _DATABRICKS_DEFAULT_JUDGE_MODEL:
        # Check if databricks-agents is available
        _check_databricks_agents_installed()
        return

    # Validate the URI format and extract provider
    provider, model_name = _parse_model_uri(model_uri)

    # Check if LiteLLM is required and available for non-native providers
    if provider not in _NATIVE_PROVIDERS:
        if not _is_litellm_available():
            raise MlflowException(
                f"LiteLLM is required for using '{provider}' as a provider. "
                "Please install it with: `pip install litellm`",
                error_code=INVALID_PARAMETER_VALUE,
            )


def format_prompt(prompt: str, **values) -> str:
    """Format double-curly variables in the prompt template."""
    for key, value in values.items():
        # Escape backslashes in the replacement string to prevent re.sub from interpreting
        # them as escape sequences (e.g. \u being treated as Unicode escape)
        replacement = str(value).replace("\\", "\\\\")
        prompt = re.sub(r"\{\{\s*" + key + r"\s*\}\}", replacement, prompt)
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


def _strip_markdown_code_blocks(response: str) -> str:
    """
    Strip markdown code blocks from LLM responses.

    Some legacy models wrap JSON responses in markdown code blocks (```json...```).
    This function removes those wrappers to extract the raw JSON content.

    Args:
        response: The raw response from the LLM

    Returns:
        The response with markdown code blocks removed
    """
    cleaned = response.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.split("\n")
    start_idx = 0
    end_idx = len(lines)

    for i, line in enumerate(lines):
        if i == 0 and line.startswith("```"):
            start_idx = 1
        elif line.strip() == "```" and i > 0:
            end_idx = i
            break

    return "\n".join(lines[start_idx:end_idx])


def _sanitize_justification(justification: str) -> str:
    # Some judge prompts instruct the model to think step by step.
    return justification.replace("Let's think step by step. ", "")


def _split_messages_for_databricks(messages: list["ChatMessage"]) -> DatabricksLLMJudgePrompts:
    """
    Split a list of ChatMessage objects into system and user prompts for Databricks API.

    Args:
        messages: List of ChatMessage objects to split.

    Returns:
        DatabricksLLMJudgePrompts namedtuple with system_prompt and user_prompt fields.
        The system_prompt may be None.

    Raises:
        MlflowException: If the messages list is empty or invalid.
    """
    from mlflow.types.llm import ChatMessage

    if not messages:
        raise MlflowException(
            "Invalid prompt format: expected non-empty list of ChatMessage",
            error_code=BAD_REQUEST,
        )

    system_prompt = None
    user_parts = []

    for msg in messages:
        if isinstance(msg, ChatMessage):
            if msg.role == "system":
                # Use the first system message as the actual system prompt for the API.
                # Any subsequent system messages are appended to the user prompt to preserve
                # their content and maintain the order in which they appear in the submitted
                # evaluation payload.
                if system_prompt is None:
                    system_prompt = msg.content
                else:
                    user_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                user_parts.append(msg.content)
            elif msg.role == "assistant":
                user_parts.append(f"Assistant: {msg.content}")

    user_prompt = "\n\n".join(user_parts) if user_parts else ""

    return DatabricksLLMJudgePrompts(system_prompt=system_prompt, user_prompt=user_prompt)


def _parse_databricks_judge_response(
    llm_output: str | None,
    assessment_name: str,
) -> Feedback:
    """
    Parse the response from Databricks judge into a Feedback object.

    Args:
        llm_output: Raw output from the LLM, or None if no response.
        assessment_name: Name of the assessment.

    Returns:
        Feedback object with parsed results or error.
    """
    source = AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE, source_id=_DATABRICKS_DEFAULT_JUDGE_MODEL
    )
    if not llm_output:
        return Feedback(
            name=assessment_name,
            error="Empty response from Databricks judge",
            source=source,
        )
    try:
        response_data = json.loads(llm_output)
    except json.JSONDecodeError as e:
        return Feedback(
            name=assessment_name,
            error=f"Invalid JSON response from Databricks judge: {e}",
            source=source,
        )
    if "result" not in response_data:
        return Feedback(
            name=assessment_name,
            error=f"Response missing 'result' field: {response_data}",
            source=source,
        )
    return Feedback(
        name=assessment_name,
        value=response_data["result"],
        rationale=response_data.get("rationale", ""),
        source=source,
    )


def call_chat_completions(user_prompt: str, system_prompt: str) -> Any:
    """
    Invokes the Databricks chat completions API using the databricks.agents.evals library.

    Args:
        user_prompt (str): The user prompt.
        system_prompt (str): The system prompt.

    Returns:
        The chat completions result.

    Raises:
        MlflowException: If databricks-agents is not installed.
    """
    _check_databricks_agents_installed()

    from databricks.rag_eval import context, env_vars

    env_vars.RAG_EVAL_EVAL_SESSION_CLIENT_NAME.set(f"mlflow-judge-optimizer-v{VERSION}")

    @context.eval_context
    def _call_chat_completions(user_prompt: str, system_prompt: str):
        managed_rag_client = context.get_context().build_managed_rag_client()

        return managed_rag_client.get_chat_completions_result(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

    return _call_chat_completions(user_prompt, system_prompt)


def _invoke_databricks_judge(
    prompt: str | list["ChatMessage"],
    assessment_name: str,
) -> Feedback:
    """
    Invoke the Databricks judge using the databricks.agents.evals library.

    Uses the direct chat completions API for clean prompt submission without
    any additional formatting or template requirements.

    Args:
        prompt: The formatted prompt with template variables filled in.
        assessment_name: The name of the assessment.

    Returns:
        Feedback object from the Databricks judge.

    Raises:
        MlflowException: If databricks-agents is not installed.
    """
    try:
        if isinstance(prompt, str):
            system_prompt = None
            user_prompt = prompt
        else:
            prompts = _split_messages_for_databricks(prompt)
            system_prompt = prompts.system_prompt
            user_prompt = prompts.user_prompt

        llm_result = call_chat_completions(user_prompt, system_prompt)
        return _parse_databricks_judge_response(llm_result.output, assessment_name)

    except Exception as e:
        _logger.debug(f"Failed to invoke Databricks judge: {e}")
        return Feedback(
            name=assessment_name,
            error=f"Failed to invoke Databricks judge: {e}",
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=_DATABRICKS_DEFAULT_JUDGE_MODEL,
            ),
        )


def _invoke_via_gateway(
    model_uri: str,
    provider: str,
    prompt: str,
) -> str:
    """
    Invoke the judge model via native AI Gateway adapters.

    Args:
        model_uri: The full model URI.
        provider: The provider name.
        prompt: The prompt to evaluate.

    Returns:
        The JSON response string from the model.

    Raises:
        MlflowException: If the provider is not natively supported or invocation fails.
    """
    from mlflow.metrics.genai.model_utils import get_endpoint_type, score_model_on_payload

    if provider not in _NATIVE_PROVIDERS:
        raise MlflowException(
            f"LiteLLM is required for using '{provider}' LLM. Please install it with "
            "`pip install litellm`.",
            error_code=BAD_REQUEST,
        )

    return score_model_on_payload(
        model_uri=model_uri,
        payload=prompt,
        endpoint_type=get_endpoint_type(model_uri) or "llm/v1/chat",
    )


class FieldExtraction(pydantic.BaseModel):
    """Schema for extracting inputs and outputs from traces using LLM."""

    inputs: str = pydantic.Field(description="The user's original request or question")
    outputs: str = pydantic.Field(description="The system's final response")


@record_usage_event(InvokeCustomJudgeModelEvent)
def invoke_judge_model(
    model_uri: str,
    prompt: str | list["ChatMessage"],
    assessment_name: str,
    trace: Trace | None = None,
    num_retries: int = 10,
) -> Feedback:
    """
    Invoke the judge model.

    Routes to the appropriate implementation based on the model URI:
    - "databricks": Uses databricks.agents.evals library for default judge,
                    direct API for regular endpoints
    - LiteLLM-supported providers: Uses LiteLLM if available
    - Native providers: Falls back to AI Gateway adapters

    Args:
        model_uri: The model URI.
        prompt: The prompt to evaluate. Can be a string (single prompt) or
                a list of ChatMessage objects.
        assessment_name: The name of the assessment.
        trace: Optional trace object for context.
        num_retries: Number of retries on transient failures when using litellm.

    Returns:
        Feedback object with the judge's assessment.

    Raises:
        MlflowException: If the model cannot be invoked or dependencies are missing.
    """
    if model_uri == _DATABRICKS_DEFAULT_JUDGE_MODEL:
        return _invoke_databricks_judge(prompt, assessment_name)

    from mlflow.metrics.genai.model_utils import _parse_model_uri
    from mlflow.types.llm import ChatMessage

    model_provider, model_name = _parse_model_uri(model_uri)
    in_databricks = _is_in_databricks()

    # Handle Databricks endpoints (not the default judge) with proper telemetry
    if model_provider in {"databricks", "endpoints"} and isinstance(prompt, str):
        if model_provider == "endpoints":
            warnings.warn(
                "The legacy provider 'endpoints' is deprecated and will be removed in a future"
                "release. Please update your code to use the 'databricks' provider instead.",
                FutureWarning,
                stacklevel=2,
            )
        try:
            output = _invoke_databricks_judge_model(
                model_name=model_name,
                prompt=prompt,
                assessment_name=assessment_name,
                num_retries=num_retries,
            )
            feedback = output.feedback
            feedback.trace_id = trace.info.trace_id if trace is not None else None

            # Record success telemetry only when in Databricks
            if in_databricks:
                try:
                    _record_judge_model_usage_success_databricks_telemetry(
                        request_id=output.request_id,
                        model_provider=output.model_provider,
                        endpoint_name=output.model_name,
                        num_prompt_tokens=output.num_prompt_tokens,
                        num_completion_tokens=output.num_completion_tokens,
                    )
                except Exception as telemetry_error:
                    _logger.debug(
                        "Failed to record judge model usage success telemetry. Error: %s",
                        telemetry_error,
                    )

            return feedback

        except Exception:
            # Record failure telemetry only when in Databricks
            if in_databricks:
                try:
                    provider = "databricks" if model_provider == "endpoints" else model_provider
                    _record_judge_model_usage_failure_databricks_telemetry(
                        model_provider=provider,
                        endpoint_name=model_name,
                        error_code="UNKNOWN",
                        error_message=traceback.format_exc(),
                    )
                except Exception as telemetry_error:
                    _logger.debug(
                        "Failed to record judge model usage failure telemetry. Error: %s",
                        telemetry_error,
                    )
            raise

    # Handle all other cases (including non-Databricks, ChatMessage prompts, traces)
    messages = [ChatMessage(role="user", content=prompt)] if isinstance(prompt, str) else prompt

    if _is_litellm_available():
        response = _invoke_litellm_and_handle_tools(
            provider=model_provider,
            model_name=model_name,
            messages=messages,
            trace=trace,
            num_retries=num_retries,
        )
    elif trace is not None:
        raise MlflowException(
            "LiteLLM is required for using traces with judges. "
            "Please install it with `pip install litellm`.",
            error_code=BAD_REQUEST,
        )
    else:
        if not isinstance(prompt, str):
            raise MlflowException(
                "This judge is not supported by native LLM providers. Please install "
                "LiteLLM with `pip install litellm` to use this judge.",
                error_code=BAD_REQUEST,
            )
        response = _invoke_via_gateway(model_uri, model_provider, prompt)

    cleaned_response = _strip_markdown_code_blocks(response)

    try:
        response_dict = json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        raise MlflowException(
            f"Failed to parse response from judge model. Response: {response}",
            error_code=BAD_REQUEST,
        ) from e

    feedback = Feedback(
        name=assessment_name,
        value=response_dict["result"],
        rationale=_sanitize_justification(response_dict.get("rationale", "")),
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id=model_uri),
        trace_id=trace.info.trace_id if trace is not None else None,
    )

    if "error" in response_dict:
        feedback.error = response_dict["error"]
        raise MlflowException(
            f"Judge evaluation failed with error: {response_dict['error']}",
            error_code=INVALID_PARAMETER_VALUE,
        )

    return feedback


def get_chat_completions_with_structured_output(
    model_uri: str,
    messages: list["ChatMessage"],
    output_schema: type[pydantic.BaseModel],
    trace: Trace | None = None,
    num_retries: int = 10,
) -> pydantic.BaseModel:
    """
    Get chat completions from an LLM with structured output conforming to a Pydantic schema.

    This function invokes an LLM and ensures the response matches the provided Pydantic schema.
    When a trace is provided, the LLM can use tool calling to examine trace spans.

    Args:
        model_uri: The model URI (e.g., "openai:/gpt-4", "anthropic:/claude-3").
        messages: List of ChatMessage objects for the conversation with the LLM.
        output_schema: Pydantic model class defining the expected output structure.
                       The LLM will be instructed to return data matching this schema.
        trace: Optional trace object for context. When provided, enables tool
               calling to examine trace spans.
        num_retries: Number of retries on transient failures. Defaults to 10 with
                     exponential backoff.

    Returns:
        Instance of output_schema with the structured data from the LLM.

    Raises:
        ImportError: If LiteLLM is not installed.
        JSONDecodeError: If the LLM response cannot be parsed as JSON.
        ValidationError: If the LLM response does not match the output schema.

    Example:
        .. code-block:: python

            from pydantic import BaseModel, Field
            from mlflow.genai.judges.utils import get_chat_completions_with_structured_output
            from mlflow.types.llm import ChatMessage


            class FieldExtraction(BaseModel):
                inputs: str = Field(description="The user's original request")
                outputs: str = Field(description="The system's final response")


            # Extract fields from a trace where root span lacks input/output
            # but nested spans contain the actual data
            result = get_chat_completions_with_structured_output(
                model_uri="openai:/gpt-4",
                messages=[
                    ChatMessage(role="system", content="Extract fields from the trace"),
                    ChatMessage(role="user", content="Find the inputs and outputs"),
                ],
                output_schema=FieldExtraction,
                trace=trace,  # Trace with nested spans containing actual data
            )
            print(result.inputs)  # Extracted from inner span
            print(result.outputs)  # Extracted from inner span
    """
    from mlflow.metrics.genai.model_utils import _parse_model_uri

    model_provider, model_name = _parse_model_uri(model_uri)

    response = _invoke_litellm_and_handle_tools(
        provider=model_provider,
        model_name=model_name,
        messages=messages,
        trace=trace,
        num_retries=num_retries,
        response_format=output_schema,
    )

    cleaned_response = _strip_markdown_code_blocks(response)
    response_dict = json.loads(cleaned_response)
    return output_schema(**response_dict)


@dataclass
class InvokeDatabricksModelOutput:
    response: str
    request_id: str | None
    num_prompt_tokens: int | None
    num_completion_tokens: int | None


def _parse_databricks_model_response(
    res_json: dict[str, Any], headers: dict[str, Any]
) -> InvokeDatabricksModelOutput:
    """
    Parse and validate the response from a Databricks model invocation.

    Args:
        res_json: The JSON response from the model
        headers: The response headers

    Returns:
        InvokeDatabricksModelOutput with parsed response data

    Raises:
        MlflowException: If the response structure is invalid
    """
    # Validate and extract choices
    choices = res_json.get("choices", [])
    if not choices:
        raise MlflowException(
            "Invalid response from Databricks model: missing 'choices' field",
            error_code=INVALID_PARAMETER_VALUE,
        )

    first_choice = choices[0]
    if "message" not in first_choice:
        raise MlflowException(
            "Invalid response from Databricks model: missing 'message' field",
            error_code=INVALID_PARAMETER_VALUE,
        )

    content = first_choice.get("message", {}).get("content")
    if content is None:
        raise MlflowException(
            "Invalid response from Databricks model: missing 'content' field",
            error_code=INVALID_PARAMETER_VALUE,
        )

    # Handle reasoning response (list of content items)
    if isinstance(content, list):
        text_content = None
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_content = item.get("text")
                break

        if text_content is None:
            raise MlflowException(
                "Invalid reasoning response: no text content found in response list",
                error_code=INVALID_PARAMETER_VALUE,
            )
        content = text_content

    usage = res_json.get("usage", {})

    return InvokeDatabricksModelOutput(
        response=content,
        request_id=headers.get("x-request-id"),
        num_prompt_tokens=usage.get("prompt_tokens"),
        num_completion_tokens=usage.get("completion_tokens"),
    )


def _invoke_databricks_model(
    *, model_name: str, prompt: str, num_retries: int
) -> InvokeDatabricksModelOutput:
    from mlflow.utils.databricks_utils import get_databricks_host_creds

    host_creds = get_databricks_host_creds()
    api_url = f"{host_creds.host}/serving-endpoints/{model_name}/invocations"

    # Implement retry logic with exponential backoff
    last_exception = None
    for attempt in range(num_retries + 1):
        try:
            res = requests.post(
                url=api_url,
                headers={"Authorization": f"Bearer {host_creds.token}"},
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                },
            )
        except (requests.RequestException, requests.ConnectionError) as e:
            last_exception = e
            if attempt < num_retries:
                _logger.debug(f"Request attempt {attempt + 1} failed with error: {e}")
                time.sleep(2**attempt)  # Exponential backoff
                continue
            else:
                raise MlflowException(
                    f"Failed to invoke Databricks model after {num_retries + 1} attempts: {e}",
                    error_code=INVALID_PARAMETER_VALUE,
                ) from e

        # Check HTTP status before parsing JSON
        if res.status_code in [400, 401, 403, 404]:
            # Don't retry on bad request, unauthorized, not found, or forbidden
            raise MlflowException(
                f"Databricks model invocation failed with status {res.status_code}: {res.text}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if res.status_code >= 400:
            # For other errors, raise exception and potentially retry
            error_msg = (
                f"Databricks model invocation failed with status {res.status_code}: {res.text}"
            )
            if attempt < num_retries:
                # Log and retry for transient errors
                _logger.debug(f"Attempt {attempt + 1} failed: {error_msg}")
                time.sleep(2**attempt)  # Exponential backoff
                continue
            else:
                raise MlflowException(error_msg, error_code=INVALID_PARAMETER_VALUE)

        # Parse JSON response
        try:
            res_json = res.json()
        except json.JSONDecodeError as e:
            raise MlflowException(
                f"Failed to parse JSON response from Databricks model: {e}",
                error_code=INVALID_PARAMETER_VALUE,
            ) from e

        # Parse and validate the response using helper function
        return _parse_databricks_model_response(res_json, res.headers)

    # This should not be reached, but just in case
    if last_exception:
        raise MlflowException(
            f"Failed to invoke Databricks model: {last_exception}",
            error_code=INVALID_PARAMETER_VALUE,
        ) from last_exception


def _record_judge_model_usage_success_databricks_telemetry(
    *,
    request_id: str | None,
    model_provider: str,
    endpoint_name: str,
    num_prompt_tokens: int | None,
    num_completion_tokens: int | None,
) -> None:
    try:
        from databricks.agents.telemetry import record_judge_model_usage_success
    except ImportError:
        _logger.debug(
            "Failed to import databricks.agents.telemetry.record_judge_model_usage_success; "
            "databricks-agents needs to be installed."
        )
        return

    from mlflow.tracking.fluent import _get_experiment_id
    from mlflow.utils.databricks_utils import get_job_id, get_job_run_id, get_workspace_id

    record_judge_model_usage_success(
        request_id=request_id,
        experiment_id=_get_experiment_id(),
        job_id=get_job_id(),
        job_run_id=get_job_run_id(),
        workspace_id=get_workspace_id(),
        model_provider=model_provider,
        endpoint_name=endpoint_name,
        num_prompt_tokens=num_prompt_tokens,
        num_completion_tokens=num_completion_tokens,
    )


def _record_judge_model_usage_failure_databricks_telemetry(
    *,
    model_provider: str,
    endpoint_name: str,
    error_code: str,
    error_message: str,
) -> None:
    try:
        from databricks.agents.telemetry import record_judge_model_usage_failure
    except ImportError:
        _logger.debug(
            "Failed to import databricks.agents.telemetry.record_judge_model_usage_success; "
            "databricks-agents needs to be installed."
        )
        return

    from mlflow.tracking.fluent import _get_experiment_id
    from mlflow.utils.databricks_utils import get_job_id, get_job_run_id, get_workspace_id

    record_judge_model_usage_failure(
        experiment_id=_get_experiment_id(),
        job_id=get_job_id(),
        job_run_id=get_job_run_id(),
        workspace_id=get_workspace_id(),
        model_provider=model_provider,
        endpoint_name=endpoint_name,
        error_code=error_code,
        error_message=error_message,
    )


@dataclass
class InvokeJudgeModelHelperOutput:
    feedback: Feedback
    model_provider: str
    model_name: str
    request_id: str | None
    num_prompt_tokens: int | None
    num_completion_tokens: int | None


def _invoke_databricks_judge_model(
    *,
    model_name: str,
    prompt: str,
    assessment_name: str,
    num_retries: int = 10,
) -> InvokeJudgeModelHelperOutput:
    output = _invoke_databricks_model(
        model_name=model_name,
        prompt=prompt,
        num_retries=num_retries,
    )
    try:
        response_dict = json.loads(output.response)
        feedback = Feedback(
            name=assessment_name,
            value=response_dict["result"],
            rationale=_sanitize_justification(response_dict.get("rationale", "")),
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=f"databricks:/{model_name}",
            ),
        )
    except json.JSONDecodeError as e:
        raise MlflowException(
            f"Failed to parse the response from the judge. Response: {output.response}",
            error_code=INVALID_PARAMETER_VALUE,
        ) from e

    return InvokeJudgeModelHelperOutput(
        feedback=feedback,
        model_provider="databricks",
        model_name=model_name,
        request_id=output.request_id,
        num_prompt_tokens=output.num_prompt_tokens,
        num_completion_tokens=output.num_completion_tokens,
    )


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


def _invoke_litellm(
    litellm_model_uri: str,
    messages: list["litellm.Message"],
    tools: list[dict[str, Any]],
    num_retries: int,
    response_format: type[pydantic.BaseModel] | None,
    include_response_format: bool,
) -> "litellm.ModelResponse":
    """
    Invoke litellm completion with retry support.

    Args:
        litellm_model_uri: Full model URI for litellm (e.g., "openai/gpt-4").
        messages: List of litellm Message objects.
        tools: List of tool definitions (empty list if no tools).
        num_retries: Number of retries with exponential backoff.
        response_format: Optional Pydantic model class for structured output.
        include_response_format: Whether to include response_format in the request.

    Returns:
        The litellm ModelResponse object.

    Raises:
        Various litellm exceptions on failure.
    """
    import litellm

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
        # LiteLLM supports passing Pydantic models directly for response_format
        kwargs["response_format"] = response_format or _get_default_judge_response_schema()
    return litellm.completion(**kwargs)


def _process_tool_calls(
    tool_calls: list["litellm.ChatCompletionMessageToolCall"],
    trace: Trace | None,
) -> list["litellm.Message"]:
    """
    Process tool calls and return tool response messages.

    Args:
        tool_calls: List of tool calls from the LLM response.
        trace: Optional trace object for context.

    Returns:
        List of litellm Message objects containing tool responses.
    """
    from mlflow.genai.judges.tools.registry import _judge_tool_registry

    tool_response_messages = []
    for tool_call in tool_calls:
        try:
            mlflow_tool_call = _create_mlflow_tool_call_from_litellm(litellm_tool_call=tool_call)
            result = _judge_tool_registry.invoke(tool_call=mlflow_tool_call, trace=trace)
        except Exception as e:
            tool_response_messages.append(
                _create_litellm_tool_response_message(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    content=f"Error: {e!s}",
                )
            )
        else:
            if is_dataclass(result):
                result = asdict(result)
            result_json = json.dumps(result, default=str) if not isinstance(result, str) else result
            tool_response_messages.append(
                _create_litellm_tool_response_message(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    content=result_json,
                )
            )
    return tool_response_messages


@_suppress_litellm_nonfatal_errors
def _invoke_litellm_and_handle_tools(
    provider: str,
    model_name: str,
    messages: list["ChatMessage"],
    trace: Trace | None,
    num_retries: int,
    response_format: type[pydantic.BaseModel] | None = None,
) -> str:
    """
    Invoke litellm with retry support and handle tool calling loop.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic').
        model_name: The model name.
        messages: List of ChatMessage objects.
        trace: Optional trace object for context with tool calling support.
        num_retries: Number of retries with exponential backoff on transient failures.
        response_format: Optional Pydantic model class for structured output format.
                       Used by get_chat_completions_with_structured_output for
                       schema-based extraction.

    Returns:
        The model's response content.

    Raises:
        MlflowException: If the request fails after all retries.
    """
    import litellm

    from mlflow.genai.judges.tools import list_judge_tools

    messages = [litellm.Message(role=msg.role, content=msg.content) for msg in messages]

    litellm_model_uri = f"{provider}/{model_name}"
    tools = []

    if trace is not None:
        judge_tools = list_judge_tools()
        tools = [tool.get_definition().to_dict() for tool in judge_tools]

    def _prune_messages_for_context_window():
        try:
            max_context_length = litellm.get_max_tokens(litellm_model_uri)
        except Exception:
            max_context_length = None

        return _prune_messages_exceeding_context_window_length(
            messages=messages,
            model=litellm_model_uri,
            max_tokens=max_context_length or 100000,
        )

    include_response_format = _MODEL_RESPONSE_FORMAT_CAPABILITIES.get(litellm_model_uri, True)

    max_iterations = MLFLOW_JUDGE_MAX_ITERATIONS.get()
    iteration_count = 0

    while True:
        iteration_count += 1
        if iteration_count > max_iterations:
            raise MlflowException(
                f"Completion iteration limit of {max_iterations} exceeded. "
                f"This usually indicates the model is not powerful enough to effectively "
                f"analyze the trace. Consider using a more intelligent/powerful model. "
                f"In rare cases, for very complex traces where a large number of completion "
                f"iterations might be required, you can increase the number of iterations by "
                f"modifying the {MLFLOW_JUDGE_MAX_ITERATIONS.name} environment variable.",
                error_code=REQUEST_LIMIT_EXCEEDED,
            )
        try:
            try:
                response = _invoke_litellm(
                    litellm_model_uri=litellm_model_uri,
                    messages=messages,
                    tools=tools,
                    num_retries=num_retries,
                    response_format=response_format,
                    include_response_format=include_response_format,
                )
            except (litellm.BadRequestError, litellm.UnsupportedParamsError) as e:
                if isinstance(e, litellm.ContextWindowExceededError) or "context length" in str(e):
                    messages = _prune_messages_for_context_window()
                    continue
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
                    _MODEL_RESPONSE_FORMAT_CAPABILITIES[litellm_model_uri] = False
                    include_response_format = False
                    continue
                else:
                    raise

            message = response.choices[0].message
            if not message.tool_calls:
                return message.content

            messages.append(message)
            tool_response_messages = _process_tool_calls(tool_calls=message.tool_calls, trace=trace)
            messages.extend(tool_response_messages)

        except MlflowException:
            raise
        except Exception as e:
            raise MlflowException(f"Failed to invoke the judge via litellm: {e}") from e


def _create_mlflow_tool_call_from_litellm(
    litellm_tool_call: "litellm.ChatCompletionMessageToolCall",
) -> "ToolCall":
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
) -> "litellm.Message":
    """
    Create a tool response message for LiteLLM.

    Args:
        tool_call_id: The ID of the tool call being responded to.
        tool_name: The name of the tool that was invoked.
        content: The content to include in the response.

    Returns:
        A litellm.Message object representing the tool response message.
    """
    import litellm

    return litellm.Message(
        tool_call_id=tool_call_id,
        role="tool",
        name=tool_name,
        content=content,
    )


def _get_default_judge_response_schema() -> type[pydantic.BaseModel]:
    """
    Get the default Pydantic schema for judge evaluations.

    Returns:
        A Pydantic BaseModel class defining the standard judge output format.
    """
    # Import here to avoid circular imports
    from mlflow.genai.judges.base import Judge

    output_fields = Judge.get_output_fields()

    field_definitions = {}
    for field in output_fields:
        field_definitions[field.name] = (str, pydantic.Field(description=field.description))

    return pydantic.create_model("JudgeEvaluation", **field_definitions)


def _prune_messages_exceeding_context_window_length(
    messages: list["litellm.Message"],
    model: str,
    max_tokens: int,
) -> list["litellm.Message"]:
    """
    Prune messages from history to stay under token limit.

    Args:
        messages: List of LiteLLM message objects.
        model: Model name for token counting.
        max_tokens: Maximum token limit.

    Returns:
        Pruned list of LiteLLM message objects under the token limit
    """
    import litellm

    initial_tokens = litellm.token_counter(model=model, messages=messages)
    if initial_tokens <= max_tokens:
        return messages

    pruned_messages = messages[:]
    # Remove tool call pairs until we're under limit
    while litellm.token_counter(model=model, messages=pruned_messages) > max_tokens:
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
        tool_call_ids = {
            tc.id if hasattr(tc, "id") else tc["id"] for tc in assistant_msg.tool_calls
        }
        pruned_messages = [
            msg
            for msg in pruned_messages
            if not (msg.role == "tool" and msg.tool_call_id in tool_call_ids)
        ]

    final_tokens = litellm.token_counter(model=model, messages=pruned_messages)
    _logger.info(f"Pruned message history from {initial_tokens} to {final_tokens} tokens")
    return pruned_messages


def _get_litellm_retry_policy(num_retries: int) -> "litellm.RetryPolicy":
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
