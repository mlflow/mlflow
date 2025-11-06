"""Databricks adapter for judge model invocation with tool calling support."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pydantic
import requests

if TYPE_CHECKING:
    import litellm

    from mlflow.entities.trace import Trace
    from mlflow.types.llm import ChatMessage

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.environment_variables import MLFLOW_JUDGE_MAX_ITERATIONS
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.constants import (
    _DATABRICKS_AGENTIC_JUDGE_MODEL,
    _DATABRICKS_DEFAULT_JUDGE_MODEL,
)
from mlflow.genai.judges.tools import list_judge_tools
from mlflow.genai.judges.utils.parsing_utils import _sanitize_justification
from mlflow.genai.judges.utils.tool_calling_utils import _process_tool_calls
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INVALID_PARAMETER_VALUE,
    REQUEST_LIMIT_EXCEEDED,
)
from mlflow.types.llm import ToolDefinition
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)


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


def call_chat_completions(
    user_prompt: str,
    system_prompt: str,
    session_name: str | None = None,
    tools: list[ToolDefinition] | None = None,
    model: str | None = None,
) -> Any:
    """
    Invoke Databricks chat completions API with optional tool calling support.

    Args:
        user_prompt: The user prompt.
        system_prompt: The system prompt.
        session_name: Session name for tracking. Defaults to "mlflow-v{VERSION}".
        tools: Optional list of ToolDefinition objects for tool calling.
        model: Optional model to use.

    Returns:
        Chat completions result with output_json attribute.

    Raises:
        MlflowException: If databricks-agents is not installed.
    """
    _check_databricks_agents_installed()

    from databricks.rag_eval import context, env_vars

    if session_name is None:
        session_name = f"mlflow-v{VERSION}"

    env_vars.RAG_EVAL_EVAL_SESSION_CLIENT_NAME.set(session_name)

    @context.eval_context
    def _call_chat_completions(
        user_prompt: str, system_prompt: str, tools: list[ToolDefinition] | None, model: str | None
    ):
        managed_rag_client = context.get_context().build_managed_rag_client()

        kwargs = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
        }

        if model is not None:
            kwargs["model"] = model

        if tools is not None:
            kwargs["tools"] = tools

        try:
            return managed_rag_client.get_chat_completions_result(**kwargs)
        except Exception:
            _logger.debug("Failed to call chat completions", exc_info=True)
            raise

    return _call_chat_completions(user_prompt, system_prompt, tools, model)


def _parse_databricks_judge_response(
    llm_output: str | None,
    assessment_name: str,
    trace: "Trace | None" = None,
) -> Feedback:
    """
    Parse the response from Databricks judge into a Feedback object.

    Args:
        llm_output: Raw output from the LLM.
        assessment_name: Name of the assessment.
        trace: Optional trace object to associate with the feedback.

    Returns:
        Feedback object with parsed results or error.
    """
    source = AssessmentSource(
        source_type=AssessmentSourceType.LLM_JUDGE, source_id=_DATABRICKS_DEFAULT_JUDGE_MODEL
    )
    trace_id = trace.info.trace_id if trace else None

    if not llm_output:
        return Feedback(
            name=assessment_name,
            error="Empty response from Databricks judge",
            source=source,
            trace_id=trace_id,
        )

    try:
        response_data = json.loads(llm_output)
    except json.JSONDecodeError as e:
        _logger.debug(f"Invalid JSON response from Databricks judge: {e}", exc_info=True)
        return Feedback(
            name=assessment_name,
            error=f"Invalid JSON response from Databricks judge: {e}\n\nLLM output: {llm_output}",
            source=source,
            trace_id=trace_id,
        )

    if "result" not in response_data:
        return Feedback(
            name=assessment_name,
            error=f"Response missing 'result' field: {response_data}",
            source=source,
            trace_id=trace_id,
        )

    return Feedback(
        name=assessment_name,
        value=response_data["result"],
        rationale=response_data.get("rationale", ""),
        source=source,
        trace_id=trace_id,
    )


def _create_litellm_message_from_databricks_response(
    response_data: dict[str, Any],
) -> "litellm.Message":
    """
    Convert Databricks OpenAI-style response to litellm Message.

    Handles both string content and reasoning model outputs.

    Args:
        response_data: Parsed JSON response from Databricks.

    Returns:
        litellm.Message object.

    Raises:
        ValueError: If response format is invalid.
    """
    import litellm

    choices = response_data.get("choices", [])
    if not choices:
        raise ValueError("Invalid response format: missing 'choices' field")

    message_data = choices[0].get("message", {})

    # Create litellm Message with tool calls if present
    tool_calls_data = message_data.get("tool_calls")
    tool_calls = None
    if tool_calls_data:
        tool_calls = [
            litellm.ChatCompletionMessageToolCall(
                id=tc["id"],
                type=tc.get("type", "function"),
                function=litellm.Function(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in tool_calls_data
        ]

    content = message_data.get("content")
    if isinstance(content, list):
        content_parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                content_parts.append(block["text"])
        content = "\n".join(content_parts) if content_parts else None

    return litellm.Message(
        role=message_data.get("role", "assistant"),
        content=content,
        tool_calls=tool_calls,
    )


def _serialize_messages_to_databricks_prompts(
    messages: list["litellm.Message"],
) -> tuple[str, str | None]:
    """
    Serialize litellm Messages to user_prompt and system_prompt for Databricks.

    This is needed because call_chat_completions only accepts string prompts.

    TODO: Replace this with a messages array parameter for call_chat_completions.
    If call_chat_completions supported a messages parameter (similar to litellm),
    this entire function could be removed and we could pass messages directly:
        call_chat_completions(messages=messages, tools=tools, model=model)
    This would make the Databricks adapter identical to the litellm adapter.

    Args:
        messages: List of litellm Message objects.

    Returns:
        Tuple of (user_prompt, system_prompt).
    """
    system_prompt = None
    user_parts = []

    for msg in messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_parts.append(msg.content)
        elif msg.role == "assistant":
            if msg.tool_calls:
                # For assistant messages with tool calls, just indicate tool usage
                user_parts.append("Assistant: [Called tools]")
            elif msg.content:
                user_parts.append(f"Assistant: {msg.content}")
        elif msg.role == "tool":
            user_parts.append(f"Tool {msg.name}: {msg.content}")

    user_prompt = "\n\n".join(user_parts)
    return user_prompt, system_prompt


def _invoke_databricks_default_judge(
    prompt: str | list["ChatMessage"],
    assessment_name: str,
    trace: "Trace | None" = None,
) -> Feedback:
    """
    Invoke the Databricks default judge with agentic tool calling support.

    When a trace is provided, enables an agentic loop where the judge can iteratively
    call tools to analyze the trace data before producing a final assessment.

    Args:
        prompt: The formatted prompt with template variables filled in.
        assessment_name: The name of the assessment.
        trace: Optional trace object for tool-based analysis.

    Returns:
        Feedback object from the Databricks judge.

    Raises:
        MlflowException: If databricks-agents is not installed or max iterations exceeded.
    """
    import litellm

    try:
        # Convert initial prompt to litellm Messages (same pattern as litellm adapter)
        if isinstance(prompt, str):
            messages = [litellm.Message(role="user", content=prompt)]
        else:
            messages = [litellm.Message(role=msg.role, content=msg.content) for msg in prompt]

        # Enable tool calling if trace is provided
        tools = None
        if trace is not None:
            tools = [tool.get_definition() for tool in list_judge_tools()]

        # Agentic loop: iteratively call LLM and execute tools until final answer
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
                # Serialize messages to Databricks format (only difference from litellm)
                user_prompt, system_prompt = _serialize_messages_to_databricks_prompts(messages)

                llm_result = call_chat_completions(
                    user_prompt, system_prompt, tools=tools, model=_DATABRICKS_AGENTIC_JUDGE_MODEL
                )

                # Parse response
                output_json = llm_result.output_json
                if not output_json:
                    return Feedback(
                        name=assessment_name,
                        error="Empty response from Databricks judge",
                        source=AssessmentSource(
                            source_type=AssessmentSourceType.LLM_JUDGE,
                            source_id=_DATABRICKS_DEFAULT_JUDGE_MODEL,
                        ),
                        trace_id=trace.info.trace_id if trace else None,
                    )

                parsed_json = (
                    json.loads(output_json) if isinstance(output_json, str) else output_json
                )

                # Convert response to litellm Message
                try:
                    message = _create_litellm_message_from_databricks_response(parsed_json)
                except ValueError as e:
                    return Feedback(
                        name=assessment_name,
                        error=f"Invalid response format from Databricks judge: {e}",
                        source=AssessmentSource(
                            source_type=AssessmentSourceType.LLM_JUDGE,
                            source_id=_DATABRICKS_DEFAULT_JUDGE_MODEL,
                        ),
                        trace_id=trace.info.trace_id if trace else None,
                    )

                # No tool calls means final answer - parse and return
                if not message.tool_calls:
                    return _parse_databricks_judge_response(message.content, assessment_name, trace)

                # Append assistant message and process tool calls (same pattern as litellm)
                messages.append(message)
                tool_response_messages = _process_tool_calls(
                    tool_calls=message.tool_calls,
                    trace=trace,
                )
                messages.extend(tool_response_messages)

            except Exception as e:
                _logger.debug(
                    f"Failed in agentic loop iteration {iteration_count}: {e}", exc_info=True
                )
                raise

    except Exception as e:
        _logger.debug(f"Failed to invoke Databricks judge: {e}", exc_info=True)
        return Feedback(
            name=assessment_name,
            error=f"Failed to invoke Databricks judge: {e}",
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE,
                source_id=_DATABRICKS_DEFAULT_JUDGE_MODEL,
            ),
            trace_id=trace.info.trace_id if trace else None,
        )


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
        text_content = next(
            (
                item.get("text")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ),
            None,
        )
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


def _invoke_databricks_serving_endpoint(
    *,
    model_name: str,
    prompt: str,
    num_retries: int,
    response_format: type[pydantic.BaseModel] | None = None,
) -> InvokeDatabricksModelOutput:
    from mlflow.utils.databricks_utils import get_databricks_host_creds

    # B-Step62: Why not use mlflow deployment client?
    host_creds = get_databricks_host_creds()
    api_url = f"{host_creds.host}/serving-endpoints/{model_name}/invocations"

    # Implement retry logic with exponential backoff
    last_exception = None
    for attempt in range(num_retries + 1):
        try:
            # Build request payload
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            }

            # Add response_schema if provided
            if response_format is not None:
                payload["response_schema"] = response_format.model_json_schema()

            res = requests.post(
                url=api_url,
                headers={"Authorization": f"Bearer {host_creds.token}"},
                json=payload,
            )
        except (requests.RequestException, requests.ConnectionError) as e:
            last_exception = e
            if attempt < num_retries:
                _logger.debug(
                    f"Request attempt {attempt + 1} failed with error: {e}", exc_info=True
                )
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
                _logger.debug(f"Attempt {attempt + 1} failed: {error_msg}", exc_info=True)
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


def _invoke_databricks_serving_endpoint_judge(
    *,
    model_name: str,
    prompt: str,
    assessment_name: str,
    num_retries: int = 10,
    response_format: type[pydantic.BaseModel] | None = None,
) -> InvokeJudgeModelHelperOutput:
    output = _invoke_databricks_serving_endpoint(
        model_name=model_name,
        prompt=prompt,
        num_retries=num_retries,
        response_format=response_format,
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
