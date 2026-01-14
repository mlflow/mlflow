from __future__ import annotations

import json
import logging
import time
import traceback
import warnings
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
from mlflow.genai.judges.adapters.base_adapter import (
    AdapterInvocationInput,
    AdapterInvocationOutput,
    BaseJudgeAdapter,
)
from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.judges.tools import list_judge_tools
from mlflow.genai.judges.utils.parsing_utils import _sanitize_justification
from mlflow.genai.judges.utils.tool_calling_utils import (
    _process_tool_calls,
    _raise_iteration_limit_exceeded,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.telemetry.utils import _log_error

_logger = logging.getLogger(__name__)

_RESPONSE_FORMAT_ERROR_MESSAGES = (
    "response_format",
    "response_schema",
    "response format",
    "responseformatobject",
    'unknown field "properties"',
    'unknown field \\"properties\\"',
)


@dataclass
class InvokeDatabricksModelOutput:
    response: str
    request_id: str | None
    num_prompt_tokens: int | None
    num_completion_tokens: int | None
    tool_calls: list[dict[str, Any]] | None = None


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

    message = first_choice.get("message", {})
    content = message.get("content")
    tool_calls = message.get("tool_calls")

    if content is None and not tool_calls:
        raise MlflowException(
            "Invalid response from Databricks model: "
            "missing both 'content' and 'tool_calls' fields",
            error_code=INVALID_PARAMETER_VALUE,
        )

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
        tool_calls=tool_calls,
    )


def _invoke_databricks_serving_endpoint(
    *,
    model_name: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    num_retries: int,
    response_format: type[pydantic.BaseModel] | None = None,
    inference_params: dict[str, Any] | None = None,
) -> InvokeDatabricksModelOutput:
    print("invoke_databricks_serving_endpoint called")  # noqa: T201

    from mlflow.utils.databricks_utils import get_databricks_host_creds

    # B-Step62: Why not use mlflow deployment client?
    host_creds = get_databricks_host_creds()
    api_url = f"{host_creds.host}/serving-endpoints/{model_name}/invocations"

    # If tools are provided, disable response_format preemptively since many models
    # don't support using both together (e.g., Claude)
    include_response_format = tools is None

    # Implement retry logic with exponential backoff
    last_exception = None
    for attempt in range(num_retries + 1):
        try:
            payload = {"messages": messages}

            if tools:
                payload["tools"] = tools

            if response_format is not None and include_response_format:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": response_format.model_json_schema(),
                    },
                }

            # Add inference parameters if provided (e.g., temperature, top_p, max_tokens)
            if inference_params:
                payload.update(inference_params)

            # print("request payload", json.dumps(payload, indent=4))

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
            # Check if this is an error related to response_format/response_schema parameter.
            # If so, drop the parameter and retry. This mimics LiteLLM's drop_params behavior.
            # Some endpoints return errors about 'response_format', others about 'response_schema'.
            error_text_lower = res.text.lower()
            is_response_format_error = any(
                s in error_text_lower for s in _RESPONSE_FORMAT_ERROR_MESSAGES
            )
            if (
                res.status_code == 400
                and include_response_format
                and response_format is not None
                and is_response_format_error
            ):
                _logger.debug(
                    f"Model '{model_name}' may not support structured outputs (response_format) "
                    f"or may not support combining response_format with tool calling. "
                    f"Retrying without structured output enforcement. The response may not follow "
                    "the expected format."
                )
                include_response_format = False
                continue

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

        print("response", res_json)  # noqa: T201

        # Parse and validate the response using helper function
        return _parse_databricks_model_response(res_json, res.headers)

    # This should not be reached, but just in case
    if last_exception:
        raise MlflowException(
            f"Failed to invoke Databricks model: {last_exception}",
            error_code=INVALID_PARAMETER_VALUE,
        ) from last_exception


def _is_gemini_3_model(model_name: str) -> bool:
    model_lower = model_name.lower()
    return "gemini-3" in model_lower


def _convert_litellm_messages_to_serving_endpoint_api_format(
    messages: list["litellm.Message"],
    model_name: str,
) -> list[dict[str, Any]]:
    """Convert litellm messages to serving endpoint API format.

    Uses OpenAI format with role="tool" for tool results. For Gemini 3 models,
    preserves the thoughtSignature field which is required for tool calling.

    Args:
        messages: List of litellm Message objects
        model_name: The model name to determine model-specific handling
    """
    api_messages = []
    is_gemini_3 = _is_gemini_3_model(model_name)

    for msg in messages:
        if msg.role == "tool":
            api_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                }
            )
        elif msg.tool_calls:
            tool_calls_data = []
            for tc in msg.tool_calls:
                tool_call = {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": (
                            tc.function.arguments
                            if isinstance(tc.function.arguments, str)
                            else json.dumps(tc.function.arguments)
                        ),
                    },
                }
                if is_gemini_3 and hasattr(tc, "thoughtSignature"):
                    tool_call["thoughtSignature"] = tc.thoughtSignature
                tool_calls_data.append(tool_call)

            message_dict = {
                "role": msg.role,
                "tool_calls": tool_calls_data,
            }

            if msg.content is not None:
                message_dict["content"] = msg.content

            api_messages.append(message_dict)
        else:
            api_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )
    return api_messages


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
    prompt: str | list["ChatMessage"],
    assessment_name: str,
    trace: "Trace | None" = None,
    num_retries: int = 10,
    response_format: type[pydantic.BaseModel] | None = None,
    inference_params: dict[str, Any] | None = None,
) -> InvokeJudgeModelHelperOutput:
    print("invoke_databricks_serving_endpoint_judge called")  # noqa: T201
    import litellm

    if isinstance(prompt, str):
        messages = [litellm.Message(role="user", content=prompt)]
    else:
        messages = [litellm.Message(role=msg.role, content=msg.content) for msg in prompt]

    tools = None
    if trace is not None:
        judge_tools = list_judge_tools()
        tools = []

        is_openai = "gpt-" in model_name.lower()

        for tool in judge_tools:
            tool_dict = tool.get_definition().to_dict()
            if not is_openai and "function" in tool_dict and "strict" in tool_dict["function"]:
                del tool_dict["function"]["strict"]
            tools.append(tool_dict)

    print(  # noqa: T201
        "tools available to the judge:",
        [item.get("function", {}).get("name", item.get("name", "unknown")) for item in tools]
        if tools
        else "[]",
    )

    max_iterations = MLFLOW_JUDGE_MAX_ITERATIONS.get()
    iteration_count = 0
    last_request_id = None
    total_prompt_tokens = 0
    total_completion_tokens = 0

    while True:
        iteration_count += 1
        print("agent loop iteration count", iteration_count)  # noqa: T201
        if iteration_count > max_iterations:
            _raise_iteration_limit_exceeded(max_iterations)

        api_messages = _convert_litellm_messages_to_serving_endpoint_api_format(
            messages, model_name
        )
        print(f"\n=== Iteration {iteration_count}: Sending {len(api_messages)} messages to API ===")  # noqa: T201
        for i, msg in enumerate(api_messages):
            print(f"  Message {i}: {json.dumps(msg, indent=2)}")  # noqa: T201

        output = _invoke_databricks_serving_endpoint(
            model_name=model_name,
            messages=api_messages,
            tools=tools,
            num_retries=num_retries,
            response_format=response_format,
            inference_params=inference_params,
        )

        last_request_id = output.request_id
        if output.num_prompt_tokens:
            total_prompt_tokens += output.num_prompt_tokens
        if output.num_completion_tokens:
            total_completion_tokens += output.num_completion_tokens

        if not output.tool_calls:
            break

        litellm_tool_calls = [
            litellm.ChatCompletionMessageToolCall(
                id=tc["id"],
                type=tc.get("type", "function"),
                function=litellm.Function(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in output.tool_calls
        ]

        if _is_gemini_3_model(model_name):
            for litellm_tc, raw_tc in zip(litellm_tool_calls, output.tool_calls):
                if "thoughtSignature" in raw_tc:
                    litellm_tc.thoughtSignature = raw_tc["thoughtSignature"]

        assistant_message = litellm.Message(
            role="assistant",
            content=output.response,
            tool_calls=litellm_tool_calls,
        )

        messages.append(assistant_message)
        tool_response_messages = _process_tool_calls(tool_calls=litellm_tool_calls, trace=trace)
        print(  # noqa: T201
            f"Processing {len(litellm_tool_calls)} tool calls, "
            f"got {len(tool_response_messages)} tool responses"
        )
        for i, msg in enumerate(tool_response_messages):
            print(  # noqa: T201
                f"  Tool response {i}: role={msg.role}, tool_call_id={msg.tool_call_id}, "
                f"name={getattr(msg, 'name', 'N/A')}, "
                f"content_len={len(msg.content) if msg.content else 0}"
            )
        messages.extend(tool_response_messages)

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
        request_id=last_request_id,
        num_prompt_tokens=total_prompt_tokens if total_prompt_tokens > 0 else None,
        num_completion_tokens=total_completion_tokens if total_completion_tokens > 0 else None,
    )


class DatabricksServingEndpointAdapter(BaseJudgeAdapter):
    """Adapter for Databricks serving endpoints using direct REST API invocations."""

    @classmethod
    def is_applicable(
        cls,
        model_uri: str,
        prompt: str | list["ChatMessage"],
    ) -> bool:
        from mlflow.metrics.genai.model_utils import _parse_model_uri

        # Don't handle the default judge (that's handled by DatabricksManagedJudgeAdapter)
        if model_uri == _DATABRICKS_DEFAULT_JUDGE_MODEL:
            return False

        model_provider, _ = _parse_model_uri(model_uri)
        return model_provider in {"databricks", "endpoints"}

    def invoke(self, input_params: AdapterInvocationInput) -> AdapterInvocationOutput:
        # Show deprecation warning for legacy 'endpoints' provider
        model_provider = input_params.model_provider
        if model_provider == "endpoints":
            warnings.warn(
                "The legacy provider 'endpoints' is deprecated and will be removed in a future "
                "release. Please update your code to use the 'databricks' provider instead.",
                FutureWarning,
                stacklevel=4,
            )

        model_name = input_params.model_name

        try:
            output = _invoke_databricks_serving_endpoint_judge(
                model_name=model_name,
                prompt=input_params.prompt,
                assessment_name=input_params.assessment_name,
                trace=input_params.trace,
                num_retries=input_params.num_retries,
                response_format=input_params.response_format,
                inference_params=input_params.inference_params,
            )

            # Set trace_id if trace was provided
            feedback = output.feedback
            if input_params.trace is not None:
                feedback.trace_id = input_params.trace.info.trace_id

            try:
                provider = "databricks" if model_provider == "endpoints" else model_provider
                _record_judge_model_usage_success_databricks_telemetry(
                    request_id=output.request_id,
                    model_provider=provider,
                    endpoint_name=model_name,
                    num_prompt_tokens=output.num_prompt_tokens,
                    num_completion_tokens=output.num_completion_tokens,
                )
            except Exception:
                _log_error("Failed to record judge model usage success telemetry")

            return AdapterInvocationOutput(
                feedback=feedback,
                request_id=output.request_id,
                num_prompt_tokens=output.num_prompt_tokens,
                num_completion_tokens=output.num_completion_tokens,
            )

        except Exception:
            try:
                provider = "databricks" if model_provider == "endpoints" else model_provider
                _record_judge_model_usage_failure_databricks_telemetry(
                    model_provider=provider,
                    endpoint_name=model_name,
                    error_code="UNKNOWN",
                    error_message=traceback.format_exc(),
                )
            except Exception:
                _log_error("Failed to record judge model usage failure telemetry")
            raise
