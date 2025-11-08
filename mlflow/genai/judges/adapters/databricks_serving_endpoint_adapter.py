"""Databricks Serving Endpoint adapter for direct REST API invocations."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import pydantic
import requests

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils.parsing_utils import _sanitize_justification
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

_logger = logging.getLogger(__name__)


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
