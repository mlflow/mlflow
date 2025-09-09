import json
import logging
import traceback
from dataclasses import dataclass

import requests

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.utils.enum_utils import StrEnum
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.telemetry.events import InvokeCustomJudgeModelEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.telemetry.utils import _is_in_databricks
from mlflow.utils.uri import is_databricks_uri

_logger = logging.getLogger(__name__)
# "endpoints" is a special case for Databricks model serving endpoints.
_NATIVE_PROVIDERS = ["openai", "anthropic", "bedrock", "mistral", "endpoints"]


def get_default_model() -> str:
    if is_databricks_uri(mlflow.get_tracking_uri()):
        return "databricks"
    else:
        return "openai:/gpt-4.1-mini"


def _sanitize_justification(justification: str) -> str:
    # Some judge prompts instruct the model to think step by step.
    return justification.replace("Let's think step by step. ", "")


@dataclass
class InvokeDatabricksModelOutput:
    response: str
    request_id: str | None
    num_prompt_tokens: int | None
    num_completion_tokens: int | None


def _invoke_databricks_model(
    *, model_name: str, prompt: str, num_retries: int
) -> InvokeDatabricksModelOutput:
    from mlflow.utils.databricks_utils import get_databricks_host_creds

    host_creds = get_databricks_host_creds()
    api_url = f"{host_creds.host}/serving-endpoints/{model_name}/invocations"

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
    res_json = res.json()
    content = res_json["choices"][0]["message"]["content"]

    # handle reasoning response
    if isinstance(content, list):
        for item in content:
            if item["type"] != "text":
                continue
            content = item["text"]
            break

    usage = res_json.get("usage", {})

    return InvokeDatabricksModelOutput(
        response=content,
        request_id=res.headers.get("x-request-id"),
        num_prompt_tokens=usage.get("prompt_tokens"),
        num_completion_tokens=usage.get("completion_tokens"),
    )


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

    from mlflow.models.model import get_job_id, get_job_run_id, get_workspace_id
    from mlflow.tracking.fluent import _get_experiment_id

    experiment_id = _get_experiment_id()
    workspace_id = get_workspace_id()
    job_id = get_job_id()
    job_run_id = get_job_run_id()

    record_judge_model_usage_success(
        request_id=request_id,
        experiment_id=experiment_id,
        job_id=job_id,
        job_run_id=job_run_id,
        workspace_id=workspace_id,
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

    from mlflow.models.model import get_job_id, get_job_run_id, get_workspace_id
    from mlflow.tracking.fluent import _get_experiment_id

    experiment_id = _get_experiment_id()
    workspace_id = get_workspace_id()
    job_id = get_job_id()
    job_run_id = get_job_run_id()

    record_judge_model_usage_failure(
        experiment_id=experiment_id,
        job_id=job_id,
        job_run_id=job_run_id,
        workspace_id=workspace_id,
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


def _invoke_judge_model(
    *,
    model_uri: str,
    prompt: str,
    assessment_name: str,
    num_retries: int = 10,
) -> InvokeJudgeModelHelperOutput:
    from mlflow.metrics.genai.model_utils import (
        _parse_model_uri,
        get_endpoint_type,
        score_model_on_payload,
    )

    provider, model_name = _parse_model_uri(model_uri)
    request_id = None
    num_prompt_tokens = None
    num_completion_tokens = None

    if provider == "databricks":
        output = _invoke_databricks_model(
            model_name=model_name,
            prompt=prompt,
            num_retries=num_retries,
        )
        response = output.response
        request_id = output.request_id
        num_prompt_tokens = output.num_prompt_tokens
        num_completion_tokens = output.num_completion_tokens
    elif _is_litellm_available():
        # prioritize litellm for better performance
        response = _invoke_litellm(provider, model_name, prompt, num_retries)
    elif provider in _NATIVE_PROVIDERS:
        response = score_model_on_payload(
            model_uri=model_uri,
            payload=prompt,
            endpoint_type=get_endpoint_type(model_uri) or "llm/v1/chat",
        )
    else:
        raise MlflowException(
            f"LiteLLM is required for using '{provider}' LLM. Please install it with "
            "`pip install litellm`.",
            error_code=INVALID_PARAMETER_VALUE,
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
            f"Failed to parse the response from the judge model. Response: {response}",
            error_code=INVALID_PARAMETER_VALUE,
        ) from e

    return InvokeJudgeModelHelperOutput(
        feedback=feedback,
        model_provider=provider,
        model_name=model_name,
        request_id=request_id,
        num_prompt_tokens=num_prompt_tokens,
        num_completion_tokens=num_completion_tokens,
    )


@record_usage_event(InvokeCustomJudgeModelEvent)
def invoke_judge_model(
    model_uri: str, prompt: str, assessment_name: str, num_retries: int = 10
) -> Feedback:
    """
    Invoke the judge model.

    First, try to invoke the judge model via litellm. If litellm is not installed,
    fallback to native parsing using the AI Gateway adapters.

    Args:
        model_uri: The model URI.
        prompt: The prompt to evaluate.
        assessment_name: The name of the assessment.
        num_retries: Number of retries on transient failures when using litellm.
    """
    error = None
    error_traceback = None
    try:
        output = _invoke_judge_model(
            model_uri=model_uri,
            prompt=prompt,
            assessment_name=assessment_name,
            num_retries=num_retries,
        )
    except Exception:
        error_traceback = traceback.format_exc()

    # Only record detailed telemetry when in Databricks
    if not _is_in_databricks():
        return output.feedback

    try:
        if error is not None:
            _record_judge_model_usage_failure_databricks_telemetry(
                model_provider=output.model_provider,
                endpoint_name=output.model_name,
                error_code="UNKNOWN",
                error_message=error_traceback,
            )
        else:
            _record_judge_model_usage_success_databricks_telemetry(
                request_id=output.request_id,
                model_provider=output.model_provider,
                endpoint_name=output.model_name,
                num_prompt_tokens=output.num_prompt_tokens,
                num_completion_tokens=output.num_completion_tokens,
            )
    except Exception as telemetry_error:
        _logger.debug("Failed to record judge model usage telemetry. Error: %s", telemetry_error)

    if error is not None:
        raise error

    return output.feedback


def _is_litellm_available() -> bool:
    try:
        import litellm  # noqa: F401

        return True
    except ImportError:
        return False


def _invoke_litellm(provider: str, model_name: str, prompt: str, num_retries: int = 7) -> str:
    """
    Invoke the judge model via litellm with retry support.

    Args:
        provider: The provider name (e.g., 'openai', 'anthropic').
        model_name: The model name.
        prompt: The prompt to send to the model.
        num_retries: Number of retries with exponential backoff on transient failures.

    Returns:
        The model's response content.

    Raises:
        MlflowException: If the request fails after all retries.
    """
    import litellm

    litellm_model_uri = f"{provider}/{model_name}"

    try:
        response = litellm.completion(
            model=litellm_model_uri,
            messages=[{"role": "user", "content": prompt}],
            retry_policy=_get_litellm_retry_policy(num_retries),
            retry_strategy="exponential_backoff_retry",
            # In LiteLLM version 1.55.3+, max_retries is stacked on top of retry_policy.
            # To avoid double-retry, we set max_retries=0
            max_retries=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise MlflowException("Failed to invoke the judge model via litellm.") from e


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
