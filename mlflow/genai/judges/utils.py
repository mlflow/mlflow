import json
import re

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.utils.enum_utils import StrEnum
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.uri import is_databricks_uri

# "endpoints" is a special case for Databricks model serving endpoints.
_NATIVE_PROVIDERS = ["openai", "anthropic", "bedrock", "mistral", "endpoints"]


def get_default_model() -> str:
    if is_databricks_uri(mlflow.get_tracking_uri()):
        return "databricks"
    else:
        return "openai:/gpt-4.1-mini"


def format_prompt(prompt: str, **values) -> str:
    """Format double-curly variables in the prompt template."""
    for key, value in values.items():
        prompt = re.sub(r"\{\{\s*" + key + r"\s*\}\}", str(value), prompt)
    return prompt


def _sanitize_justification(justification: str) -> str:
    # Some judge prompts instruct the model to think step by step.
    return justification.replace("Let's think step by step. ", "")


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
    from mlflow.metrics.genai.model_utils import (
        _parse_model_uri,
        get_endpoint_type,
        score_model_on_payload,
    )

    provider, model_name = _parse_model_uri(model_uri)

    # Try litellm first for better performance.
    if _is_litellm_available():
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

    return feedback


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
