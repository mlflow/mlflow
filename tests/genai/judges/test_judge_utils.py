import json
from unittest import mock

import pytest
from litellm.types.utils import ModelResponse

from mlflow.entities.assessment import AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating, format_prompt, invoke_judge_model


@pytest.mark.parametrize("num_retries", [None, 3])
def test_invoke_judge_model_successful_with_litellm(num_retries):
    mock_content = json.dumps({"result": "yes", "rationale": "The response meets all criteria."})
    mock_response = ModelResponse(choices=[{"message": {"content": mock_content}}])

    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        kwargs = {
            "model_uri": "openai:/gpt-4",
            "prompt": "Evaluate this response",
            "assessment_name": "quality_check",
        }
        if num_retries is not None:
            kwargs["num_retries"] = num_retries

        feedback = invoke_judge_model(**kwargs)

    from litellm import RetryPolicy

    expected_retries = 10 if num_retries is None else num_retries
    expected_retry_policy = RetryPolicy(
        TimeoutErrorRetries=expected_retries,
        RateLimitErrorRetries=expected_retries,
        InternalServerErrorRetries=expected_retries,
        ContentPolicyViolationErrorRetries=expected_retries,
        BadRequestErrorRetries=0,
        AuthenticationErrorRetries=0,
    )

    mock_litellm.assert_called_once_with(
        model="openai/gpt-4",
        messages=[{"role": "user", "content": "Evaluate this response"}],
        retry_policy=expected_retry_policy,
        retry_strategy="exponential_backoff_retry",
        max_retries=0,
    )

    assert feedback.name == "quality_check"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The response meets all criteria."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4"


def test_invoke_judge_model_successful_with_native_provider():
    mock_response = json.dumps({"result": "yes", "rationale": "The response meets all criteria."})

    with (
        mock.patch("mlflow.genai.judges.utils._is_litellm_available", return_value=False),
        mock.patch(
            "mlflow.metrics.genai.model_utils.score_model_on_payload", return_value=mock_response
        ) as mock_score_model_on_payload,
    ):
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check",
        )

    mock_score_model_on_payload.assert_called_once_with(
        model_uri="openai:/gpt-4",
        payload="Evaluate this response",
        endpoint_type="llm/v1/chat",
    )

    assert feedback.name == "quality_check"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The response meets all criteria."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4"


def test_invoke_judge_model_with_unsupported_provider():
    with pytest.raises(MlflowException, match=r"LiteLLM is required for using 'unsupported' LLM"):
        with mock.patch("mlflow.genai.judges.utils._is_litellm_available", return_value=False):
            invoke_judge_model(
                model_uri="unsupported:/model", prompt="Test prompt", assessment_name="test"
            )


def test_invoke_judge_model_invalid_json_response():
    mock_content = "This is not valid JSON"
    mock_response = ModelResponse(choices=[{"message": {"content": mock_content}}])

    with mock.patch("litellm.completion", return_value=mock_response):
        with pytest.raises(MlflowException, match=r"Failed to parse"):
            invoke_judge_model(
                model_uri="openai:/gpt-4", prompt="Test prompt", assessment_name="test"
            )


@pytest.mark.parametrize(
    ("prompt_template", "values", "expected"),
    [
        # Test with Unicode escape-like sequences
        (
            "User input: {{ user_text }}",
            {"user_text": r"Path is C:\users\john"},
            r"User input: Path is C:\users\john",
        ),
        # Test with newlines and tabs
        (
            "Data: {{ data }}",
            {"data": "Line1\\nLine2\\tTabbed"},
            "Data: Line1\\nLine2\\tTabbed",
        ),
        # Test with multiple variables
        (
            "Path: {{ path }}, Command: {{ cmd }}",
            {"path": r"C:\temp", "cmd": r"echo \u0041"},
            r"Path: C:\temp, Command: echo \u0041",
        ),
    ],
)
def test_format_prompt_with_backslashes(prompt_template, values, expected):
    """Test that format_prompt correctly handles values containing backslashes."""
    result = format_prompt(prompt_template, **values)
    assert result == expected
