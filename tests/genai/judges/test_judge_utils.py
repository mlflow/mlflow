import json
from typing import Any
from unittest import mock

import pytest
import requests
from litellm.types.utils import ModelResponse

from mlflow.entities.assessment import AssessmentSourceType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import (
    CategoricalRating,
    InvokeDatabricksModelOutput,
    _invoke_databricks_model,
    _parse_databricks_model_response,
    invoke_judge_model,
)
from mlflow.genai.prompts.utils import format_prompt


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
def test_format_prompt_with_backslashes(
    prompt_template: str, values: dict[str, str], expected: str
) -> None:
    """Test that format_prompt correctly handles values containing backslashes."""
    result = format_prompt(prompt_template, **values)
    assert result == expected


def test_parse_databricks_model_response_valid_response() -> None:
    """Test parsing a valid response with all fields."""
    res_json = {
        "choices": [{"message": {"content": "This is the response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    headers = {"x-request-id": "test-request-id"}

    result = _parse_databricks_model_response(res_json, headers)

    assert isinstance(result, InvokeDatabricksModelOutput)
    assert result.response == "This is the response"
    assert result.request_id == "test-request-id"
    assert result.num_prompt_tokens == 10
    assert result.num_completion_tokens == 5


def test_parse_databricks_model_response_reasoning_response() -> None:
    """Test parsing a response with list content (reasoning mode)."""
    res_json = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "Reasoning response"},
                        {"type": "other", "data": "ignored"},
                    ]
                }
            }
        ]
    }
    headers: dict[str, str] = {}

    result = _parse_databricks_model_response(res_json, headers)

    assert result.response == "Reasoning response"
    assert result.request_id is None
    assert result.num_prompt_tokens is None
    assert result.num_completion_tokens is None


@pytest.mark.parametrize(
    "invalid_content", [[{"type": "other", "data": "no text content"}], [{}], []]
)
def test_parse_databricks_model_response_invalid_reasoning_response(
    invalid_content: list[dict[str, str]],
) -> None:
    """Test parsing invalid reasoning responses."""
    res_json = {"choices": [{"message": {"content": invalid_content}}]}
    headers: dict[str, str] = {}

    with pytest.raises(MlflowException, match="no text content found"):
        _parse_databricks_model_response(res_json, headers)


def test_parse_databricks_model_response_missing_choices() -> None:
    """Test error when choices field is missing."""
    res_json: dict[str, Any] = {}
    headers: dict[str, str] = {}

    with pytest.raises(MlflowException, match="missing 'choices' field"):
        _parse_databricks_model_response(res_json, headers)


def test_parse_databricks_model_response_empty_choices() -> None:
    """Test error when choices array is empty."""
    res_json = {"choices": []}
    headers: dict[str, str] = {}

    with pytest.raises(MlflowException, match="missing 'choices' field"):
        _parse_databricks_model_response(res_json, headers)


def test_parse_databricks_model_response_missing_message() -> None:
    """Test error when message field is missing."""
    res_json = {"choices": [{}]}
    headers: dict[str, str] = {}

    with pytest.raises(MlflowException, match="missing 'message' field"):
        _parse_databricks_model_response(res_json, headers)


def test_parse_databricks_model_response_missing_content() -> None:
    """Test error when content field is missing."""
    res_json = {"choices": [{"message": {}}]}
    headers: dict[str, str] = {}

    with pytest.raises(MlflowException, match="missing 'content' field"):
        _parse_databricks_model_response(res_json, headers)


def test_parse_databricks_model_response_none_content() -> None:
    """Test error when content field is None."""
    res_json = {"choices": [{"message": {"content": None}}]}
    headers: dict[str, str] = {}

    with pytest.raises(MlflowException, match="missing 'content' field"):
        _parse_databricks_model_response(res_json, headers)


@mock.patch("mlflow.utils.databricks_utils.get_databricks_host_creds")
@mock.patch("mlflow.genai.judges.utils.requests.post")
def test_invoke_databricks_model_successful_invocation(
    mock_post: mock.Mock, mock_get_creds: mock.Mock
) -> None:
    """Test successful Databricks model invocation."""
    mock_creds = mock.Mock()
    mock_creds.host = "https://test.databricks.com"
    mock_creds.token = "test-token"
    mock_get_creds.return_value = mock_creds

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    mock_response.headers = {"x-request-id": "test-id"}
    mock_post.return_value = mock_response

    result = _invoke_databricks_model(model_name="test-model", prompt="test prompt", num_retries=3)

    mock_post.assert_called_once_with(
        url="https://test.databricks.com/serving-endpoints/test-model/invocations",
        headers={"Authorization": "Bearer test-token"},
        json={"messages": [{"role": "user", "content": "test prompt"}]},
    )

    assert result.response == "Test response"
    assert result.request_id == "test-id"
    assert result.num_prompt_tokens == 10
    assert result.num_completion_tokens == 5


@mock.patch("mlflow.utils.databricks_utils.get_databricks_host_creds")
@mock.patch("mlflow.genai.judges.utils.requests.post")
def test_invoke_databricks_model_bad_request_error_no_retry(
    mock_post: mock.Mock, mock_get_creds: mock.Mock
) -> None:
    """Test that 400/403 errors are not retried."""
    mock_creds = mock.Mock()
    mock_creds.host = "https://test.databricks.com"
    mock_creds.token = "test-token"
    mock_get_creds.return_value = mock_creds

    mock_response = mock.Mock()
    mock_response.status_code = 400
    mock_response.text = "Bad request"
    mock_post.return_value = mock_response

    with pytest.raises(MlflowException, match="failed with status 400"):
        _invoke_databricks_model(model_name="test-model", prompt="test prompt", num_retries=3)

    mock_post.assert_called_once()


@mock.patch("mlflow.utils.databricks_utils.get_databricks_host_creds")
@mock.patch("mlflow.genai.judges.utils.requests.post")
@mock.patch("mlflow.genai.judges.utils.time.sleep")
def test_invoke_databricks_model_retry_logic_with_transient_errors(
    mock_sleep: mock.Mock, mock_post: mock.Mock, mock_get_creds: mock.Mock
) -> None:
    """Test retry logic with transient errors."""
    mock_creds = mock.Mock()
    mock_creds.host = "https://test.databricks.com"
    mock_creds.token = "test-token"
    mock_get_creds.return_value = mock_creds

    # First call fails with 500, second succeeds
    error_response = mock.Mock()
    error_response.status_code = 500
    error_response.text = "Internal server error"

    success_response = mock.Mock()
    success_response.status_code = 200
    success_response.json.return_value = {"choices": [{"message": {"content": "Success"}}]}
    success_response.headers = {}

    mock_post.side_effect = [error_response, success_response]

    result = _invoke_databricks_model(model_name="test-model", prompt="test prompt", num_retries=3)

    assert mock_post.call_count == 2
    mock_sleep.assert_called_once_with(1)  # 2^0 = 1
    assert result.response == "Success"


@mock.patch("mlflow.utils.databricks_utils.get_databricks_host_creds")
@mock.patch("mlflow.genai.judges.utils.requests.post")
def test_invoke_databricks_model_json_decode_error(
    mock_post: mock.Mock, mock_get_creds: mock.Mock
) -> None:
    """Test error handling for invalid JSON response."""
    mock_creds = mock.Mock()
    mock_creds.host = "https://test.databricks.com"
    mock_creds.token = "test-token"
    mock_get_creds.return_value = mock_creds

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
    mock_post.return_value = mock_response

    with pytest.raises(MlflowException, match="Failed to parse JSON response"):
        _invoke_databricks_model(model_name="test-model", prompt="test prompt", num_retries=0)


@mock.patch("mlflow.utils.databricks_utils.get_databricks_host_creds")
@mock.patch("mlflow.genai.judges.utils.requests.post")
@mock.patch("mlflow.genai.judges.utils.time.sleep")
def test_invoke_databricks_model_connection_error_with_retries(
    mock_sleep: mock.Mock, mock_post: mock.Mock, mock_get_creds: mock.Mock
) -> None:
    """Test connection error handling with retries."""
    mock_creds = mock.Mock()
    mock_get_creds.return_value = mock_creds

    mock_post.side_effect = requests.ConnectionError("Connection failed")

    with pytest.raises(MlflowException, match="Failed to invoke Databricks model after 3 attempts"):
        _invoke_databricks_model(model_name="test-model", prompt="test prompt", num_retries=2)

    assert mock_post.call_count == 3  # Initial + 2 retries
    assert mock_sleep.call_count == 2


@mock.patch("mlflow.tracking.fluent._get_experiment_id")
@mock.patch("mlflow.utils.databricks_utils.get_workspace_id")
@mock.patch("mlflow.utils.databricks_utils.get_job_id")
@mock.patch("mlflow.utils.databricks_utils.get_job_run_id")
@mock.patch("sys.modules", {"databricks.agents.telemetry": mock.MagicMock()})
def test_record_success_telemetry_with_databricks_agents(
    mock_job_run_id: mock.Mock,
    mock_job_id: mock.Mock,
    mock_workspace_id: mock.Mock,
    mock_experiment_id: mock.Mock,
) -> None:
    """Test successful telemetry recording when databricks-agents is available."""
    from mlflow.genai.judges.utils import _record_judge_model_usage_success_databricks_telemetry

    # Mock the context values
    mock_experiment_id.return_value = "exp-123"
    mock_workspace_id.return_value = "ws-456"
    mock_job_id.return_value = "job-789"
    mock_job_run_id.return_value = "run-101"

    # Mock the telemetry function separately
    mock_telemetry_module = mock.MagicMock()
    mock_record = mock.MagicMock()
    mock_telemetry_module.record_judge_model_usage_success = mock_record

    with mock.patch.dict("sys.modules", {"databricks.agents.telemetry": mock_telemetry_module}):
        _record_judge_model_usage_success_databricks_telemetry(
            request_id="req-123",
            model_provider="databricks",
            endpoint_name="test-endpoint",
            num_prompt_tokens=10,
            num_completion_tokens=5,
        )

        mock_record.assert_called_once_with(
            request_id="req-123",
            experiment_id="exp-123",
            job_id="job-789",
            job_run_id="run-101",
            workspace_id="ws-456",
            model_provider="databricks",
            endpoint_name="test-endpoint",
            num_prompt_tokens=10,
            num_completion_tokens=5,
        )


def test_record_success_telemetry_without_databricks_agents() -> None:
    """Test telemetry recording gracefully handles missing databricks-agents."""
    from mlflow.genai.judges.utils import _record_judge_model_usage_success_databricks_telemetry

    with mock.patch.dict("sys.modules", {"databricks.agents.telemetry": None}):
        # Should not raise exception
        _record_judge_model_usage_success_databricks_telemetry(
            request_id="req-123",
            model_provider="databricks",
            endpoint_name="test-endpoint",
            num_prompt_tokens=10,
            num_completion_tokens=5,
        )


@mock.patch("mlflow.tracking.fluent._get_experiment_id")
@mock.patch("mlflow.utils.databricks_utils.get_workspace_id")
@mock.patch("mlflow.utils.databricks_utils.get_job_id")
@mock.patch("mlflow.utils.databricks_utils.get_job_run_id")
def test_record_failure_telemetry_with_databricks_agents(
    mock_job_run_id: mock.Mock,
    mock_job_id: mock.Mock,
    mock_workspace_id: mock.Mock,
    mock_experiment_id: mock.Mock,
) -> None:
    """Test failure telemetry recording when databricks-agents is available."""
    from mlflow.genai.judges.utils import _record_judge_model_usage_failure_databricks_telemetry

    mock_experiment_id.return_value = "exp-123"
    mock_workspace_id.return_value = "ws-456"
    mock_job_id.return_value = "job-789"
    mock_job_run_id.return_value = "run-101"

    # Mock the telemetry function separately
    mock_telemetry_module = mock.MagicMock()
    mock_record = mock.MagicMock()
    mock_telemetry_module.record_judge_model_usage_failure = mock_record

    with mock.patch.dict("sys.modules", {"databricks.agents.telemetry": mock_telemetry_module}):
        _record_judge_model_usage_failure_databricks_telemetry(
            model_provider="databricks",
            endpoint_name="test-endpoint",
            error_code="TIMEOUT",
            error_message="Request timed out",
        )

        mock_record.assert_called_once_with(
            experiment_id="exp-123",
            job_id="job-789",
            job_run_id="run-101",
            workspace_id="ws-456",
            model_provider="databricks",
            endpoint_name="test-endpoint",
            error_code="TIMEOUT",
            error_message="Request timed out",
        )


@mock.patch("mlflow.genai.judges.utils._is_in_databricks")
@mock.patch("mlflow.genai.judges.utils._invoke_databricks_model")
def test_invoke_judge_model_databricks_success_not_in_databricks(
    mock_invoke_db: mock.Mock, mock_in_db: mock.Mock
) -> None:
    """Test successful invocation with Databricks model when not in Databricks environment."""
    mock_in_db.return_value = False
    mock_invoke_db.return_value = InvokeDatabricksModelOutput(
        response='{"result": "yes", "rationale": "Good response"}',
        request_id="req-123",
        num_prompt_tokens=10,
        num_completion_tokens=5,
    )

    feedback = invoke_judge_model(
        model_uri="databricks:/test-model",
        prompt="Test prompt",
        assessment_name="test_assessment",
    )

    mock_invoke_db.assert_called_once_with(
        model_name="test-model", prompt="Test prompt", num_retries=10
    )

    assert feedback.name == "test_assessment"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "Good response"


@mock.patch("mlflow.genai.judges.utils._is_in_databricks")
@mock.patch("mlflow.genai.judges.utils._invoke_databricks_model")
@mock.patch("mlflow.genai.judges.utils._record_judge_model_usage_success_databricks_telemetry")
def test_invoke_judge_model_databricks_success_in_databricks(
    mock_success_telemetry: mock.Mock, mock_invoke_db: mock.Mock, mock_in_db: mock.Mock
) -> None:
    """Test successful invocation with telemetry recording when in Databricks."""
    mock_in_db.return_value = True
    mock_invoke_db.return_value = InvokeDatabricksModelOutput(
        response='{"result": "no", "rationale": "Bad response"}',
        request_id="req-456",
        num_prompt_tokens=15,
        num_completion_tokens=8,
    )

    feedback = invoke_judge_model(
        model_uri="databricks:/test-model",
        prompt="Test prompt",
        assessment_name="test_assessment",
    )

    # Verify telemetry was called
    mock_success_telemetry.assert_called_once_with(
        request_id="req-456",
        model_provider="databricks",
        endpoint_name="test-model",
        num_prompt_tokens=15,
        num_completion_tokens=8,
    )

    assert feedback.value == CategoricalRating.NO
    assert feedback.rationale == "Bad response"


@mock.patch("mlflow.genai.judges.utils._is_in_databricks")
@mock.patch("mlflow.genai.judges.utils._invoke_databricks_model")
@mock.patch("mlflow.genai.judges.utils._record_judge_model_usage_failure_databricks_telemetry")
def test_invoke_judge_model_databricks_failure_in_databricks(
    mock_failure_telemetry: mock.Mock, mock_invoke_db: mock.Mock, mock_in_db: mock.Mock
) -> None:
    """Test failure handling with telemetry recording when in Databricks."""
    mock_in_db.return_value = True
    mock_invoke_db.side_effect = MlflowException("Model invocation failed")

    with pytest.raises(MlflowException, match="Model invocation failed"):
        invoke_judge_model(
            model_uri="databricks:/test-model",
            prompt="Test prompt",
            assessment_name="test_assessment",
        )

    # Verify failure telemetry was called
    mock_failure_telemetry.assert_called_once_with(
        model_provider="databricks",
        endpoint_name="test-model",
        error_code="UNKNOWN",
        error_message=mock.ANY,  # Check that error message contains the exception
    )

    # Verify error message contains the traceback
    call_args = mock_failure_telemetry.call_args[1]
    assert "Model invocation failed" in call_args["error_message"]


@mock.patch("mlflow.genai.judges.utils._is_in_databricks")
@mock.patch("mlflow.genai.judges.utils._invoke_databricks_model")
@mock.patch("mlflow.genai.judges.utils._record_judge_model_usage_success_databricks_telemetry")
def test_invoke_judge_model_databricks_telemetry_error_handling(
    mock_success_telemetry: mock.Mock, mock_invoke_db: mock.Mock, mock_in_db: mock.Mock
) -> None:
    """Test that telemetry errors don't break the main flow."""
    mock_in_db.return_value = True
    mock_invoke_db.return_value = InvokeDatabricksModelOutput(
        response='{"result": "yes", "rationale": "Good"}',
        request_id="req-789",
        num_prompt_tokens=5,
        num_completion_tokens=3,
    )
    mock_success_telemetry.side_effect = Exception("Telemetry failed")

    # Should still return feedback despite telemetry failure
    feedback = invoke_judge_model(
        model_uri="databricks:/test-model",
        prompt="Test prompt",
        assessment_name="test_assessment",
    )

    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "Good"
