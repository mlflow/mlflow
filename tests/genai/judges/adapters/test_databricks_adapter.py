import json
from typing import Any
from unittest import mock

import pytest
import requests
from pydantic import BaseModel

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    call_chat_completions,
)
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    InvokeDatabricksModelOutput,
    _invoke_databricks_serving_endpoint,
    _parse_databricks_model_response,
    _record_judge_model_usage_failure_databricks_telemetry,
    _record_judge_model_usage_success_databricks_telemetry,
)
from mlflow.utils import AttrDict


@pytest.fixture
def mock_trace():
    trace_info = TraceInfo(
        trace_id="test-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
    )
    return Trace(info=trace_info, data=None)


def test_parse_databricks_model_response_valid_response() -> None:
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
    res_json = {"choices": [{"message": {"content": invalid_content}}]}
    headers: dict[str, str] = {}

    with pytest.raises(MlflowException, match="no text content found"):
        _parse_databricks_model_response(res_json, headers)


@pytest.mark.parametrize(
    ("res_json", "expected_error"),
    [
        ({}, "missing 'choices' field"),
        ({"choices": []}, "missing 'choices' field"),
        ({"choices": [{}]}, "missing 'message' field"),
        ({"choices": [{"message": {}}]}, "missing 'content' field"),
        ({"choices": [{"message": {"content": None}}]}, "missing 'content' field"),
    ],
)
def test_parse_databricks_model_response_errors(
    res_json: dict[str, Any], expected_error: str
) -> None:
    headers: dict[str, str] = {}

    with pytest.raises(MlflowException, match=expected_error):
        _parse_databricks_model_response(res_json, headers)


def test_invoke_databricks_serving_endpoint_successful_invocation() -> None:
    mock_creds = mock.Mock()
    mock_creds.host = "https://test.databricks.com"
    mock_creds.token = "test-token"

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Test response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    mock_response.headers = {"x-request-id": "test-id"}

    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_creds,
        ) as mock_get_creds,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.requests.post",
            return_value=mock_response,
        ) as mock_post,
    ):
        result = _invoke_databricks_serving_endpoint(
            model_name="test-model", prompt="test prompt", num_retries=3
        )

        mock_post.assert_called_once_with(
            url="https://test.databricks.com/serving-endpoints/test-model/invocations",
            headers={"Authorization": "Bearer test-token"},
            json={"messages": [{"role": "user", "content": "test prompt"}]},
        )
        mock_get_creds.assert_called_once()

    assert result.response == "Test response"
    assert result.request_id == "test-id"
    assert result.num_prompt_tokens == 10
    assert result.num_completion_tokens == 5


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_invoke_databricks_serving_endpoint_bad_request_error_no_retry(status_code: int) -> None:
    mock_creds = mock.Mock()
    mock_creds.host = "https://test.databricks.com"
    mock_creds.token = "test-token"

    mock_response = mock.Mock()
    mock_response.status_code = status_code
    mock_response.text = f"Error {status_code}"

    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_creds,
        ) as mock_get_creds,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.requests.post",
            return_value=mock_response,
        ) as mock_post,
    ):
        with pytest.raises(MlflowException, match=f"failed with status {status_code}"):
            _invoke_databricks_serving_endpoint(
                model_name="test-model", prompt="test prompt", num_retries=3
            )

        mock_post.assert_called_once()
        mock_get_creds.assert_called_once()


def test_invoke_databricks_serving_endpoint_retry_logic_with_transient_errors() -> None:
    mock_creds = mock.Mock()
    mock_creds.host = "https://test.databricks.com"
    mock_creds.token = "test-token"

    # First call fails with 500, second succeeds
    error_response = mock.Mock()
    error_response.status_code = 500
    error_response.text = "Internal server error"

    success_response = mock.Mock()
    success_response.status_code = 200
    success_response.json.return_value = {"choices": [{"message": {"content": "Success"}}]}
    success_response.headers = {}

    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_creds,
        ) as mock_get_creds,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.requests.post",
            side_effect=[error_response, success_response],
        ) as mock_post,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.time.sleep"
        ) as mock_sleep,
    ):
        result = _invoke_databricks_serving_endpoint(
            model_name="test-model", prompt="test prompt", num_retries=3
        )

        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1
        mock_get_creds.assert_called_once()

    assert result.response == "Success"


def test_invoke_databricks_serving_endpoint_json_decode_error() -> None:
    mock_creds = mock.Mock()
    mock_creds.host = "https://test.databricks.com"
    mock_creds.token = "test-token"

    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_creds,
        ) as mock_get_creds,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.requests.post",
            return_value=mock_response,
        ) as mock_post,
    ):
        with pytest.raises(MlflowException, match="Failed to parse JSON response"):
            _invoke_databricks_serving_endpoint(
                model_name="test-model", prompt="test prompt", num_retries=0
            )

        mock_post.assert_called_once()
        mock_get_creds.assert_called_once()


def test_invoke_databricks_serving_endpoint_connection_error_with_retries() -> None:
    mock_creds = mock.Mock()

    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_creds,
        ) as mock_get_creds,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.requests.post",
            side_effect=requests.ConnectionError("Connection failed"),
        ) as mock_post,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.time.sleep"
        ) as mock_sleep,
    ):
        with pytest.raises(
            MlflowException, match="Failed to invoke Databricks model after 3 attempts"
        ):
            _invoke_databricks_serving_endpoint(
                model_name="test-model", prompt="test prompt", num_retries=2
            )

        assert mock_post.call_count == 3  # Initial + 2 retries
        assert mock_sleep.call_count == 2
        mock_get_creds.assert_called_once()


def test_invoke_databricks_serving_endpoint_with_response_schema():
    class ResponseFormat(BaseModel):
        result: int
        rationale: str

    # Mock the Databricks API call
    captured_payload = None

    def mock_post(*args, **kwargs):
        nonlocal captured_payload
        captured_payload = kwargs.get("json")

        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"result": 8, "rationale": "Good quality"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "id": "test-request-id",
        }
        return mock_response

    # Mock Databricks host creds
    mock_creds = mock.Mock()
    mock_creds.host = "https://test.databricks.com"
    mock_creds.token = "test-token"

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.requests.post",
            side_effect=mock_post,
        ),
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds", return_value=mock_creds
        ),
    ):
        output = _invoke_databricks_serving_endpoint(
            model_name="my-endpoint",
            prompt="Rate this",
            num_retries=1,
            response_format=ResponseFormat,
        )

        # Verify response_schema was included in the payload
        assert captured_payload is not None
        assert "response_schema" in captured_payload
        assert captured_payload["response_schema"] == ResponseFormat.model_json_schema()

        # Verify the response was returned correctly
        assert output.response == '{"result": 8, "rationale": "Good quality"}'


def test_invoke_databricks_serving_endpoint_with_inference_params() -> None:
    captured_payload = None

    def mock_post(*args, **kwargs):
        nonlocal captured_payload
        captured_payload = kwargs.get("json")

        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_response.headers = {"x-request-id": "test-id"}
        return mock_response

    mock_creds = mock.Mock()
    mock_creds.host = "https://test.databricks.com"
    mock_creds.token = "test-token"

    inference_params = {"temperature": 0.5, "max_tokens": 100, "top_p": 0.9}

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.requests.post",
            side_effect=mock_post,
        ),
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds", return_value=mock_creds
        ),
    ):
        result = _invoke_databricks_serving_endpoint(
            model_name="test-model",
            prompt="test prompt",
            num_retries=1,
            inference_params=inference_params,
        )

        # Verify inference_params were included in the payload
        assert captured_payload is not None
        assert captured_payload["temperature"] == 0.5
        assert captured_payload["max_tokens"] == 100
        assert captured_payload["top_p"] == 0.9

    assert result.response == "Test response"


@pytest.mark.parametrize(
    "invalid_prompt",
    [
        123,  # Not a string or list
        ["not a ChatMessage", "also invalid"],  # List but not ChatMessage instances
    ],
)
def test_invoke_databricks_serving_endpoint_invalid_prompt_type(invalid_prompt) -> None:
    mock_creds = mock.Mock()
    mock_creds.host = "https://test.databricks.com"
    mock_creds.token = "test-token"

    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_creds,
        ),
    ):
        with pytest.raises(
            MlflowException,
            match="Invalid prompt type: expected str or list\\[ChatMessage\\]",
        ):
            _invoke_databricks_serving_endpoint(
                model_name="test-model", prompt=invalid_prompt, num_retries=0
            )


def test_record_success_telemetry_with_databricks_agents() -> None:
    # Mock the telemetry function separately
    mock_telemetry_module = mock.MagicMock()
    mock_record = mock.MagicMock()
    mock_telemetry_module.record_judge_model_usage_success = mock_record

    with (
        mock.patch(
            "mlflow.tracking.fluent._get_experiment_id",
            return_value="exp-123",
        ) as mock_experiment_id,
        mock.patch(
            "mlflow.utils.databricks_utils.get_workspace_id",
            return_value="ws-456",
        ) as mock_workspace_id,
        mock.patch(
            "mlflow.utils.databricks_utils.get_job_id",
            return_value="job-789",
        ) as mock_job_id,
        mock.patch(
            "mlflow.utils.databricks_utils.get_job_run_id",
            return_value="run-101",
        ) as mock_job_run_id,
        mock.patch.dict(
            "sys.modules",
            {"databricks.agents.telemetry": mock_telemetry_module},
        ),
    ):
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
        mock_experiment_id.assert_called_once()
        mock_workspace_id.assert_called_once()
        mock_job_id.assert_called_once()
        mock_job_run_id.assert_called_once()


def test_record_success_telemetry_without_databricks_agents() -> None:
    with mock.patch.dict("sys.modules", {"databricks.agents.telemetry": None}):
        # Should not raise exception
        _record_judge_model_usage_success_databricks_telemetry(
            request_id="req-123",
            model_provider="databricks",
            endpoint_name="test-endpoint",
            num_prompt_tokens=10,
            num_completion_tokens=5,
        )


def test_record_failure_telemetry_with_databricks_agents() -> None:
    # Mock the telemetry function separately
    mock_telemetry_module = mock.MagicMock()
    mock_record = mock.MagicMock()
    mock_telemetry_module.record_judge_model_usage_failure = mock_record

    with (
        mock.patch(
            "mlflow.tracking.fluent._get_experiment_id",
            return_value="exp-123",
        ) as mock_experiment_id,
        mock.patch(
            "mlflow.utils.databricks_utils.get_workspace_id",
            return_value="ws-456",
        ) as mock_workspace_id,
        mock.patch(
            "mlflow.utils.databricks_utils.get_job_id",
            return_value="job-789",
        ) as mock_job_id,
        mock.patch(
            "mlflow.utils.databricks_utils.get_job_run_id",
            return_value="run-101",
        ) as mock_job_run_id,
        mock.patch.dict(
            "sys.modules",
            {"databricks.agents.telemetry": mock_telemetry_module},
        ),
    ):
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
        mock_experiment_id.assert_called_once()
        mock_workspace_id.assert_called_once()
        mock_job_id.assert_called_once()
        mock_job_run_id.assert_called_once()


@pytest.fixture
def mock_databricks_rag_eval():
    mock_rag_client = mock.MagicMock()
    mock_rag_client.get_chat_completions_result.return_value = AttrDict(
        {"output": "test response", "error_message": None}
    )

    mock_context = mock.MagicMock()
    mock_context.get_context.return_value.build_managed_rag_client.return_value = mock_rag_client
    mock_context.eval_context = lambda func: func

    mock_env_vars = mock.MagicMock()

    mock_module = mock.MagicMock()
    mock_module.context = mock_context
    mock_module.env_vars = mock_env_vars

    return {"module": mock_module, "rag_client": mock_rag_client, "env_vars": mock_env_vars}


@pytest.mark.parametrize(
    ("user_prompt", "system_prompt"),
    [
        ("test user prompt", "test system prompt"),
        ("user prompt only", None),
    ],
)
def test_call_chat_completions_success(user_prompt, system_prompt, mock_databricks_rag_eval):
    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._check_databricks_agents_installed"
        ),
        mock.patch.dict("sys.modules", {"databricks.rag_eval": mock_databricks_rag_eval["module"]}),
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.VERSION", "1.0.0"
        ),
    ):
        result = call_chat_completions(user_prompt, system_prompt)

        # Verify the client name was set
        mock_databricks_rag_eval[
            "env_vars"
        ].RAG_EVAL_EVAL_SESSION_CLIENT_NAME.set.assert_called_once_with("mlflow-v1.0.0")

        # Verify the managed RAG client was called with correct parameters
        mock_databricks_rag_eval["rag_client"].get_chat_completions_result.assert_called_once_with(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        assert result.output == "test response"


def test_call_chat_completions_with_custom_session_name(mock_databricks_rag_eval):
    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._check_databricks_agents_installed"
        ),
        mock.patch.dict("sys.modules", {"databricks.rag_eval": mock_databricks_rag_eval["module"]}),
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.VERSION", "1.0.0"
        ),
    ):
        custom_session_name = "custom-session-name"
        result = call_chat_completions(
            "test prompt", "system prompt", session_name=custom_session_name
        )

        # Verify the custom session name was set
        mock_databricks_rag_eval[
            "env_vars"
        ].RAG_EVAL_EVAL_SESSION_CLIENT_NAME.set.assert_called_once_with(custom_session_name)

        # Verify the managed RAG client was called with correct parameters
        mock_databricks_rag_eval["rag_client"].get_chat_completions_result.assert_called_once_with(
            user_prompt="test prompt",
            system_prompt="system prompt",
        )

        assert result.output == "test response"


def test_call_chat_completions_client_error(mock_databricks_rag_eval):
    mock_databricks_rag_eval["rag_client"].get_chat_completions_result.side_effect = RuntimeError(
        "RAG client failed"
    )

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._check_databricks_agents_installed"
        ),
        mock.patch.dict("sys.modules", {"databricks.rag_eval": mock_databricks_rag_eval["module"]}),
    ):
        with pytest.raises(RuntimeError, match="RAG client failed"):
            call_chat_completions("test prompt", "system prompt")


def test_call_chat_completions_with_use_case_supported():
    call_args = None

    # Create a mock client with the real method (not a MagicMock) so inspect.signature works
    class MockRAGClient:
        def get_chat_completions_result(self, user_prompt, system_prompt, use_case=None):
            nonlocal call_args
            call_args = {
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "use_case": use_case,
            }
            return AttrDict({"output": "test response", "error_message": None})

    mock_context = mock.MagicMock()
    mock_context.get_context.return_value.build_managed_rag_client.return_value = MockRAGClient()
    mock_context.eval_context = lambda func: func

    mock_env_vars = mock.MagicMock()

    mock_module = mock.MagicMock()
    mock_module.context = mock_context
    mock_module.env_vars = mock_env_vars

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._check_databricks_agents_installed"
        ),
        mock.patch.dict("sys.modules", {"databricks.rag_eval": mock_module}),
    ):
        result = call_chat_completions("test prompt", "system prompt", use_case="judge_alignment")

        # Verify use_case was passed since the method signature supports it
        assert call_args == {
            "user_prompt": "test prompt",
            "system_prompt": "system prompt",
            "use_case": "judge_alignment",
        }

        assert result.output == "test response"


def test_call_chat_completions_with_use_case_not_supported(mock_databricks_rag_eval):
    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._check_databricks_agents_installed"
        ),
        mock.patch.dict("sys.modules", {"databricks.rag_eval": mock_databricks_rag_eval["module"]}),
    ):
        # Even though we pass use_case, it won't be forwarded since the mock doesn't support it
        result = call_chat_completions("test prompt", "system prompt", use_case="judge_alignment")

        # Verify use_case was NOT passed (backward compatibility)
        mock_databricks_rag_eval["rag_client"].get_chat_completions_result.assert_called_once_with(
            user_prompt="test prompt",
            system_prompt="system prompt",
        )

        assert result.output == "test response"
