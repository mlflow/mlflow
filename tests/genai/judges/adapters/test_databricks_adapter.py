import json
from typing import Any
from unittest import mock

import litellm
import pytest
import requests
from pydantic import BaseModel

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    _run_databricks_agentic_loop,
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


@pytest.fixture
def mock_databricks_creds():
    """Fixture for mocked Databricks credentials."""
    creds = mock.Mock()
    creds.host = "https://test.databricks.com"
    creds.token = "test-token"
    return creds


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


def test_invoke_databricks_serving_endpoint_successful_invocation(
    mock_databricks_creds,
) -> None:
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
            return_value=mock_databricks_creds,
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
def test_invoke_databricks_serving_endpoint_bad_request_error_no_retry(
    mock_databricks_creds, status_code: int
) -> None:
    mock_response = mock.Mock()
    mock_response.status_code = status_code
    mock_response.text = f"Error {status_code}"

    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_databricks_creds,
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


def test_invoke_databricks_serving_endpoint_retry_logic_with_transient_errors(
    mock_databricks_creds,
) -> None:
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
            return_value=mock_databricks_creds,
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


def test_invoke_databricks_serving_endpoint_json_decode_error(mock_databricks_creds) -> None:
    mock_response = mock.Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_databricks_creds,
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


def test_invoke_databricks_serving_endpoint_connection_error_with_retries(
    mock_databricks_creds,
) -> None:
    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_databricks_creds,
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


def test_invoke_databricks_serving_endpoint_with_response_format(mock_databricks_creds):
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

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.requests.post",
            side_effect=mock_post,
        ),
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_databricks_creds,
        ),
    ):
        output = _invoke_databricks_serving_endpoint(
            model_name="my-endpoint",
            prompt="Rate this",
            num_retries=1,
            response_format=ResponseFormat,
        )

        # Verify response_format was included in the payload
        assert captured_payload is not None
        assert "response_format" in captured_payload
        assert captured_payload["response_format"] == ResponseFormat.model_json_schema()

        # Verify the response was returned correctly
        assert output.response == '{"result": 8, "rationale": "Good quality"}'


def test_invoke_databricks_serving_endpoint_response_format_fallback(mock_databricks_creds):
    class ResponseFormat(BaseModel):
        result: int
        rationale: str

    call_count = 0
    captured_payloads = []

    def mock_post(*args, **kwargs):
        nonlocal call_count
        captured_payloads.append(kwargs.get("json"))
        call_count += 1

        mock_response = mock.Mock()
        if call_count == 1:
            # First call with response_format fails
            mock_response.status_code = 400
            mock_response.text = '{"error": "Invalid parameter: response_format is not supported"}'
        else:
            # Second call without response_format succeeds
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"content": '{"result": 8, "rationale": "Good quality"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "id": "test-request-id",
            }
        return mock_response

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.requests.post",
            side_effect=mock_post,
        ),
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_databricks_creds,
        ),
    ):
        output = _invoke_databricks_serving_endpoint(
            model_name="my-endpoint",
            prompt="Rate this",
            num_retries=1,
            response_format=ResponseFormat,
        )

        # Verify we made 2 calls, the first with response_format, the second without
        assert call_count == 2
        assert "response_format" in captured_payloads[0]
        assert "response_format" not in captured_payloads[1]
        assert output.response == '{"result": 8, "rationale": "Good quality"}'


def test_invoke_databricks_serving_endpoint_with_inference_params(mock_databricks_creds) -> None:
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

    inference_params = {"temperature": 0.5, "max_tokens": 100, "top_p": 0.9}

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.requests.post",
            side_effect=mock_post,
        ),
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_databricks_creds,
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
def test_invoke_databricks_serving_endpoint_invalid_prompt_type(
    mock_databricks_creds, invalid_prompt
) -> None:
    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_databricks_creds,
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


# Tests for _run_databricks_agentic_loop


def test_agentic_loop_final_answer_without_tool_calls():
    # Mock response with no tool calls (final answer)
    mock_response = AttrDict(
        {
            "output_json": json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": '{"result": "yes", "rationale": "Looks good"}',
                            }
                        }
                    ]
                }
            )
        }
    )

    messages = [litellm.Message(role="user", content="Test prompt")]
    callback_called_with = []

    def callback(content):
        callback_called_with.append(content)
        return {"parsed": content}

    with mock.patch(
        "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.call_chat_completions",
        return_value=mock_response,
    ) as mock_call:
        result = _run_databricks_agentic_loop(
            messages=messages,
            trace=None,
            on_final_answer=callback,
        )

        # Verify callback was called with the content
        assert len(callback_called_with) == 1
        assert callback_called_with[0] == '{"result": "yes", "rationale": "Looks good"}'
        assert result == {"parsed": '{"result": "yes", "rationale": "Looks good"}'}

        # Verify call_chat_completions was called once (no loop)
        assert mock_call.call_count == 1


def test_agentic_loop_tool_calling_loop(mock_trace):
    # First response has tool calls, second response is final answer
    mock_responses = [
        # First call: LLM requests tool call
        AttrDict(
            {
                "output_json": json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_123",
                                            "type": "function",
                                            "function": {
                                                "name": "get_root_span",
                                                "arguments": "{}",
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                )
            }
        ),
        # Second call: LLM returns final answer
        AttrDict(
            {
                "output_json": json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": '{"outputs": "The answer is 42"}',
                                }
                            }
                        ]
                    }
                )
            }
        ),
    ]

    messages = [litellm.Message(role="user", content="Extract outputs")]

    def callback(content):
        return {"result": content}

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.call_chat_completions",
            side_effect=mock_responses,
        ) as mock_call,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._process_tool_calls"
        ) as mock_process,
    ):
        mock_process.return_value = [
            litellm.Message(
                role="tool",
                content='{"name": "root_span", "inputs": null}',
                tool_call_id="call_123",
                name="get_root_span",
            )
        ]

        result = _run_databricks_agentic_loop(
            messages=messages,
            trace=mock_trace,
            on_final_answer=callback,
        )

        # Verify we looped twice
        assert mock_call.call_count == 2

        # Verify tool calls were processed
        mock_process.assert_called_once()

        # Verify final result
        assert result == {"result": '{"outputs": "The answer is 42"}'}


def test_agentic_loop_max_iteration_limit(mock_trace):
    # Always return tool calls (never a final answer)
    mock_response = AttrDict(
        {
            "output_json": json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "type": "function",
                                        "function": {
                                            "name": "get_root_span",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            )
        }
    )

    messages = [litellm.Message(role="user", content="Extract outputs")]

    def callback(content):
        return content

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.call_chat_completions",
            return_value=mock_response,
        ) as mock_call,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._process_tool_calls"
        ) as mock_process,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.MLFLOW_JUDGE_MAX_ITERATIONS"
        ) as mock_max_iter,
    ):
        mock_process.return_value = [
            litellm.Message(
                role="tool", content="{}", tool_call_id="call_123", name="get_root_span"
            )
        ]
        # Set max iterations to 3 for faster test
        mock_max_iter.get.return_value = 3

        with pytest.raises(MlflowException, match="iteration limit of 3 exceeded"):
            _run_databricks_agentic_loop(
                messages=messages,
                trace=mock_trace,
                on_final_answer=callback,
            )

        # Verify we hit the limit (called 3 times before raising)
        assert mock_call.call_count == 3


def test_agentic_loop_callback_exception_propagation():
    # Mock response with final answer
    mock_response = AttrDict(
        {
            "output_json": json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "invalid json content",
                            }
                        }
                    ]
                }
            )
        }
    )

    messages = [litellm.Message(role="user", content="Test prompt")]

    def callback(content):
        # Simulate parsing error in callback
        raise ValueError("Failed to parse response")

    with mock.patch(
        "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.call_chat_completions",
        return_value=mock_response,
    ):
        with pytest.raises(ValueError, match="Failed to parse response"):
            _run_databricks_agentic_loop(
                messages=messages,
                trace=None,
                on_final_answer=callback,
            )
