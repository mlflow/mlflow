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
from mlflow.environment_variables import MLFLOW_JUDGE_MAX_ITERATIONS
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.databricks_managed_judge_adapter import (
    _run_databricks_agentic_loop,
    call_chat_completions,
    create_litellm_message_from_databricks_response,
    serialize_messages_to_databricks_prompts,
)
from mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter import (
    InvokeDatabricksModelOutput,
    _convert_litellm_messages_to_serving_endpoint_api_format,
    _invoke_databricks_serving_endpoint,
    _invoke_databricks_serving_endpoint_judge,
    _parse_databricks_model_response,
    _record_judge_model_usage_failure_databricks_telemetry,
    _record_judge_model_usage_success_databricks_telemetry,
)
from mlflow.types.llm import ChatMessage
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


def test_parse_databricks_model_response_with_tool_calls() -> None:
    res_json = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_root_span", "arguments": "{}"},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    headers = {"x-request-id": "test-request-id"}

    result = _parse_databricks_model_response(res_json, headers)

    assert result.response is None
    assert result.tool_calls is not None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["id"] == "call_123"
    assert result.request_id == "test-request-id"


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
        ({"choices": [{"message": {}}]}, "missing both 'content' and 'tool_calls' fields"),
        (
            {"choices": [{"message": {"content": None}}]},
            "missing both 'content' and 'tool_calls' fields",
        ),
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
            model_name="test-model",
            messages=[{"role": "user", "content": "test prompt"}],
            num_retries=3,
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
                model_name="test-model",
                messages=[{"role": "user", "content": "test prompt"}],
                num_retries=3,
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
            model_name="test-model",
            messages=[{"role": "user", "content": "test prompt"}],
            num_retries=3,
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
                model_name="test-model",
                messages=[{"role": "user", "content": "test prompt"}],
                num_retries=0,
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
                model_name="test-model",
                messages=[{"role": "user", "content": "test prompt"}],
                num_retries=2,
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
            messages=[{"role": "user", "content": "Rate this"}],
            num_retries=1,
            response_format=ResponseFormat,
        )

        # Verify response_format was included in the payload
        assert captured_payload is not None
        assert "response_format" in captured_payload
        assert captured_payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": ResponseFormat.model_json_schema(),
            },
        }

        # Verify the response was returned correctly
        assert output.response == '{"result": 8, "rationale": "Good quality"}'


@pytest.mark.parametrize(
    "error_text",
    [
        '{"error": "Invalid parameter: response_format is not supported"}',
        (
            '{"error_code":"BAD_REQUEST",'
            '"message":"Bad request: json: unknown field \\"response_schema\\""}'
        ),
        (
            '{"error_code":"INVALID_PARAMETER_VALUE",'
            '"message": "INVALID_PARAMETER_VALUE: Response format type object '
            'is not supported for this model."}'
        ),
        '{"error": "ResponseFormatObject is not supported by this model"}',
        '{"error": "Bad request: json: unknown field \\"properties\\""}',
        '{"error": "Bad request: json: unknown field "properties""}',
    ],
)
def test_invoke_databricks_serving_endpoint_response_format_error_detection(
    mock_databricks_creds, error_text: str
):
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
            mock_response.status_code = 400
            mock_response.text = error_text
        else:
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
            model_name="test-model",
            messages=[{"role": "user", "content": "Rate this"}],
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
            messages=[{"role": "user", "content": "test prompt"}],
            num_retries=1,
            inference_params=inference_params,
        )

        # Verify inference_params were included in the payload
        assert captured_payload is not None
        assert captured_payload["temperature"] == 0.5
        assert captured_payload["max_tokens"] == 100
        assert captured_payload["top_p"] == 0.9

    assert result.response == "Test response"


def test_serving_endpoint_judge_without_tools(mock_databricks_creds):
    def mock_post(*args, **kwargs):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"result": "yes", "rationale": "Good"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_response.headers = {"x-request-id": "test-id"}
        return mock_response

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.requests.post",
            side_effect=mock_post,
        ) as mock_requests,
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_databricks_creds,
        ),
    ):
        output = _invoke_databricks_serving_endpoint_judge(
            model_name="test-model",
            prompt="Rate this response",
            assessment_name="test-assessment",
            trace=None,
            num_retries=1,
        )

        assert mock_requests.call_count == 1
        payload = mock_requests.call_args.kwargs["json"]
        assert "tools" not in payload
        assert output.feedback.value == "yes"
        assert output.feedback.rationale == "Good"
        assert output.request_id == "test-id"


def test_serving_endpoint_judge_with_tool_calling(mock_databricks_creds, mock_trace):
    call_count = 0

    def mock_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_response = mock.Mock()
        mock_response.status_code = 200

        if call_count == 1:
            mock_response.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": "Let me check the trace",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {"name": "get_root_span", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            }
        else:
            mock_response.json.return_value = {
                "choices": [{"message": {"content": '{"result": "yes", "rationale": "Verified"}'}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 15},
            }

        mock_response.headers = {"x-request-id": f"test-id-{call_count}"}
        return mock_response

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.requests.post",
            side_effect=mock_post,
        ) as mock_requests,
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_databricks_creds,
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter._process_tool_calls"
        ) as mock_process,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.list_judge_tools"
        ) as mock_list_tools,
    ):
        mock_tool = mock.Mock()
        mock_tool.get_definition.return_value.to_dict.return_value = {"name": "get_root_span"}
        mock_list_tools.return_value = [mock_tool]

        mock_process.return_value = [
            litellm.Message(
                role="tool",
                content='{"name": "root"}',
                tool_call_id="call_123",
                name="get_root_span",
            )
        ]

        output = _invoke_databricks_serving_endpoint_judge(
            model_name="test-model",
            prompt="Analyze the trace",
            assessment_name="test-assessment",
            trace=mock_trace,
            num_retries=1,
        )

        assert mock_requests.call_count == 2
        first_payload = mock_requests.call_args_list[0].kwargs["json"]
        assert "tools" in first_payload
        mock_process.assert_called_once()
        assert output.feedback.value == "yes"
        assert output.feedback.rationale == "Verified"
        assert output.num_prompt_tokens == 50
        assert output.num_completion_tokens == 25


def test_serving_endpoint_judge_disables_response_format_with_tools(
    mock_databricks_creds, mock_trace
):
    class ResponseFormat(BaseModel):
        result: str
        rationale: str

    captured_payload = None

    def mock_post(*args, **kwargs):
        nonlocal captured_payload
        captured_payload = kwargs.get("json")
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"result": "yes", "rationale": "Good"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_response.headers = {"x-request-id": "test-id"}
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
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.list_judge_tools"
        ) as mock_list_tools,
    ):
        mock_tool = mock.Mock()
        mock_tool_def = mock.Mock()
        mock_tool_def.to_dict.return_value = {
            "type": "function",
            "function": {
                "name": "get_root_span",
                "description": "Get root span",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        mock_tool.get_definition.return_value = mock_tool_def
        mock_list_tools.return_value = [mock_tool]

        output = _invoke_databricks_serving_endpoint_judge(
            model_name="test-model",
            prompt="Rate this",
            assessment_name="test-assessment",
            trace=mock_trace,
            num_retries=1,
            response_format=ResponseFormat,
        )

        assert captured_payload is not None
        assert isinstance(captured_payload.get("tools"), list)
        assert len(captured_payload["tools"]) > 0
        assert captured_payload.get("response_format") is None
        assert output.feedback.value == "yes"


@pytest.mark.parametrize(
    ("model_name", "should_filter_strict"),
    [
        ("gpt-4", False),
        ("gpt-3.5-turbo", False),
        ("gpt-4o", False),
        ("claude-3-5-sonnet", True),
        ("anthropic.claude-3-sonnet", True),
        ("gemini-1.5-pro", True),
        ("llama-3-70b", True),
        ("my-custom-model", True),
    ],
    ids=["gpt4", "gpt35", "gpt4o", "claude", "anthropic_claude", "gemini", "llama", "custom"],
)
def test_serving_endpoint_judge_filters_strict_from_tools(
    mock_databricks_creds, mock_trace, model_name, should_filter_strict
):
    captured_payload = None

    def mock_post(*args, **kwargs):
        nonlocal captured_payload
        captured_payload = kwargs.get("json")
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"result": "yes", "rationale": "Good"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_response.headers = {"x-request-id": "test-id"}
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
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.list_judge_tools"
        ) as mock_list_tools,
    ):
        mock_tool = mock.Mock()
        mock_tool_def = mock.Mock()
        mock_tool_def.to_dict.return_value = {
            "type": "function",
            "function": {
                "name": "get_root_span",
                "description": "Get root span",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            },
        }
        mock_tool.get_definition.return_value = mock_tool_def
        mock_list_tools.return_value = [mock_tool]

        output = _invoke_databricks_serving_endpoint_judge(
            model_name=model_name,
            prompt="Rate this",
            assessment_name="test-assessment",
            trace=mock_trace,
            num_retries=1,
        )

        assert captured_payload is not None
        assert "tools" in captured_payload
        assert len(captured_payload["tools"]) == 1

        if should_filter_strict:
            assert "strict" not in captured_payload["tools"][0]["function"]
        else:
            assert captured_payload["tools"][0]["function"]["strict"] is True

        assert captured_payload["tools"][0]["function"]["name"] == "get_root_span"
        assert output.feedback.value == "yes"


def test_serving_endpoint_judge_max_iteration_limit(mock_databricks_creds, mock_trace, monkeypatch):
    monkeypatch.setattr(MLFLOW_JUDGE_MAX_ITERATIONS, "get", lambda: 3)

    def mock_post(*args, **kwargs):
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Calling tool",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "get_root_span", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_response.headers = {"x-request-id": "test-id"}
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
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter._process_tool_calls",
            return_value=[litellm.Message(role="tool", content="{}", tool_call_id="call_123")],
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_serving_endpoint_adapter.list_judge_tools"
        ) as mock_list_tools,
    ):
        mock_tool = mock.Mock()
        mock_tool_def = mock.Mock()
        mock_tool_def.to_dict.return_value = {
            "type": "function",
            "function": {
                "name": "get_root_span",
                "description": "Get root span",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        mock_tool.get_definition.return_value = mock_tool_def
        mock_list_tools.return_value = [mock_tool]

        with pytest.raises(MlflowException, match="Completion iteration limit of 3 exceeded"):
            _invoke_databricks_serving_endpoint_judge(
                model_name="test-model",
                prompt="This will loop forever",
                assessment_name="test-assessment",
                trace=mock_trace,
                num_retries=1,
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


def test_agentic_loop_max_iteration_limit(mock_trace, monkeypatch):
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

    # Set max iterations to 1 for minimal test run
    monkeypatch.setenv("MLFLOW_JUDGE_MAX_ITERATIONS", "1")

    with (
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter.call_chat_completions",
            return_value=mock_response,
        ) as mock_call,
        mock.patch(
            "mlflow.genai.judges.adapters.databricks_managed_judge_adapter._process_tool_calls"
        ) as mock_process,
    ):
        mock_process.return_value = [
            litellm.Message(
                role="tool", content="{}", tool_call_id="call_123", name="get_root_span"
            )
        ]

        with pytest.raises(MlflowException, match="iteration limit of 1 exceeded"):
            _run_databricks_agentic_loop(
                messages=messages,
                trace=mock_trace,
                on_final_answer=callback,
            )

        # Verify we hit the limit (called once before raising)
        assert mock_call.call_count == 1


def _create_message(message_type, role, content=None, **kwargs):
    if message_type == "litellm":
        return litellm.Message(role=role, content=content, **kwargs)
    else:
        return ChatMessage(role=role, content=content, **kwargs)


def _create_tool_call(message_type, tool_id, function_name, arguments):
    if message_type == "litellm":
        return litellm.ChatCompletionMessageToolCall(
            id=tool_id,
            type="function",
            function=litellm.Function(name=function_name, arguments=arguments),
        )
    else:
        from mlflow.types.llm import FunctionToolCallArguments, ToolCall

        return ToolCall(
            id=tool_id,
            type="function",
            function=FunctionToolCallArguments(name=function_name, arguments=arguments),
        )


@pytest.mark.parametrize("message_type", ["litellm", "chatmessage"])
@pytest.mark.parametrize(
    ("messages_builder", "expected_user_prompt", "expected_system_prompt"),
    [
        # System message is extracted separately, user message becomes user_prompt
        (
            lambda mt: [
                _create_message(mt, "system", "You are a helpful assistant"),
                _create_message(mt, "user", "What is 2+2?"),
            ],
            "What is 2+2?",
            "You are a helpful assistant",
        ),
        # No system message results in None for system_prompt
        (
            lambda mt: [_create_message(mt, "user", "What is 2+2?")],
            "What is 2+2?",
            None,
        ),
        # Multiple user messages are concatenated with \n\n separator
        (
            lambda mt: [
                _create_message(mt, "user", "First question"),
                _create_message(mt, "user", "Second question"),
            ],
            "First question\n\nSecond question",
            None,
        ),
        # Assistant messages are prefixed with "Assistant: " in the user_prompt
        (
            lambda mt: [
                _create_message(mt, "user", "What is 2+2?"),
                _create_message(mt, "assistant", "4"),
                _create_message(mt, "user", "What is 3+3?"),
            ],
            "What is 2+2?\n\nAssistant: 4\n\nWhat is 3+3?",
            None,
        ),
        # Assistant with tool calls shows "[Called tools]" placeholder
        (
            lambda mt: [
                _create_message(mt, "user", "Get the root span"),
                _create_message(
                    mt,
                    "assistant",
                    None,
                    tool_calls=[_create_tool_call(mt, "call_123", "get_root_span", "{}")],
                ),
            ],
            "Get the root span\n\nAssistant: [Called tools]",
            None,
        ),
        # Tool messages are formatted as "Tool {name}: {content}"
        (
            lambda mt: [
                _create_message(mt, "user", "Get the root span"),
                _create_message(
                    mt,
                    "assistant",
                    None,
                    tool_calls=[_create_tool_call(mt, "call_123", "get_root_span", "{}")],
                ),
                _create_message(
                    mt,
                    "tool",
                    '{"name": "root_span"}',
                    tool_call_id="call_123",
                    name="get_root_span",
                ),
            ],
            "Get the root span\n\nAssistant: [Called tools]\n\nTool get_root_span: "
            '{"name": "root_span"}',
            None,
        ),
        # Full multi-turn conversation with system prompt
        (
            lambda mt: [
                _create_message(mt, "system", "You are helpful"),
                _create_message(mt, "user", "Question 1"),
                _create_message(mt, "assistant", "Answer 1"),
                _create_message(mt, "user", "Question 2"),
            ],
            "Question 1\n\nAssistant: Answer 1\n\nQuestion 2",
            "You are helpful",
        ),
    ],
    ids=[
        "with_system_message",
        "without_system_message",
        "multiple_user_messages",
        "with_assistant_messages",
        "with_tool_calls",
        "with_tool_messages",
        "full_conversation",
    ],
)
def test_serialize_messages_to_databricks_prompts(
    message_type, messages_builder, expected_user_prompt, expected_system_prompt
):
    messages = messages_builder(message_type)
    user_prompt, system_prompt = serialize_messages_to_databricks_prompts(messages)

    assert user_prompt == expected_user_prompt
    assert system_prompt == expected_system_prompt


@pytest.mark.parametrize(
    ("response_data", "expected_role", "expected_content"),
    [
        # Simple string content is extracted as-is
        (
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "This is a response",
                        }
                    }
                ]
            },
            "assistant",
            "This is a response",
        ),
        # List content with text blocks are concatenated with newlines (reasoning models)
        (
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "First part"},
                                {"type": "text", "text": "Second part"},
                                {"type": "other", "data": "ignored"},
                            ],
                        }
                    }
                ]
            },
            "assistant",
            "First part\nSecond part",
        ),
        # List content with no text blocks results in None
        (
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "other", "data": "no text"}],
                        }
                    }
                ]
            },
            "assistant",
            None,
        ),
    ],
    ids=["string_content", "list_content", "empty_list_content"],
)
def test_create_litellm_message_from_databricks_response(
    response_data, expected_role, expected_content
):
    message = create_litellm_message_from_databricks_response(response_data)

    assert isinstance(message, litellm.Message)
    assert message.role == expected_role
    assert message.content == expected_content
    assert message.tool_calls is None


def test_create_litellm_message_from_databricks_response_with_single_tool_call():
    response_data = {
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

    message = create_litellm_message_from_databricks_response(response_data)

    assert isinstance(message, litellm.Message)
    assert message.role == "assistant"
    assert message.content is None
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].id == "call_123"
    assert message.tool_calls[0].type == "function"
    assert message.tool_calls[0].function.name == "get_root_span"
    assert message.tool_calls[0].function.arguments == "{}"


def test_create_litellm_message_from_databricks_response_with_multiple_tool_calls():
    response_data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_root_span", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "list_spans", "arguments": "{}"},
                        },
                    ],
                }
            }
        ]
    }

    message = create_litellm_message_from_databricks_response(response_data)

    assert isinstance(message, litellm.Message)
    assert message.role == "assistant"
    assert message.content is None
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 2
    assert message.tool_calls[0].id == "call_1"
    assert message.tool_calls[0].function.name == "get_root_span"
    assert message.tool_calls[1].id == "call_2"
    assert message.tool_calls[1].function.name == "list_spans"


@pytest.mark.parametrize(
    ("response_data", "expected_error"),
    [
        # Response without choices field raises ValueError
        ({}, "missing 'choices' field"),
        # Response with empty choices array raises ValueError
        ({"choices": []}, "missing 'choices' field"),
    ],
    ids=["missing_choices", "empty_choices"],
)
def test_create_litellm_message_from_databricks_response_errors(response_data, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        create_litellm_message_from_databricks_response(response_data)


@pytest.mark.parametrize(
    ("messages", "expected", "model_name"),
    [
        pytest.param(
            [
                litellm.Message(role="user", content="Hello"),
                litellm.Message(role="assistant", content="Hi there"),
            ],
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "gpt-4",
            id="simple_messages",
        ),
        pytest.param(
            [
                litellm.Message(
                    role="tool",
                    content='{"name": "root_span"}',
                    tool_call_id="call_123",
                    name="get_root_span",
                )
            ],
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": '{"name": "root_span"}',
                }
            ],
            "gpt-4",
            id="tool_message",
        ),
        pytest.param(
            [
                litellm.Message(
                    role="assistant",
                    content="Let me check",
                    tool_calls=[
                        litellm.ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=litellm.Function(name="get_root_span", arguments="{}"),
                        )
                    ],
                )
            ],
            [
                {
                    "role": "assistant",
                    "content": "Let me check",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_root_span", "arguments": "{}"},
                        }
                    ],
                }
            ],
            "gpt-4",
            id="message_with_tool_calls",
        ),
        pytest.param(
            [
                litellm.Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        litellm.ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=litellm.Function(name="get_root_span", arguments="{}"),
                        )
                    ],
                )
            ],
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_root_span", "arguments": "{}"},
                        }
                    ],
                }
            ],
            "gpt-4",
            id="tool_calls_no_content",
        ),
        pytest.param(
            [
                litellm.Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        litellm.ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=litellm.Function(
                                name="search_spans", arguments={"query": "error", "limit": 10}
                            ),
                        )
                    ],
                )
            ],
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "search_spans",
                                "arguments": '{"query": "error", "limit": 10}',
                            },
                        }
                    ],
                }
            ],
            "gpt-4",
            id="dict_arguments",
        ),
        pytest.param(
            [
                litellm.Message(role="user", content="Analyze the trace"),
                litellm.Message(
                    role="assistant",
                    content="Let me check",
                    tool_calls=[
                        litellm.ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=litellm.Function(name="get_root_span", arguments="{}"),
                        )
                    ],
                ),
                litellm.Message(
                    role="tool",
                    content='{"name": "root"}',
                    tool_call_id="call_123",
                    name="get_root_span",
                ),
                litellm.Message(role="assistant", content="The trace looks good"),
            ],
            [
                {"role": "user", "content": "Analyze the trace"},
                {
                    "role": "assistant",
                    "content": "Let me check",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_root_span", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_123",
                    "content": '{"name": "root"}',
                },
                {"role": "assistant", "content": "The trace looks good"},
            ],
            "gpt-4",
            id="mixed_messages",
        ),
        pytest.param(
            [
                # Create a message with tool_call that has thoughtSignature attribute
                # This simulates what happens when Gemini 3 returns tool calls
                (
                    lambda: (
                        tc := litellm.ChatCompletionMessageToolCall(
                            id="call_gemini_123",
                            type="function",
                            function=litellm.Function(
                                name="get_trace_info",
                                arguments='{"trace_id": "abc"}',
                            ),
                        ),
                        setattr(tc, "thoughtSignature", "Let me check the trace information"),
                        litellm.Message(
                            role="assistant",
                            content=None,
                            tool_calls=[tc],
                        ),
                    )[2]
                )()
            ],
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_gemini_123",
                            "type": "function",
                            "function": {
                                "name": "get_trace_info",
                                "arguments": '{"trace_id": "abc"}',
                            },
                            "thoughtSignature": "Let me check the trace information",
                        }
                    ],
                }
            ],
            "gemini-3-pro",
            id="preserves_thoughtSignature_for_gemini",
        ),
    ],
)
def test_convert_litellm_messages_to_api_format(messages, expected, model_name):
    result = _convert_litellm_messages_to_serving_endpoint_api_format(messages, model_name)
    assert result == expected
