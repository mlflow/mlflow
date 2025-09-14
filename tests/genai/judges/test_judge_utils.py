import json
from typing import Any
from unittest import mock

import litellm
import pytest
import requests
from litellm.types.utils import ModelResponse

from mlflow.entities.assessment import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import Judge
from mlflow.genai.judges.optimizers.simba import SIMBAAlignmentOptimizer
from mlflow.genai.judges.utils import (
    _MODEL_RESPONSE_FORMAT_CAPABILITIES,
    CategoricalRating,
    InvokeDatabricksModelOutput,
    _invoke_databricks_model,
    _parse_databricks_model_response,
    add_output_format_instructions,
    call_chat_completions,
    format_prompt,
    get_default_optimizer,
    invoke_judge_model,
)
from mlflow.genai.prompts.utils import format_prompt
from mlflow.types.llm import ChatMessage, ToolCall
from mlflow.utils import AttrDict


@pytest.fixture(autouse=True)
def clear_model_capabilities_cache():
    """Clear the global model capabilities cache before each test."""
    from mlflow.genai.judges.utils import _MODEL_RESPONSE_FORMAT_CAPABILITIES

    _MODEL_RESPONSE_FORMAT_CAPABILITIES.clear()


@pytest.fixture
def mock_response():
    """Fixture that creates a mock ModelResponse with default result and rationale."""
    content = json.dumps({"result": "yes", "rationale": "The response meets all criteria."})
    return ModelResponse(choices=[{"message": {"content": content}}])


@pytest.fixture
def mock_tool_response():
    """Fixture that creates a mock ModelResponse with tool calls."""
    tool_calls = [{"id": "call_123", "function": {"name": "get_trace_info", "arguments": "{}"}}]
    return ModelResponse(choices=[{"message": {"tool_calls": tool_calls, "content": None}}])


@pytest.fixture
def mock_trace():
    """Fixture that creates a test Trace object."""
    trace_info = TraceInfo(
        trace_id="test-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
    )
    return Trace(info=trace_info, data=None)


@pytest.mark.parametrize("num_retries", [None, 3])
def test_invoke_judge_model_successful_with_litellm(num_retries, mock_response):
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

    # Check that the messages were converted to litellm.Message objects
    call_args = mock_litellm.call_args
    assert len(call_args.kwargs["messages"]) == 1
    msg = call_args.kwargs["messages"][0]
    assert isinstance(msg, litellm.Message)
    assert msg.role == "user"
    assert msg.content == "Evaluate this response"

    mock_litellm.assert_called_once_with(
        model="openai/gpt-4",
        messages=mock.ANY,  # We check the messages separately above
        tools=None,
        tool_choice=None,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "judge_evaluation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "result": {"type": "string", "description": "The evaluation rating/result"},
                        "rationale": {
                            "type": "string",
                            "description": "Detailed explanation for the evaluation",
                        },
                    },
                    "required": ["result", "rationale"],
                    "additionalProperties": False,
                },
            },
        },
        retry_policy=expected_retry_policy,
        retry_strategy="exponential_backoff_retry",
        max_retries=0,
        drop_params=True,
    )

    assert feedback.name == "quality_check"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The response meets all criteria."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4"


def test_invoke_judge_model_with_chat_messages(mock_response):
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant"),
        ChatMessage(role="user", content="Evaluate this response"),
    ]

    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt=messages,
            assessment_name="quality_check",
        )

    mock_litellm.assert_called_once()
    call_args = mock_litellm.call_args
    messages_arg = call_args.kwargs["messages"]

    assert len(messages_arg) == 2
    assert isinstance(messages_arg[0], litellm.Message)
    assert messages_arg[0].role == "system"
    assert messages_arg[0].content == "You are a helpful assistant"
    assert isinstance(messages_arg[1], litellm.Message)
    assert messages_arg[1].role == "user"
    assert messages_arg[1].content == "Evaluate this response"

    assert feedback.name == "quality_check"
    assert feedback.value == CategoricalRating.YES


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
        payload=[{"role": "user", "content": "Evaluate this response"}],
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


def test_invoke_judge_model_with_trace_requires_litellm(mock_trace):
    with pytest.raises(MlflowException, match=r"LiteLLM is required for using traces with judges"):
        with mock.patch("mlflow.genai.judges.utils._is_litellm_available", return_value=False):
            invoke_judge_model(
                model_uri="openai:/gpt-4",
                prompt="Test prompt",
                assessment_name="test",
                trace=mock_trace,
            )


def test_invoke_judge_model_invalid_json_response():
    mock_content = "This is not valid JSON"
    mock_response = ModelResponse(choices=[{"message": {"content": mock_content}}])

    with mock.patch("litellm.completion", return_value=mock_response):
        with pytest.raises(MlflowException, match=r"Failed to parse"):
            invoke_judge_model(
                model_uri="openai:/gpt-4", prompt="Test prompt", assessment_name="test"
            )


def test_invoke_judge_model_with_trace_passes_tools(mock_trace, mock_response):
    with (
        mock.patch("litellm.completion", return_value=mock_response) as mock_litellm,
        mock.patch("mlflow.genai.judges.tools.list_judge_tools") as mock_list_tools,
    ):
        # Mock some tools being available
        mock_tool1 = mock.Mock()
        mock_tool1.name = "get_trace_info"
        mock_tool1.get_definition.return_value.to_dict.return_value = {
            "name": "get_trace_info",
            "description": "Get trace info",
        }

        mock_tool2 = mock.Mock()
        mock_tool2.name = "list_spans"
        mock_tool2.get_definition.return_value.to_dict.return_value = {
            "name": "list_spans",
            "description": "List spans",
        }

        mock_list_tools.return_value = [mock_tool1, mock_tool2]

        invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check",
            trace=mock_trace,
        )

    # Verify tools were passed to litellm completion
    mock_litellm.assert_called_once()
    call_kwargs = mock_litellm.call_args.kwargs
    assert call_kwargs["tools"] == [
        {"name": "get_trace_info", "description": "Get trace info"},
        {"name": "list_spans", "description": "List spans"},
    ]
    assert call_kwargs["tool_choice"] == "auto"


def test_invoke_judge_model_tool_calling_loop(mock_trace):
    # First call: model requests tool call
    mock_tool_call_response = ModelResponse(
        choices=[
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "function": {"name": "get_trace_info", "arguments": "{}"},
                        }
                    ],
                    "content": None,
                }
            }
        ]
    )

    # Second call: model provides final answer
    mock_final_response = ModelResponse(
        choices=[
            {
                "message": {
                    "content": json.dumps({"result": "yes", "rationale": "The trace looks good."})
                }
            }
        ]
    )

    with (
        mock.patch(
            "litellm.completion", side_effect=[mock_tool_call_response, mock_final_response]
        ) as mock_litellm,
        mock.patch("mlflow.genai.judges.tools.list_judge_tools") as mock_list_tools,
        mock.patch(
            "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
        ) as mock_invoke_tool,
    ):
        mock_tool = mock.Mock()
        mock_tool.name = "get_trace_info"
        mock_tool.get_definition.return_value.to_dict.return_value = {"name": "get_trace_info"}
        mock_list_tools.return_value = [mock_tool]

        mock_invoke_tool.return_value = {"trace_id": "test-trace", "state": "OK"}

        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check",
            trace=mock_trace,
        )

    # Verify litellm.completion was called twice (tool call + final response)
    assert mock_litellm.call_count == 2

    # Verify tool was invoked
    mock_invoke_tool.assert_called_once()
    tool_call_arg = mock_invoke_tool.call_args.kwargs["tool_call"]
    assert isinstance(tool_call_arg, ToolCall)
    assert tool_call_arg.function.name == "get_trace_info"

    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The trace looks good."


def test_add_output_format_instructions():
    output_fields = Judge.get_output_fields()

    simple_prompt = "Evaluate this response"
    formatted = add_output_format_instructions(simple_prompt, output_fields=output_fields)

    assert simple_prompt in formatted
    assert "JSON format" in formatted
    assert '"result"' in formatted
    assert '"rationale"' in formatted
    assert "no markdown" in formatted.lower()
    assert "The evaluation rating/result" in formatted
    assert "Detailed explanation for the evaluation" in formatted

    complex_prompt = "This is a multi-line\nprompt with various\ninstruction details"
    formatted = add_output_format_instructions(complex_prompt, output_fields=output_fields)

    assert complex_prompt in formatted
    assert formatted.startswith(complex_prompt)
    assert formatted.endswith("}")

    assert formatted.index(complex_prompt) < formatted.index("JSON format")
    assert formatted.index(complex_prompt) < formatted.index('"result"')
    assert formatted.index(complex_prompt) < formatted.index('"rationale"')


@pytest.mark.parametrize(
    ("error_type", "error_class"),
    [
        ("BadRequestError", litellm.BadRequestError),
        ("UnsupportedParamsError", litellm.UnsupportedParamsError),
    ],
)
def test_invoke_judge_model_retries_without_response_format_on_bad_request(
    mock_response, error_type, error_class
):
    """
    Test that when BadRequestError or UnsupportedParamsError occurs, we retry
    without response_format.
    """
    error = error_class(
        message="response_format not supported", model="openai/gpt-4", llm_provider="openai"
    )

    with mock.patch("litellm.completion", side_effect=[error, mock_response]) as mock_litellm:
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Test prompt",
            assessment_name="test",
        )

        # Should have been called twice - once with response_format, once without
        assert mock_litellm.call_count == 2

        # First call should include response_format
        first_call_kwargs = mock_litellm.call_args_list[0].kwargs
        assert "response_format" in first_call_kwargs

        # Second call should not include response_format
        second_call_kwargs = mock_litellm.call_args_list[1].kwargs
        assert "response_format" not in second_call_kwargs

        # Should still return valid feedback
        assert feedback.name == "test"
        assert feedback.value == CategoricalRating.YES


def test_invoke_judge_model_stops_trying_response_format_after_failure():
    """Test that after BadRequestError, subsequent tool calls don't try response_format."""
    bad_request_error = litellm.BadRequestError(
        message="response_format not supported", model="openai/gpt-4", llm_provider="openai"
    )

    # Mock responses for: initial fail, retry success, tool call 1, tool call 2
    tool_call_response = ModelResponse(
        choices=[
            {
                "message": {
                    "tool_calls": [
                        {"id": "call_123", "function": {"name": "test_tool", "arguments": "{}"}}
                    ],
                    "content": None,
                }
            }
        ]
    )

    success_response = ModelResponse(
        choices=[{"message": {"content": '{"result": "yes", "rationale": "Test rationale"}'}}]
    )

    with (
        mock.patch(
            "litellm.completion",
            side_effect=[
                bad_request_error,  # First call fails with response_format
                tool_call_response,  # Retry succeeds without response_format, returns tool call
                success_response,  # Tool call response succeeds (no response_format)
            ],
        ) as mock_litellm,
        mock.patch("mlflow.genai.judges.tools.list_judge_tools") as mock_list_tools,
        mock.patch("mlflow.genai.judges.tools.registry._judge_tool_registry.invoke") as mock_invoke,
    ):
        mock_tool = mock.Mock()
        mock_tool.get_definition.return_value.to_dict.return_value = {"name": "test_tool"}
        mock_list_tools.return_value = [mock_tool]
        mock_invoke.return_value = {"result": "tool executed"}

        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Test prompt",
            assessment_name="test",
            trace=mock.Mock(),  # Include trace to enable tool calls
        )

        # Should have been called 3 times total
        assert mock_litellm.call_count == 3

        # First call should include response_format and fail
        first_call_kwargs = mock_litellm.call_args_list[0].kwargs
        assert "response_format" in first_call_kwargs

        # Second call should not include response_format and succeed with tool call
        second_call_kwargs = mock_litellm.call_args_list[1].kwargs
        assert "response_format" not in second_call_kwargs

        # Third call (after tool execution) should also not include response_format
        third_call_kwargs = mock_litellm.call_args_list[2].kwargs
        assert "response_format" not in third_call_kwargs

        assert feedback.name == "test"


def test_invoke_judge_model_caches_capabilities_globally():
    """Test that model capabilities are cached globally across function calls."""
    from mlflow.genai.judges.utils import _MODEL_RESPONSE_FORMAT_CAPABILITIES

    bad_request_error = litellm.BadRequestError(
        message="response_format not supported", model="openai/gpt-4", llm_provider="openai"
    )

    success_response = ModelResponse(
        choices=[{"message": {"content": '{"result": "yes", "rationale": "Test rationale"}'}}]
    )

    # First call - should try response_format and cache the failure
    with mock.patch(
        "litellm.completion", side_effect=[bad_request_error, success_response]
    ) as mock_litellm:
        feedback1 = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Test prompt 1",
            assessment_name="test1",
        )

        # Should have been called twice (initial fail + retry)
        assert mock_litellm.call_count == 2
        assert feedback1.name == "test1"

        # Verify capability was cached
        assert _MODEL_RESPONSE_FORMAT_CAPABILITIES.get("openai/gpt-4") is False

    # Second call - should directly use cached capability (no response_format)
    with mock.patch("litellm.completion", return_value=success_response) as mock_litellm_2:
        feedback2 = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Test prompt 2",
            assessment_name="test2",
        )

        # Should only be called once (no retry needed)
        assert mock_litellm_2.call_count == 1

        # Should not include response_format
        call_kwargs = mock_litellm_2.call_args.kwargs
        assert "response_format" not in call_kwargs

        assert feedback2.name == "test2"


def test_unsupported_response_format_handling_supports_multiple_threads():
    """
    When an LLM is invoked with structured outputs and returns a BadRequestsError,
    we cache the lack of support for structured outputs ("response_format") in a
    "model capability cache" to avoid retrying with structured outputs again.

    This test simulates a race condition where another thread modifies the
    model capability cache between the initial check and the exception handler,
    ensuring that we still retry correctly without response_format.
    """
    model_key = "openai/gpt-4-race-bug"
    _MODEL_RESPONSE_FORMAT_CAPABILITIES.clear()

    bad_request_error = litellm.BadRequestError(
        message="response_format not supported", model=model_key, llm_provider="openai"
    )

    call_count = 0
    capabilities_cache_call_count = 0

    def mock_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        if "response_format" in kwargs:
            raise bad_request_error
        else:
            return ModelResponse(
                choices=[{"message": {"content": '{"result": "yes", "rationale": "Success"}'}}]
            )

    class MockCapabilitiesCache(dict):
        """Mock cache that simulates the race condition."""

        def get(self, key, default=None):
            nonlocal capabilities_cache_call_count
            capabilities_cache_call_count += 1

            if capabilities_cache_call_count == 1:
                return True
            elif capabilities_cache_call_count == 2:
                return False
            else:
                return False

    with (
        mock.patch("litellm.completion", side_effect=mock_completion),
        mock.patch(
            "mlflow.genai.judges.utils._MODEL_RESPONSE_FORMAT_CAPABILITIES", MockCapabilitiesCache()
        ),
    ):
        result = invoke_judge_model(
            model_uri=f"openai:/{model_key}",
            prompt="Test prompt",
            assessment_name="race_bug_test",
        )

        assert call_count == 2, "Should make 2 calls: initial (fails) + retry (succeeds)"
        assert capabilities_cache_call_count == 1
        assert result.value == "yes"


def test_litellm_nonfatal_error_messages_suppressed():
    """Test that LiteLLM nonfatal error messages are suppressed during judge execution."""
    suppression_state_during_call = {}

    def mock_completion(**kwargs):
        # Capture the state of litellm flags during the call
        suppression_state_during_call["set_verbose"] = litellm.set_verbose
        suppression_state_during_call["suppress_debug_info"] = litellm.suppress_debug_info

        return ModelResponse(
            choices=[{"message": {"content": '{"result": "pass", "rationale": "Test completed"}'}}]
        )

    with mock.patch("litellm.completion", side_effect=mock_completion):
        # Call invoke_judge_model - the decorator should suppress litellm messages
        result = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Test prompt for suppression",
            assessment_name="suppression_test",
        )

        # Verify suppression was active during the litellm.completion call
        assert suppression_state_during_call["set_verbose"] is False
        assert suppression_state_during_call["suppress_debug_info"] is True

        # Verify the call succeeded
        assert result.value == "pass"


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


def test_invoke_databricks_model_successful_invocation() -> None:
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
            "mlflow.genai.judges.utils.requests.post",
            return_value=mock_response,
        ) as mock_post,
    ):
        result = _invoke_databricks_model(
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
def test_invoke_databricks_model_bad_request_error_no_retry(status_code: int) -> None:
    """Test that 400/401/403/404 errors are not retried."""
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
            "mlflow.genai.judges.utils.requests.post",
            return_value=mock_response,
        ) as mock_post,
    ):
        with pytest.raises(MlflowException, match=f"failed with status {status_code}"):
            _invoke_databricks_model(model_name="test-model", prompt="test prompt", num_retries=3)

        mock_post.assert_called_once()
        mock_get_creds.assert_called_once()


def test_invoke_databricks_model_retry_logic_with_transient_errors() -> None:
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
            "mlflow.genai.judges.utils.requests.post",
            side_effect=[error_response, success_response],
        ) as mock_post,
        mock.patch("mlflow.genai.judges.utils.time.sleep") as mock_sleep,
    ):
        result = _invoke_databricks_model(
            model_name="test-model", prompt="test prompt", num_retries=3
        )

        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1
        mock_get_creds.assert_called_once()

    assert result.response == "Success"


def test_invoke_databricks_model_json_decode_error() -> None:
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
            "mlflow.genai.judges.utils.requests.post",
            return_value=mock_response,
        ) as mock_post,
    ):
        with pytest.raises(MlflowException, match="Failed to parse JSON response"):
            _invoke_databricks_model(model_name="test-model", prompt="test prompt", num_retries=0)

        mock_post.assert_called_once()
        mock_get_creds.assert_called_once()


def test_invoke_databricks_model_connection_error_with_retries() -> None:
    mock_creds = mock.Mock()

    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=mock_creds,
        ) as mock_get_creds,
        mock.patch(
            "mlflow.genai.judges.utils.requests.post",
            side_effect=requests.ConnectionError("Connection failed"),
        ) as mock_post,
        mock.patch("mlflow.genai.judges.utils.time.sleep") as mock_sleep,
    ):
        with pytest.raises(
            MlflowException, match="Failed to invoke Databricks model after 3 attempts"
        ):
            _invoke_databricks_model(model_name="test-model", prompt="test prompt", num_retries=2)

        assert mock_post.call_count == 3  # Initial + 2 retries
        assert mock_sleep.call_count == 2
        mock_get_creds.assert_called_once()


def test_record_success_telemetry_with_databricks_agents() -> None:
    from mlflow.genai.judges.utils import _record_judge_model_usage_success_databricks_telemetry

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


def test_record_failure_telemetry_with_databricks_agents() -> None:
    from mlflow.genai.judges.utils import _record_judge_model_usage_failure_databricks_telemetry

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


def test_invoke_judge_model_databricks_success_not_in_databricks() -> None:
    with (
        mock.patch(
            "mlflow.genai.judges.utils._is_in_databricks",
            return_value=False,
        ) as mock_in_db,
        mock.patch(
            "mlflow.genai.judges.utils._invoke_databricks_model",
            return_value=InvokeDatabricksModelOutput(
                response='{"result": "yes", "rationale": "Good response"}',
                request_id="req-123",
                num_prompt_tokens=10,
                num_completion_tokens=5,
            ),
        ) as mock_invoke_db,
    ):
        feedback = invoke_judge_model(
            model_uri="databricks:/test-model",
            prompt="Test prompt",
            assessment_name="test_assessment",
        )

        mock_invoke_db.assert_called_once_with(
            model_name="test-model", prompt="Test prompt", num_retries=10
        )
        mock_in_db.assert_called_once()

    assert feedback.name == "test_assessment"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "Good response"


def test_invoke_judge_model_databricks_success_in_databricks() -> None:
    with (
        mock.patch(
            "mlflow.genai.judges.utils._is_in_databricks",
            return_value=True,
        ) as mock_in_db,
        mock.patch(
            "mlflow.genai.judges.utils._invoke_databricks_model",
            return_value=InvokeDatabricksModelOutput(
                response='{"result": "no", "rationale": "Bad response"}',
                request_id="req-456",
                num_prompt_tokens=15,
                num_completion_tokens=8,
            ),
        ) as mock_invoke_db,
        mock.patch(
            "mlflow.genai.judges.utils._record_judge_model_usage_success_databricks_telemetry"
        ) as mock_success_telemetry,
    ):
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
        mock_invoke_db.assert_called_once()
        mock_in_db.assert_called_once()

    assert feedback.value == CategoricalRating.NO
    assert feedback.rationale == "Bad response"


def test_invoke_judge_model_databricks_failure_in_databricks() -> None:
    with (
        mock.patch(
            "mlflow.genai.judges.utils._is_in_databricks",
            return_value=True,
        ) as mock_in_db,
        mock.patch(
            "mlflow.genai.judges.utils._invoke_databricks_model",
            side_effect=MlflowException("Model invocation failed"),
        ) as mock_invoke_db,
        mock.patch(
            "mlflow.genai.judges.utils._record_judge_model_usage_failure_databricks_telemetry"
        ) as mock_failure_telemetry,
    ):
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
        mock_invoke_db.assert_called_once()
        mock_in_db.assert_called_once()

        # Verify error message contains the traceback
        call_args = mock_failure_telemetry.call_args[1]
        assert "Model invocation failed" in call_args["error_message"]


def test_invoke_judge_model_databricks_telemetry_error_handling() -> None:
    with (
        mock.patch(
            "mlflow.genai.judges.utils._is_in_databricks",
            return_value=True,
        ) as mock_in_db,
        mock.patch(
            "mlflow.genai.judges.utils._invoke_databricks_model",
            return_value=InvokeDatabricksModelOutput(
                response='{"result": "yes", "rationale": "Good"}',
                request_id="req-789",
                num_prompt_tokens=5,
                num_completion_tokens=3,
            ),
        ) as mock_invoke_db,
        mock.patch(
            "mlflow.genai.judges.utils._record_judge_model_usage_success_databricks_telemetry",
            side_effect=Exception("Telemetry failed"),
        ) as mock_success_telemetry,
    ):
        # Should still return feedback despite telemetry failure
        feedback = invoke_judge_model(
            model_uri="databricks:/test-model",
            prompt="Test prompt",
            assessment_name="test_assessment",
        )

        mock_success_telemetry.assert_called_once()
        mock_invoke_db.assert_called_once()
        mock_in_db.assert_called_once()

    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "Good"


# Tests for call_chat_completions function
@pytest.fixture
def mock_databricks_rag_eval():
    """Clean fixture for mocking databricks.rag_eval dependencies."""
    mock_rag_client = mock.MagicMock()
    mock_rag_client.get_chat_completions_result.return_value = AttrDict(
        {"output": "test response", "error_message": None}
    )

    mock_context = mock.MagicMock()
    mock_context.get_context.return_value.build_managed_rag_client.return_value = mock_rag_client
    mock_context.eval_context = lambda func: func  # Pass-through decorator

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
@mock.patch("mlflow.genai.judges.utils._check_databricks_agents_installed")
def test_call_chat_completions_success(
    mock_check, user_prompt, system_prompt, mock_databricks_rag_eval
):
    """Test successful call to call_chat_completions with different prompt combinations."""
    with (
        mock.patch.dict("sys.modules", {"databricks.rag_eval": mock_databricks_rag_eval["module"]}),
        mock.patch("mlflow.genai.judges.utils.VERSION", "1.0.0"),
    ):
        result = call_chat_completions(user_prompt, system_prompt)

        # Verify the client name was set
        mock_databricks_rag_eval[
            "env_vars"
        ].RAG_EVAL_EVAL_SESSION_CLIENT_NAME.set.assert_called_once_with(
            "mlflow-judge-optimizer-v1.0.0"
        )

        # Verify the managed RAG client was called with correct parameters
        mock_databricks_rag_eval["rag_client"].get_chat_completions_result.assert_called_once_with(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        assert result.output == "test response"


@mock.patch("mlflow.genai.judges.utils._check_databricks_agents_installed")
def test_call_chat_completions_client_error(mock_check, mock_databricks_rag_eval):
    """Test call_chat_completions when managed RAG client raises an error."""
    mock_databricks_rag_eval["rag_client"].get_chat_completions_result.side_effect = RuntimeError(
        "RAG client failed"
    )

    with mock.patch.dict(
        "sys.modules", {"databricks.rag_eval": mock_databricks_rag_eval["module"]}
    ):
        with pytest.raises(RuntimeError, match="RAG client failed"):
            call_chat_completions("test prompt", "system prompt")


def test_get_default_optimizer():
    """Test that get_default_optimizer returns a SIMBA optimizer."""
    optimizer = get_default_optimizer()
    assert isinstance(optimizer, SIMBAAlignmentOptimizer)
