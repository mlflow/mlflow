import json
from unittest import mock

import litellm
import pytest
from litellm.types.utils import ModelResponse

from mlflow.entities.assessment import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.base import Judge
from mlflow.genai.judges.utils import (
    _MODEL_RESPONSE_FORMAT_CAPABILITIES,
    CategoricalRating,
    add_output_format_instructions,
    invoke_judge_model,
)
from mlflow.types.llm import ChatMessage, ToolCall


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

    mock_litellm.assert_called_once_with(
        model="openai/gpt-4",
        messages=[{"role": "user", "content": "Evaluate this response"}],
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
    assert messages_arg[0] == {"role": "system", "content": "You are a helpful assistant"}
    assert messages_arg[1] == {"role": "user", "content": "Evaluate this response"}

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
