import json
from typing import Any
from unittest import mock

import pytest
from litellm.types.utils import ModelResponse

from mlflow.entities.assessment import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.utils import CategoricalRating, invoke_judge_model
from mlflow.types.llm import ToolCall


# Utility functions for testing
def create_mock_response(
    result: str = "yes", rationale: str = "The response meets all criteria."
) -> ModelResponse:
    """Create a mock ModelResponse with the given result and rationale."""
    content = json.dumps({"result": result, "rationale": rationale})
    return ModelResponse(choices=[{"message": {"content": content}}])


def create_mock_tool_response(
    tool_calls: list[dict[str, Any]] | None = None, content: str | None = None
) -> ModelResponse:
    """Create a mock ModelResponse with tool calls or content."""
    message = {"content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return ModelResponse(choices=[{"message": message}])


def create_test_trace(trace_id: str = "test-trace", state: TraceState = TraceState.OK) -> Trace:
    """Create a test Trace object with the given parameters."""
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=state,
    )
    return Trace(info=trace_info, data=None)


def create_mock_tool(name: str = "get_trace_info", description: str | None = None) -> mock.Mock:
    """Create a mock tool with the given name and description."""
    tool = mock.Mock()
    tool.name = name
    tool_def = {"name": name}
    if description:
        tool_def["description"] = description
    tool.get_definition.return_value.to_dict.return_value = tool_def
    return tool


@pytest.mark.parametrize("num_retries", [None, 3])
def test_invoke_judge_model_successful_with_litellm(num_retries):
    mock_response = create_mock_response()

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


def test_invoke_judge_model_with_trace_passes_tools():
    trace = create_test_trace()
    mock_response = create_mock_response()

    with (
        mock.patch("litellm.completion", return_value=mock_response) as mock_litellm,
        mock.patch("mlflow.genai.judges.tools.list_judge_tools") as mock_list_tools,
    ):
        # Mock some tools being available
        mock_tool1 = create_mock_tool("get_trace_info", "Get trace info")
        mock_tool2 = create_mock_tool("list_spans", "List spans")
        mock_list_tools.return_value = [mock_tool1, mock_tool2]

        invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check",
            trace=trace,
        )

    # Verify tools were passed to litellm completion
    mock_litellm.assert_called_once()
    call_kwargs = mock_litellm.call_args.kwargs
    assert call_kwargs["tools"] == [
        {"name": "get_trace_info", "description": "Get trace info"},
        {"name": "list_spans", "description": "List spans"},
    ]
    assert call_kwargs["tool_choice"] == "auto"


def test_invoke_judge_model_tool_calling_loop():
    trace = create_test_trace()

    # First call: model requests tool call
    mock_tool_call_response = create_mock_tool_response(
        tool_calls=[{"id": "call_123", "function": {"name": "get_trace_info", "arguments": "{}"}}]
    )

    # Second call: model provides final answer
    mock_final_response = create_mock_response(rationale="The trace looks good.")

    with (
        mock.patch(
            "litellm.completion", side_effect=[mock_tool_call_response, mock_final_response]
        ) as mock_litellm,
        mock.patch("mlflow.genai.judges.tools.list_judge_tools") as mock_list_tools,
        mock.patch(
            "mlflow.genai.judges.tools.registry._judge_tool_registry.invoke"
        ) as mock_invoke_tool,
    ):
        mock_tool = create_mock_tool("get_trace_info")
        mock_list_tools.return_value = [mock_tool]

        mock_invoke_tool.return_value = {"trace_id": "test-trace", "state": "OK"}

        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check",
            trace=trace,
        )

    # Verify litellm.completion was called twice (tool call + final response)
    assert mock_litellm.call_count == 2

    # Verify tool was invoked
    mock_invoke_tool.assert_called_once()
    tool_call_arg = mock_invoke_tool.call_args[0][0]
    assert isinstance(tool_call_arg, ToolCall)
    assert tool_call_arg.function.name == "get_trace_info"

    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The trace looks good."
