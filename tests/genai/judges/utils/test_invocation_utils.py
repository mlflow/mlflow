import json
from unittest import mock

import litellm
import pytest
from litellm import RetryPolicy
from litellm.types.utils import ModelResponse
from pydantic import BaseModel

from mlflow.entities.assessment import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.optimizers.simba import SIMBAAlignmentOptimizer
from mlflow.genai.judges.utils import CategoricalRating, get_default_optimizer
from mlflow.genai.judges.utils.invocation_utils import invoke_judge_model
from mlflow.tracing.constant import AssessmentMetadataKey
from mlflow.types.llm import ChatMessage


@pytest.fixture
def mock_response():
    content = json.dumps({"result": "yes", "rationale": "The response meets all criteria."})
    response = ModelResponse(choices=[{"message": {"content": content}}])
    response._hidden_params = {"response_cost": 0.123}
    return response


@pytest.fixture
def mock_tool_response():
    tool_calls = [{"id": "call_123", "function": {"name": "get_trace_info", "arguments": "{}"}}]
    response = ModelResponse(choices=[{"message": {"tool_calls": tool_calls, "content": None}}])
    response._hidden_params = {"response_cost": 0.05}
    return response


@pytest.fixture
def mock_trace():
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

    call_kwargs = mock_litellm.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-4"
    assert call_kwargs["tools"] is None
    assert call_kwargs["tool_choice"] is None
    assert call_kwargs["retry_policy"] == expected_retry_policy
    assert call_kwargs["retry_strategy"] == "exponential_backoff_retry"
    assert call_kwargs["max_retries"] == 0
    assert call_kwargs["drop_params"] is True

    response_format = call_kwargs["response_format"]
    assert issubclass(response_format, BaseModel)
    assert "result" in response_format.model_fields
    assert "rationale" in response_format.model_fields

    assert feedback.name == "quality_check"
    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The response meets all criteria."
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4"
    assert feedback.trace_id is None
    assert feedback.metadata is not None
    assert feedback.metadata[AssessmentMetadataKey.JUDGE_COST] == pytest.approx(0.123)


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
    assert feedback.trace_id is None


def test_invoke_judge_model_successful_with_native_provider():
    mock_response = json.dumps({"result": "yes", "rationale": "The response meets all criteria."})

    with (
        mock.patch(
            "mlflow.genai.judges.utils.invocation_utils._is_litellm_available", return_value=False
        ),
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
    assert feedback.trace_id is None
    assert feedback.metadata is None


def test_invoke_judge_model_with_unsupported_provider():
    with pytest.raises(MlflowException, match=r"LiteLLM is required for using 'unsupported' LLM"):
        with mock.patch(
            "mlflow.genai.judges.utils.invocation_utils._is_litellm_available", return_value=False
        ):
            invoke_judge_model(
                model_uri="unsupported:/model", prompt="Test prompt", assessment_name="test"
            )


def test_invoke_judge_model_with_trace_requires_litellm(mock_trace):
    with pytest.raises(MlflowException, match=r"LiteLLM is required for using traces with judges"):
        with mock.patch(
            "mlflow.genai.judges.utils.invocation_utils._is_litellm_available", return_value=False
        ):
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

        feedback = invoke_judge_model(
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
    assert feedback.trace_id == "test-trace"


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
        ],
    )
    mock_tool_call_response._hidden_params = {"response_cost": 0.05}

    # Second call: model provides final answer
    mock_final_response = ModelResponse(
        choices=[
            {
                "message": {
                    "content": json.dumps({"result": "yes", "rationale": "The trace looks good."})
                }
            }
        ],
    )
    mock_final_response._hidden_params = {"response_cost": 0.15}

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
    from mlflow.types.llm import ToolCall

    assert isinstance(tool_call_arg, ToolCall)
    assert tool_call_arg.function.name == "get_trace_info"

    assert feedback.value == CategoricalRating.YES
    assert feedback.rationale == "The trace looks good."
    assert feedback.trace_id == "test-trace"
    assert feedback.metadata is not None
    assert feedback.metadata[AssessmentMetadataKey.JUDGE_COST] == pytest.approx(0.20)


def test_get_default_optimizer():
    optimizer = get_default_optimizer()
    assert isinstance(optimizer, SIMBAAlignmentOptimizer)


@pytest.mark.parametrize("env_var_value", ["3", None])
def test_invoke_judge_model_completion_iteration_limit(mock_trace, monkeypatch, env_var_value):
    if env_var_value is not None:
        monkeypatch.setenv("MLFLOW_JUDGE_MAX_ITERATIONS", env_var_value)
        expected_limit = int(env_var_value)
    else:
        monkeypatch.delenv("MLFLOW_JUDGE_MAX_ITERATIONS", raising=False)
        expected_limit = 30
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

    with (
        mock.patch("litellm.completion", return_value=mock_tool_call_response) as mock_litellm,
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

        with pytest.raises(
            MlflowException, match="Completion iteration limit.*exceeded"
        ) as exc_info:
            invoke_judge_model(
                model_uri="openai:/gpt-4",
                prompt="Evaluate this response",
                assessment_name="quality_check",
                trace=mock_trace,
            )

        error_msg = str(exc_info.value)
        assert f"Completion iteration limit of {expected_limit} exceeded" in error_msg
        assert "model is not powerful enough" in error_msg
        assert mock_litellm.call_count == expected_limit


def test_invoke_judge_model_with_custom_response_format():
    class ResponseFormat(BaseModel):
        result: int
        rationale: str

    # Mock litellm to return structured JSON
    mock_response = ModelResponse(
        choices=[
            {
                "message": {
                    "content": '{"result": 8, "rationale": "High quality"}',
                    "tool_calls": None,
                }
            }
        ]
    )

    with mock.patch("litellm.completion", return_value=mock_response) as mock_completion:
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt=[ChatMessage(role="user", content="Evaluate this")],
            assessment_name="test_judge",
            response_format=ResponseFormat,
        )

    # Verify the result was correctly parsed and converted to dict
    assert feedback.name == "test_judge"
    assert feedback.value == 8
    assert feedback.rationale == "High quality"

    # Verify response_format was passed to litellm.completion
    call_kwargs = mock_completion.call_args.kwargs
    assert "response_format" in call_kwargs
    assert call_kwargs["response_format"] == ResponseFormat
