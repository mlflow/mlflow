from unittest import mock

import litellm
import pytest
from litellm import RetryPolicy
from litellm.types.utils import ModelResponse
from pydantic import BaseModel, Field

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges.adapters.litellm_adapter import (
    _MODEL_RESPONSE_FORMAT_CAPABILITIES,
    _invoke_litellm,
    _remove_oldest_tool_call_pair,
)
from mlflow.types.llm import ChatMessage


@pytest.fixture(autouse=True)
def clear_model_capabilities_cache():
    _MODEL_RESPONSE_FORMAT_CAPABILITIES.clear()


@pytest.fixture
def mock_trace():
    trace_info = TraceInfo(
        trace_id="test-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
    )
    return Trace(info=trace_info, data=None)


def test_invoke_litellm_basic():
    mock_response = ModelResponse(
        choices=[{"message": {"content": '{"result": "yes", "rationale": "Good"}'}}]
    )

    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        result = _invoke_litellm(
            litellm_model="openai/gpt-4",
            messages=[litellm.Message(role="user", content="Test")],
            tools=[],
            num_retries=5,
            response_format=None,
            include_response_format=False,
        )

    assert result == mock_response
    mock_litellm.assert_called_once()
    call_kwargs = mock_litellm.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-4"
    assert call_kwargs["tools"] is None
    assert call_kwargs["tool_choice"] is None
    assert call_kwargs["max_retries"] == 0
    assert call_kwargs["drop_params"] is True
    assert "response_format" not in call_kwargs
    assert "api_base" not in call_kwargs
    assert "api_key" not in call_kwargs


def test_invoke_litellm_with_tools():
    mock_response = ModelResponse(choices=[{"message": {"content": "response"}}])
    tools = [{"name": "test_tool", "description": "A test tool"}]

    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        result = _invoke_litellm(
            litellm_model="openai/gpt-4",
            messages=[litellm.Message(role="user", content="Test")],
            tools=tools,
            num_retries=3,
            response_format=None,
            include_response_format=False,
        )

    assert result == mock_response
    call_kwargs = mock_litellm.call_args.kwargs
    assert call_kwargs["tools"] == tools
    assert call_kwargs["tool_choice"] == "auto"


def test_invoke_litellm_with_response_format():
    class TestSchema(BaseModel):
        result: str = Field(description="The result")

    mock_response = ModelResponse(choices=[{"message": {"content": '{"result": "yes"}'}}])

    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        result = _invoke_litellm(
            litellm_model="openai/gpt-4",
            messages=[litellm.Message(role="user", content="Test")],
            tools=[],
            num_retries=3,
            response_format=TestSchema,
            include_response_format=True,
        )

    assert result == mock_response
    call_kwargs = mock_litellm.call_args.kwargs
    assert "response_format" in call_kwargs
    assert call_kwargs["response_format"] == TestSchema


def test_invoke_litellm_exception_propagation():
    with mock.patch(
        "litellm.completion",
        side_effect=litellm.RateLimitError(
            message="Rate limit exceeded", model="openai/gpt-4", llm_provider="openai"
        ),
    ):
        with pytest.raises(litellm.RateLimitError, match="Rate limit exceeded"):
            _invoke_litellm(
                litellm_model="openai/gpt-4",
                messages=[litellm.Message(role="user", content="Test")],
                tools=[],
                num_retries=3,
                response_format=None,
                include_response_format=False,
            )


def test_invoke_litellm_retry_policy_configured():
    mock_response = ModelResponse(choices=[{"message": {"content": "test"}}])

    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        _invoke_litellm(
            litellm_model="openai/gpt-4",
            messages=[litellm.Message(role="user", content="Test")],
            tools=[],
            num_retries=7,
            response_format=None,
            include_response_format=False,
        )

    call_kwargs = mock_litellm.call_args.kwargs
    retry_policy = call_kwargs["retry_policy"]
    assert isinstance(retry_policy, RetryPolicy)
    assert retry_policy.TimeoutErrorRetries == 7
    assert retry_policy.RateLimitErrorRetries == 7
    assert retry_policy.InternalServerErrorRetries == 7
    assert retry_policy.BadRequestErrorRetries == 0
    assert retry_policy.AuthenticationErrorRetries == 0


def test_invoke_litellm_with_gateway_params():
    mock_response = ModelResponse(choices=[{"message": {"content": '{"result": "yes"}'}}])

    with mock.patch("litellm.completion", return_value=mock_response) as mock_litellm:
        _invoke_litellm(
            litellm_model="my-endpoint",
            messages=[litellm.Message(role="user", content="Test")],
            tools=[],
            num_retries=3,
            response_format=None,
            include_response_format=False,
            api_base="http://localhost:5000/gateway/mlflow/v1/",
            api_key="mlflow-gateway-auth",
        )

    call_kwargs = mock_litellm.call_args.kwargs
    assert call_kwargs["model"] == "my-endpoint"
    assert call_kwargs["api_base"] == "http://localhost:5000/gateway/mlflow/v1/"
    assert call_kwargs["api_key"] == "mlflow-gateway-auth"


def test_invoke_litellm_and_handle_tools_with_context_window_exceeded_direct_provider(mock_trace):
    # For direct providers (non-gateway), use token-counting based pruning
    context_error = litellm.ContextWindowExceededError(
        message="Context window exceeded", model="openai/gpt-4", llm_provider="openai"
    )

    success_response = ModelResponse(
        choices=[{"message": {"content": '{"result": "yes", "rationale": "OK"}'}}]
    )

    with (
        mock.patch(
            "litellm.completion",
            side_effect=[context_error, success_response],
        ) as mock_litellm,
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._prune_messages_exceeding_context_window_length"
        ) as mock_prune,
        mock.patch("litellm.get_max_tokens", return_value=8000),
    ):
        mock_prune.return_value = [litellm.Message(role="user", content="Pruned")]

        from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm_and_handle_tools

        result, cost = _invoke_litellm_and_handle_tools(
            provider="openai",
            model_name="gpt-4",
            messages=[ChatMessage(role="user", content="Very long message" * 100)],
            trace=None,
            num_retries=3,
        )

    assert mock_litellm.call_count == 2
    mock_prune.assert_called_once()
    assert result == '{"result": "yes", "rationale": "OK"}'
    assert cost is None


def test_invoke_litellm_and_handle_tools_with_context_window_exceeded_gateway_provider():
    # For gateway provider, use DSPy-style reactive truncation
    context_error = litellm.ContextWindowExceededError(
        message="Context window exceeded", model="my-endpoint", llm_provider="openai"
    )

    success_response = ModelResponse(
        choices=[{"message": {"content": '{"result": "yes", "rationale": "OK"}'}}]
    )

    with (
        mock.patch(
            "litellm.completion",
            side_effect=[context_error, success_response],
        ) as mock_litellm,
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._remove_oldest_tool_call_pair"
        ) as mock_truncate,
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter.get_tracking_uri",
            return_value="http://localhost:5000",
        ),
    ):
        mock_truncate.return_value = [litellm.Message(role="user", content="Truncated")]

        from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm_and_handle_tools

        result, cost = _invoke_litellm_and_handle_tools(
            provider="gateway",
            model_name="my-endpoint",
            messages=[ChatMessage(role="user", content="Very long message" * 100)],
            trace=None,
            num_retries=3,
        )

    assert mock_litellm.call_count == 2
    mock_truncate.assert_called_once()
    assert result == '{"result": "yes", "rationale": "OK"}'
    assert cost is None


def test_invoke_litellm_and_handle_tools_gateway_context_window_no_tool_calls_to_truncate():
    # For gateway provider, when there are no tool calls to truncate, raise an error
    context_error = litellm.ContextWindowExceededError(
        message="Context window exceeded", model="my-endpoint", llm_provider="openai"
    )

    with (
        mock.patch("litellm.completion", side_effect=context_error),
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._remove_oldest_tool_call_pair",
            return_value=None,  # No tool calls to truncate
        ),
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter.get_tracking_uri",
            return_value="http://localhost:5000",
        ),
    ):
        from mlflow.exceptions import MlflowException
        from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm_and_handle_tools

        with pytest.raises(MlflowException, match="no tool calls to truncate"):
            _invoke_litellm_and_handle_tools(
                provider="gateway",
                model_name="my-endpoint",
                messages=[ChatMessage(role="user", content="Very long message")],
                trace=None,
                num_retries=3,
            )


def test_invoke_litellm_and_handle_tools_integration(mock_trace):
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
    tool_call_response._hidden_params = {"response_cost": 0.05}

    final_response = ModelResponse(
        choices=[{"message": {"content": '{"result": "yes", "rationale": "Good"}'}}]
    )
    final_response._hidden_params = {"response_cost": 0.15}

    with (
        mock.patch(
            "litellm.completion",
            side_effect=[tool_call_response, final_response],
        ) as mock_litellm,
        mock.patch("mlflow.genai.judges.tools.list_judge_tools") as mock_list_tools,
        mock.patch("mlflow.genai.judges.tools.registry._judge_tool_registry.invoke") as mock_invoke,
    ):
        mock_tool = mock.Mock()
        mock_tool.get_definition.return_value.to_dict.return_value = {"name": "test_tool"}
        mock_list_tools.return_value = [mock_tool]
        mock_invoke.return_value = {"trace_data": "some data"}

        from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm_and_handle_tools

        result, cost = _invoke_litellm_and_handle_tools(
            provider="openai",
            model_name="gpt-4",
            messages=[ChatMessage(role="user", content="Test with trace")],
            trace=mock_trace,
            num_retries=3,
        )

    assert mock_litellm.call_count == 2
    mock_invoke.assert_called_once()
    assert result == '{"result": "yes", "rationale": "Good"}'
    assert cost == pytest.approx(0.20)

    second_call_messages = mock_litellm.call_args_list[1].kwargs["messages"]
    assert len(second_call_messages) == 3
    assert second_call_messages[1].role == "assistant"
    assert second_call_messages[2].role == "tool"
    assert "trace_data" in second_call_messages[2].content


def test_gateway_provider_integration():
    mock_response = ModelResponse(
        choices=[{"message": {"content": '{"result": "yes", "rationale": "Good"}'}}]
    )

    with (
        mock.patch("litellm.completion", return_value=mock_response) as mock_litellm,
        mock.patch("mlflow.genai.judges.adapters.litellm_adapter.get_tracking_uri") as mock_get_uri,
    ):
        mock_get_uri.return_value = "http://localhost:5000"

        from mlflow.genai.judges.adapters.litellm_adapter import (
            _invoke_litellm_and_handle_tools,
        )

        result, cost = _invoke_litellm_and_handle_tools(
            provider="gateway",
            model_name="my-endpoint",
            messages=[ChatMessage(role="user", content="Test")],
            trace=None,
            num_retries=3,
        )

    assert result == '{"result": "yes", "rationale": "Good"}'

    call_kwargs = mock_litellm.call_args.kwargs
    assert call_kwargs["model"] == "openai/my-endpoint"
    assert call_kwargs["api_base"] == "http://localhost:5000/gateway/mlflow/v1/"
    assert call_kwargs["api_key"] == "mlflow-gateway-auth"


def test_gateway_provider_requires_http_tracking_uri():
    from mlflow.exceptions import MlflowException
    from mlflow.genai.judges.adapters.litellm_adapter import _invoke_litellm_and_handle_tools

    with mock.patch(
        "mlflow.genai.judges.adapters.litellm_adapter.get_tracking_uri", return_value="databricks"
    ):
        with pytest.raises(MlflowException, match="Gateway provider requires an HTTP"):
            _invoke_litellm_and_handle_tools(
                provider="gateway",
                model_name="my-endpoint",
                messages=[ChatMessage(role="user", content="Test")],
                trace=None,
                num_retries=3,
            )


def test_remove_oldest_tool_call_pair_removes_oldest():
    messages = [
        litellm.Message(role="user", content="Hello"),
        litellm.Message(
            role="assistant",
            content=None,
            tool_calls=[{"id": "call_1", "function": {"name": "tool1", "arguments": "{}"}}],
        ),
        litellm.Message(role="tool", content="Result 1", tool_call_id="call_1"),
        litellm.Message(
            role="assistant",
            content=None,
            tool_calls=[{"id": "call_2", "function": {"name": "tool2", "arguments": "{}"}}],
        ),
        litellm.Message(role="tool", content="Result 2", tool_call_id="call_2"),
    ]

    result = _remove_oldest_tool_call_pair(messages)

    assert result is not None
    assert len(result) == 3
    assert result[0].role == "user"
    assert result[1].role == "assistant"
    assert result[1].tool_calls[0]["id"] == "call_2"
    assert result[2].role == "tool"
    assert result[2].tool_call_id == "call_2"


def test_remove_oldest_tool_call_pair_returns_none_when_no_tool_calls():
    messages = [
        litellm.Message(role="user", content="Hello"),
        litellm.Message(role="assistant", content="Hi there!"),
    ]

    result = _remove_oldest_tool_call_pair(messages)

    assert result is None


def test_remove_oldest_tool_call_pair_handles_multiple_tool_calls_in_single_message():
    messages = [
        litellm.Message(role="user", content="Hello"),
        litellm.Message(
            role="assistant",
            content=None,
            tool_calls=[
                {"id": "call_1", "function": {"name": "tool1", "arguments": "{}"}},
                {"id": "call_2", "function": {"name": "tool2", "arguments": "{}"}},
            ],
        ),
        litellm.Message(role="tool", content="Result 1", tool_call_id="call_1"),
        litellm.Message(role="tool", content="Result 2", tool_call_id="call_2"),
    ]

    result = _remove_oldest_tool_call_pair(messages)

    assert result is not None
    assert len(result) == 1
    assert result[0].role == "user"


def test_remove_oldest_tool_call_pair_preserves_non_tool_messages():
    messages = [
        litellm.Message(role="system", content="You are helpful"),
        litellm.Message(role="user", content="Hello"),
        litellm.Message(
            role="assistant",
            content=None,
            tool_calls=[{"id": "call_1", "function": {"name": "tool1", "arguments": "{}"}}],
        ),
        litellm.Message(role="tool", content="Result", tool_call_id="call_1"),
        litellm.Message(role="user", content="Thanks"),
    ]

    result = _remove_oldest_tool_call_pair(messages)

    assert result is not None
    assert len(result) == 3
    assert result[0].role == "system"
    assert result[1].role == "user"
    assert result[1].content == "Hello"
    assert result[2].role == "user"
    assert result[2].content == "Thanks"
