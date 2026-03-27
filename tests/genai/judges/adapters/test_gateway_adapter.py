import json
from unittest import mock

import pytest

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.base_adapter import AdapterInvocationInput
from mlflow.genai.judges.adapters.gateway_adapter import GatewayAdapter, InvokeOutput
from mlflow.types.llm import ChatMessage


@pytest.fixture
def mock_trace():
    trace_info = TraceInfo(
        trace_id="test-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
    )
    return Trace(info=trace_info, data=None)


# --- is_applicable tests ---


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "mistral"])
def test_native_providers_applicable(provider):
    assert GatewayAdapter.is_applicable(model_uri=f"{provider}:/test-model", prompt="test")


def test_unsupported_provider_not_applicable():
    assert not GatewayAdapter.is_applicable(model_uri="unknown_provider:/test-model", prompt="test")


def test_endpoints_with_string_prompt_applicable():
    assert GatewayAdapter.is_applicable(model_uri="endpoints:/my-endpoint", prompt="test prompt")


def test_endpoints_with_list_prompt_not_applicable():
    assert not GatewayAdapter.is_applicable(
        model_uri="endpoints:/my-endpoint",
        prompt=[ChatMessage(role="user", content="test")],
    )


# --- invoke with trace tests ---


def test_invoke_with_trace_calls_gateway_invocation(mock_trace):
    mock_output = InvokeOutput(
        response=json.dumps({"result": "yes", "rationale": "Looks good"}),
        request_id="req-123",
        num_prompt_tokens=10,
        num_completion_tokens=5,
    )

    adapter = GatewayAdapter()
    input_params = AdapterInvocationInput(
        model_uri="openai:/gpt-4",
        prompt=[ChatMessage(role="user", content="evaluate this")],
        assessment_name="test_metric",
        trace=mock_trace,
    )

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.GatewayAdapter._run_tool_calling_loop",
        return_value=mock_output,
    ) as mock_invoke:
        result = adapter.invoke(input_params)

    mock_invoke.assert_called_once()
    call_kwargs = mock_invoke.call_args[1]
    assert call_kwargs["provider"] == "openai"
    assert call_kwargs["model_name"] == "gpt-4"
    assert call_kwargs["trace"] is mock_trace

    assert result.feedback.name == "test_metric"
    assert result.feedback.value == "yes"
    assert result.feedback.rationale == "Looks good"
    assert result.feedback.trace_id == "test-trace"


def test_invoke_with_trace_string_prompt(mock_trace):
    mock_output = InvokeOutput(
        response=json.dumps({"result": "no", "rationale": "Bad"}),
        request_id=None,
        num_prompt_tokens=None,
        num_completion_tokens=None,
    )

    adapter = GatewayAdapter()
    input_params = AdapterInvocationInput(
        model_uri="openai:/gpt-4",
        prompt="evaluate this string",
        assessment_name="test_metric",
        trace=mock_trace,
    )

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.GatewayAdapter._run_tool_calling_loop",
        return_value=mock_output,
    ) as mock_invoke:
        result = adapter.invoke(input_params)

    # String prompt should be converted to ChatMessage
    mock_invoke.assert_called_once()
    assert result.feedback.value == "no"


# --- invoke without trace tests ---


def test_invoke_without_trace_uses_gateway():
    adapter = GatewayAdapter()
    input_params = AdapterInvocationInput(
        model_uri="openai:/gpt-4",
        prompt=[ChatMessage(role="user", content="test")],
        assessment_name="test_metric",
    )

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter._invoke_via_gateway",
        return_value=json.dumps({"result": "yes", "rationale": "ok"}),
    ) as mock_invoke:
        result = adapter.invoke(input_params)

    mock_invoke.assert_called_once()
    assert result.feedback.value == "yes"


# --- endpoint restriction tests ---


def test_endpoints_rejects_base_url():
    adapter = GatewayAdapter()
    input_params = AdapterInvocationInput(
        model_uri="endpoints:/my-ep",
        prompt="test",
        assessment_name="test_metric",
        base_url="http://proxy:8080",
    )

    with pytest.raises(MlflowException, match="base_url and extra_headers are not supported"):
        adapter.invoke(input_params)
