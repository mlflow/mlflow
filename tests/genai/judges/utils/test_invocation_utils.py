"""Tests for invoke_judge_model and get_chat_completions_with_structured_output.

These tests verify routing (which adapter is selected), payload passthrough,
and output parsing. Adapter internals (RetryPolicy, litellm.Message types,
cost tracking, response_format caching) are tested in the adapter-specific
test files (test_litellm_adapter.py, test_gateway_adapter.py).
"""

import json
from unittest import mock

import pytest
from pydantic import BaseModel, Field

from mlflow.entities.assessment import AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.adapters.gateway_adapter import InvokeOutput
from mlflow.genai.judges.utils.invocation_utils import (
    _invoke_databricks_structured_output,
    get_chat_completions_with_structured_output,
    invoke_judge_model,
)
from mlflow.types.llm import ChatMessage

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_MOCK_JSON = json.dumps({"result": "yes", "rationale": "The response meets all criteria."})


@pytest.fixture
def mock_trace():
    trace_info = TraceInfo(
        trace_id="test-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        state=TraceState.OK,
    )
    return Trace(info=trace_info, data=None)


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


def test_invoke_judge_model_routes_to_gateway_for_openai():
    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.score_model_on_payload",
        return_value=_MOCK_JSON,
    ) as mock_score:
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check",
        )

    mock_score.assert_called_once()
    assert mock_score.call_args.kwargs["model_uri"] == "openai:/gpt-4"
    assert feedback.name == "quality_check"
    assert feedback.value == "yes"
    assert feedback.source.source_type == AssessmentSourceType.LLM_JUDGE
    assert feedback.source.source_id == "openai:/gpt-4"


def test_invoke_judge_model_routes_databricks_uri_to_gateway():
    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.score_model_on_payload",
        return_value=_MOCK_JSON,
    ) as mock_score:
        feedback = invoke_judge_model(
            model_uri="databricks:/test-model",
            prompt="Test prompt",
            assessment_name="test_assessment",
        )

    mock_score.assert_called_once()
    assert mock_score.call_args.kwargs["model_uri"] == "databricks:/test-model"
    assert feedback.name == "test_assessment"
    assert feedback.value == "yes"
    assert feedback.source.source_id == "databricks:/test-model"


def test_invoke_judge_model_routes_endpoints_uri_to_gateway():
    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter._invoke_via_gateway",
        return_value=_MOCK_JSON,
    ) as mock_invoke:
        feedback = invoke_judge_model(
            model_uri="endpoints:/my-endpoint",
            prompt="Test prompt",
            assessment_name="test_assessment",
        )

    mock_invoke.assert_called_once()
    assert mock_invoke.call_args[0][0] == "endpoints:/my-endpoint"
    assert feedback.name == "test_assessment"
    assert feedback.value == "yes"
    assert feedback.source.source_id == "endpoints:/my-endpoint"


def test_invoke_judge_model_with_unsupported_provider():
    with (
        mock.patch(
            "mlflow.genai.judges.adapters.litellm_adapter._is_litellm_available",
            return_value=False,
        ),
        pytest.raises(MlflowException, match=r"No suitable adapter found"),
    ):
        invoke_judge_model(
            model_uri="unsupported:/model", prompt="Test prompt", assessment_name="test"
        )


# ---------------------------------------------------------------------------
# String prompt path (score_model_on_payload)
# ---------------------------------------------------------------------------


def test_invoke_judge_model_string_prompt():
    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.score_model_on_payload",
        return_value=_MOCK_JSON,
    ) as mock_score:
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check",
        )

    mock_score.assert_called_once_with(
        model_uri="openai:/gpt-4",
        payload="Evaluate this response",
        eval_parameters=None,
        extra_headers=None,
        proxy_url=None,
        endpoint_type="llm/v1/chat",
    )
    assert feedback.name == "quality_check"
    assert feedback.value == "yes"
    assert feedback.rationale == "The response meets all criteria."
    assert feedback.trace_id is None


@pytest.mark.parametrize(
    "inference_params",
    [
        None,
        {"temperature": 0},
        {"temperature": 0.5, "max_tokens": 100},
    ],
)
def test_invoke_judge_model_inference_params_passed_through(inference_params):
    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.score_model_on_payload",
        return_value=_MOCK_JSON,
    ) as mock_score:
        invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this",
            assessment_name="test",
            inference_params=inference_params,
        )

    assert mock_score.call_args.kwargs["eval_parameters"] == inference_params


# ---------------------------------------------------------------------------
# Chat message prompt path (_call_llm_provider_api)
# ---------------------------------------------------------------------------


def test_invoke_judge_model_with_chat_messages():
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant"),
        ChatMessage(role="user", content="Evaluate this response"),
    ]

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter._call_llm_provider_api",
        return_value=_MOCK_JSON,
    ) as mock_call:
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt=messages,
            assessment_name="quality_check",
        )

    mock_call.assert_called_once()
    call_kwargs = mock_call.call_args
    assert call_kwargs[0][0] == "openai"  # provider
    assert call_kwargs[0][1] == "gpt-4"  # model_name
    assert feedback.name == "quality_check"
    assert feedback.value == "yes"


# ---------------------------------------------------------------------------
# Trace / tool-calling path (send_chat_request)
# ---------------------------------------------------------------------------


def test_invoke_judge_model_with_trace(mock_trace):
    mock_output = InvokeOutput(
        response=_MOCK_JSON,
        request_id="req-123",
        num_prompt_tokens=10,
        num_completion_tokens=5,
    )

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.GatewayAdapter._invoke_and_handle_tools",
        return_value=mock_output,
    ):
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Test prompt",
            assessment_name="test",
            trace=mock_trace,
        )

    assert feedback.value == "yes"
    assert feedback.trace_id == "test-trace"


def test_invoke_judge_model_with_trace_uses_tool_calling(mock_trace):
    mock_output = InvokeOutput(
        response=_MOCK_JSON,
        request_id="req-456",
        num_prompt_tokens=15,
        num_completion_tokens=8,
    )

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.GatewayAdapter._invoke_and_handle_tools",
        return_value=mock_output,
    ) as mock_invoke:
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check",
            trace=mock_trace,
        )

    mock_invoke.assert_called_once()
    call_kwargs = mock_invoke.call_args.kwargs
    assert call_kwargs["provider"] == "openai"
    assert call_kwargs["model_name"] == "gpt-4"
    assert call_kwargs["trace"] is mock_trace
    assert feedback.trace_id == "test-trace"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invoke_judge_model_invalid_json_response():
    with (
        mock.patch(
            "mlflow.genai.judges.adapters.gateway_adapter.score_model_on_payload",
            return_value="This is not valid JSON",
        ),
        pytest.raises(MlflowException, match=r"Failed to parse"),
    ):
        invoke_judge_model(model_uri="openai:/gpt-4", prompt="Test prompt", assessment_name="test")


# ---------------------------------------------------------------------------
# Endpoint restrictions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {"base_url": "http://proxy:8080"},
        {"extra_headers": {"Authorization": "Bearer token"}},
        {"base_url": "http://proxy:8080", "extra_headers": {"Authorization": "Bearer token"}},
    ],
)
def test_invoke_judge_model_base_url_and_extra_headers_not_supported_for_endpoints(extra_kwargs):
    with pytest.raises(MlflowException, match="not supported for deployment endpoints"):
        invoke_judge_model(
            model_uri="endpoints:/my-endpoint",
            prompt="Evaluate this",
            assessment_name="test",
            **extra_kwargs,
        )


# ---------------------------------------------------------------------------
# get_chat_completions_with_structured_output
# ---------------------------------------------------------------------------


def test_get_chat_completions_with_structured_output():
    class FieldExtraction(BaseModel):
        inputs: str = Field(description="The user's original request")
        outputs: str = Field(description="The system's final response")

    mock_output = InvokeOutput(
        response='{"inputs": "What is MLflow?", "outputs": "MLflow is a platform"}',
        request_id="req-1",
        num_prompt_tokens=10,
        num_completion_tokens=5,
    )

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.GatewayAdapter._invoke_and_handle_tools",
        return_value=mock_output,
    ):
        result = get_chat_completions_with_structured_output(
            model_uri="openai:/gpt-4",
            messages=[
                ChatMessage(role="system", content="Extract fields"),
                ChatMessage(role="user", content="Find inputs and outputs"),
            ],
            output_schema=FieldExtraction,
        )

    assert isinstance(result, FieldExtraction)
    assert result.inputs == "What is MLflow?"
    assert result.outputs == "MLflow is a platform"


def test_get_chat_completions_with_structured_output_with_trace(mock_trace):
    class FieldExtraction(BaseModel):
        inputs: str = Field(description="The user's original request")
        outputs: str = Field(description="The system's final response")

    mock_output = InvokeOutput(
        response='{"inputs": "question from trace", "outputs": "answer from trace"}',
        request_id="req-2",
        num_prompt_tokens=15,
        num_completion_tokens=8,
    )

    with mock.patch(
        "mlflow.genai.judges.adapters.gateway_adapter.GatewayAdapter._invoke_and_handle_tools",
        return_value=mock_output,
    ):
        result = get_chat_completions_with_structured_output(
            model_uri="openai:/gpt-4",
            messages=[
                ChatMessage(role="system", content="Extract fields"),
                ChatMessage(role="user", content="Find inputs and outputs"),
            ],
            output_schema=FieldExtraction,
            trace=mock_trace,
        )

    assert isinstance(result, FieldExtraction)
    assert result.inputs == "question from trace"
    assert result.outputs == "answer from trace"


# ---------------------------------------------------------------------------
# _invoke_databricks_structured_output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_messages", "mock_response", "has_existing_system_message"),
    [
        pytest.param(
            [
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="Extract the outputs"),
            ],
            '{"outputs": "test result"}',
            True,
            id="with_existing_system_message",
        ),
        pytest.param(
            [
                ChatMessage(role="user", content="Extract the outputs"),
            ],
            '{"outputs": "test result"}',
            False,
            id="without_system_message",
        ),
    ],
)
def test_structured_output_schema_injection(
    input_messages, mock_response, has_existing_system_message
):
    class TestSchema(BaseModel):
        outputs: str = Field(description="The outputs")

    captured_messages = []

    def mock_loop(messages, trace, on_final_answer):
        captured_messages.extend(messages)
        return on_final_answer(mock_response)

    with mock.patch(
        "mlflow.genai.judges.utils.invocation_utils._run_databricks_agentic_loop",
        side_effect=mock_loop,
    ):
        result = _invoke_databricks_structured_output(
            messages=input_messages,
            output_schema=TestSchema,
            trace=None,
        )

    expected_message_count = len(input_messages) + (0 if has_existing_system_message else 1)
    assert len(captured_messages) == expected_message_count
    assert captured_messages[0].role == "system"
    assert "You must return your response as JSON matching this schema:" in (
        captured_messages[0].content
    )
    assert '"outputs"' in captured_messages[0].content

    if has_existing_system_message:
        assert "You are a helpful assistant." in captured_messages[0].content
    else:
        assert captured_messages[1].role == "user"
        assert captured_messages[1].content == "Extract the outputs"

    assert isinstance(result, TestSchema)
    assert result.outputs == "test result"
