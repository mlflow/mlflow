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
from mlflow.genai.judges.utils import CategoricalRating
from mlflow.genai.judges.utils.invocation_utils import (
    _invoke_databricks_structured_output,
    get_chat_completions_with_structured_output,
    invoke_judge_model,
)
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


def test_invoke_judge_model_successful_with_native_provider():
    mock_response = json.dumps({"result": "yes", "rationale": "The response meets all criteria."})

    with mock.patch(
        "mlflow.metrics.genai.model_utils.score_model_on_payload", return_value=mock_response
    ) as mock_score_model_on_payload:
        feedback = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt="Evaluate this response",
            assessment_name="quality_check",
        )

    mock_score_model_on_payload.assert_called_once_with(
        model_uri="openai:/gpt-4",
        payload="Evaluate this response",
        eval_parameters=None,
        extra_headers=None,
        proxy_url=None,
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
    with pytest.raises(MlflowException, match=r"No suitable adapter found"):
        invoke_judge_model(
            model_uri="unsupported:/model", prompt="Test prompt", assessment_name="test"
        )


@pytest.mark.parametrize(
    ("extra_kwargs"),
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


# Tests for _invoke_databricks_structured_output


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

    # Verify schema injection result
    # With existing system message, schema is appended; without, a new system message is added
    expected_message_count = len(input_messages) + (0 if has_existing_system_message else 1)
    assert len(captured_messages) == expected_message_count
    assert captured_messages[0].role == "system"
    assert "You must return your response as JSON matching this schema:" in (
        captured_messages[0].content
    )
    assert '"outputs"' in captured_messages[0].content

    if has_existing_system_message:
        # Verify schema was appended to existing system message
        assert "You are a helpful assistant." in captured_messages[0].content
    else:
        # Verify user message remains unchanged
        assert captured_messages[1].role == "user"
        assert captured_messages[1].content == "Extract the outputs"

    assert isinstance(result, TestSchema)
    assert result.outputs == "test result"


def test_get_chat_completions_with_structured_output_non_databricks():
    class FieldExtraction(BaseModel):
        inputs: str = Field(description="The user's original request")
        outputs: str = Field(description="The system's final response")

    mock_response = '{"inputs": "What is MLflow?", "outputs": "MLflow is a platform"}'

    with mock.patch(
        "mlflow.metrics.genai.model_utils.score_model_on_payload",
        return_value=mock_response,
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
