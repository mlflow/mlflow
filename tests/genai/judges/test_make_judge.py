import json
import sys
import types
import typing
from dataclasses import asdict
from typing import Any, Literal
from unittest import mock
from unittest.mock import patch

import litellm
import pandas as pd
import pydantic
import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow
import mlflow.genai
import mlflow.genai.judges.instructions_judge
from mlflow.entities import Span, SpanType, Trace, TraceData, TraceInfo
from mlflow.entities.assessment import (
    AssessmentSource,
    AssessmentSourceType,
    Expectation,
    Feedback,
)
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai import make_judge
from mlflow.genai.judges.constants import _RESULT_FIELD_DESCRIPTION
from mlflow.genai.judges.instructions_judge import InstructionsJudge
from mlflow.genai.judges.instructions_judge.constants import JUDGE_BASE_PROMPT
from mlflow.genai.judges.utils import _NATIVE_PROVIDERS, validate_judge_model
from mlflow.genai.scorers.base import Scorer, ScorerKind, SerializedScorer
from mlflow.genai.scorers.registry import _get_scorer_store
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.utils import build_otel_context
from mlflow.types.llm import ChatMessage


@pytest.fixture
def mock_databricks_rag_eval(monkeypatch):
    """Mock the databricks.rag_eval module structure for testing databricks judges.

    NB: The databricks judge uses the following call chain:
    databricks.rag_eval.context.get_context().build_managed_rag_client().get_chat_completions_result()
    This fixture mocks the entire module hierarchy to test without actual databricks dependencies.
    """
    # Mock the entire databricks.agents.evals module hierarchy
    mock_evals_module = types.ModuleType("databricks.agents.evals")
    monkeypatch.setitem(sys.modules, "databricks.agents.evals", mock_evals_module)

    mock_judges_module = types.ModuleType("databricks.agents.evals.judges")
    monkeypatch.setitem(sys.modules, "databricks.agents.evals.judges", mock_judges_module)

    class MockLLMResult:
        def __init__(self, output_data=None):
            data = output_data or {"result": True, "rationale": "Test passed"}
            self.output = json.dumps(data)
            self.output_json = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": json.dumps(data)}}]}
            )
            self.error_message = None

    class MockManagedRAGClient:
        def __init__(self, expected_content=None, response_data=None):
            self.expected_content = expected_content
            self.response_data = response_data

        def get_chat_completions_result(self, user_prompt, system_prompt, use_case=None, **kwargs):
            # Check that expected content is in either user or system prompt
            if self.expected_content:
                combined = (system_prompt or "") + " " + user_prompt
                assert self.expected_content in combined
            return MockLLMResult(self.response_data)

    class MockContext:
        def __init__(self, expected_content=None, response_data=None):
            self.expected_content = expected_content
            self.response_data = response_data

        def build_managed_rag_client(self):
            return MockManagedRAGClient(self.expected_content, self.response_data)

    mock_rag_eval = types.ModuleType("databricks.rag_eval")
    monkeypatch.setitem(sys.modules, "databricks.rag_eval", mock_rag_eval)

    mock_context_module = types.ModuleType("databricks.rag_eval.context")

    mock_context_module.MockContext = MockContext
    mock_context_module.get_context = lambda: MockContext()
    mock_context_module.eval_context = lambda func: func  # Pass-through decorator
    mock_context_module.context = mock_context_module  # Self-reference for import

    mock_rag_eval.context = mock_context_module
    monkeypatch.setitem(sys.modules, "databricks.rag_eval.context", mock_context_module)

    # Mock env_vars module needed by call_chat_completions
    mock_env_vars_module = types.ModuleType("databricks.rag_eval.env_vars")

    class MockEnvVar:
        def set(self, value):
            pass

    mock_env_vars_module.RAG_EVAL_EVAL_SESSION_CLIENT_NAME = MockEnvVar()
    mock_rag_eval.env_vars = mock_env_vars_module
    monkeypatch.setitem(sys.modules, "databricks.rag_eval.env_vars", mock_env_vars_module)

    return mock_context_module


@pytest.fixture
def mock_invoke_judge_model(monkeypatch):
    """Unified fixture that captures all invocation details and supports different use cases."""
    calls = []
    captured_args = {}

    def _mock(
        model_uri,
        prompt,
        assessment_name,
        trace=None,
        num_retries=10,
        response_format=None,
        use_case=None,
        inference_params=None,
    ):
        # Store call details in list format (for backward compatibility)
        calls.append((model_uri, prompt, assessment_name))

        # Store latest call details in dict format
        captured_args.update(
            {
                "model_uri": model_uri,
                "prompt": prompt,
                "assessment_name": assessment_name,
                "trace": trace,
                "num_retries": num_retries,
                "response_format": response_format,
                "use_case": use_case,
                "inference_params": inference_params,
            }
        )

        # Return appropriate Feedback based on whether trace is provided
        if trace is not None:
            return Feedback(name=assessment_name, value=True, rationale="Trace analyzed")
        else:
            return Feedback(name=assessment_name, value=True, rationale="The response is formal")

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", _mock)

    # Attach convenience properties for different usage patterns
    _mock.calls = calls
    _mock.captured_args = captured_args
    _mock.reset_mock = lambda: (calls.clear(), captured_args.clear())

    return _mock


def create_test_span(
    span_id=1,
    parent_id=None,
    name="test_span",
    inputs=None,
    outputs=None,
    span_type=SpanType.UNKNOWN,
):
    otel_span = OTelReadableSpan(
        name=name,
        context=build_otel_context(trace_id=123456789, span_id=span_id),
        parent=build_otel_context(trace_id=123456789, span_id=parent_id) if parent_id else None,
        start_time=100000000,
        end_time=200000000,
        attributes={
            "mlflow.spanInputs": json.dumps(inputs or {}),
            "mlflow.spanOutputs": json.dumps(outputs or {}),
            "mlflow.spanType": json.dumps(span_type),
        },
    )
    return Span(otel_span)


@pytest.fixture
def mock_trace():
    trace_info = TraceInfo(
        trace_id="test-trace-123",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        execution_duration=1000,
        state=TraceState.OK,
        trace_metadata={
            "mlflow.trace_schema.version": "2",
            "mlflow.traceInputs": json.dumps({"question": "What is MLflow?"}),
            "mlflow.traceOutputs": json.dumps(
                {"answer": "MLflow is an open source platform for ML lifecycle management."}
            ),
        },
        tags={
            "mlflow.traceName": "test_trace",
            "mlflow.source.name": "test",
            "mlflow.source.type": "LOCAL",
        },
    )

    spans = [
        create_test_span(
            span_id=1,
            parent_id=None,
            name="root_span",
            inputs={"question": "What is MLflow?"},
            outputs={"response": "MLflow is an open source platform"},
            span_type=SpanType.CHAIN,
        ),
        create_test_span(
            span_id=2,
            parent_id=1,
            name="llm_call",
            inputs={"prompt": "Explain MLflow"},
            outputs={"text": "MLflow is an open source platform for ML lifecycle management."},
            span_type=SpanType.LLM,
        ),
    ]

    trace_data = TraceData(spans=spans)
    return Trace(info=trace_info, data=trace_data)


def test_make_judge_creates_instructions_judge():
    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ outputs }} is formal",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    assert isinstance(judge, InstructionsJudge)
    assert judge.name == "test_judge"
    assert judge.instructions == "Check if {{ outputs }} is formal"
    assert judge.model == "openai:/gpt-4"


def test_make_judge_with_default_model(monkeypatch):
    expected_model = "openai:/gpt-4-test"
    monkeypatch.setattr(
        "mlflow.genai.judges.instructions_judge.get_default_model",
        lambda: expected_model,
    )

    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ outputs }} is accurate",
        feedback_value_type=str,
    )

    assert judge.model == expected_model


def test_make_judge_with_databricks_default(monkeypatch):
    # Mock the parent module first to prevent ImportError
    mock_evals_module = types.ModuleType("databricks.agents.evals")
    monkeypatch.setitem(sys.modules, "databricks.agents.evals", mock_evals_module)

    # Then mock the judges submodule
    mock_judges_module = types.ModuleType("databricks.agents.evals.judges")
    monkeypatch.setitem(sys.modules, "databricks.agents.evals.judges", mock_judges_module)

    monkeypatch.setattr("mlflow.genai.judges.utils.is_databricks_uri", lambda x: True)

    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ outputs }} is valid",
        feedback_value_type=str,
    )

    assert judge.model == "databricks"


def test_databricks_model_requires_databricks_agents(monkeypatch):
    # NB: Mock both the parent module and the specific module to simulate missing databricks-agents
    monkeypatch.setitem(sys.modules, "databricks.agents.evals", None)
    monkeypatch.setitem(sys.modules, "databricks.agents.evals.judges", None)

    with pytest.raises(
        MlflowException,
        match="To use 'databricks' as the judge model, the Databricks agents library",
    ):
        make_judge(
            name="test_judge",
            instructions="Check if {{ outputs }} is valid",
            feedback_value_type=str,
            model="databricks",
        )


@pytest.mark.parametrize("provider", {"vertexai", "cohere", "replicate", "groq", "together"})
def test_litellm_provider_requires_litellm(monkeypatch, provider):
    monkeypatch.setitem(sys.modules, "litellm", None)

    with pytest.raises(
        MlflowException,
        match=f"LiteLLM is required for using '{provider}' as a provider",
    ):
        make_judge(
            name="test_judge",
            instructions="Check if {{ outputs }} is valid",
            feedback_value_type=str,
            model=f"{provider}:/test-model",
        )


@pytest.mark.parametrize(
    "provider",
    _NATIVE_PROVIDERS,
)
def test_native_providers_work_without_litellm(monkeypatch, provider):
    monkeypatch.setitem(sys.modules, "litellm", None)

    judge = make_judge(
        name=f"test_judge_{provider}",
        instructions="Check if {{ outputs }} is valid",
        feedback_value_type=str,
        model=f"{provider}:/test-model",
    )
    assert judge.model == f"{provider}:/test-model"


def test_validate_judge_model_function():
    # Test valid models don't raise
    validate_judge_model("openai:/gpt-4")
    validate_judge_model("anthropic:/claude-3")
    validate_judge_model("endpoints:/my-endpoint")

    # Test invalid model format raises
    with pytest.raises(MlflowException, match="Malformed model uri"):
        validate_judge_model("invalid-model")

    with pytest.raises(MlflowException, match="Malformed model uri"):
        validate_judge_model("openai:")

    with pytest.raises(MlflowException, match="Malformed model uri"):
        validate_judge_model(":/model")


def test_databricks_model_works_with_chat_completions(mock_databricks_rag_eval):
    mock_databricks_rag_eval.get_context = lambda: mock_databricks_rag_eval.MockContext(
        expected_content="outputs", response_data={"result": True, "rationale": "Valid output"}
    )

    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ outputs }} is valid",
        feedback_value_type=str,
        model="databricks",
    )

    result = judge(outputs={"text": "test output"})
    assert isinstance(result, Feedback)
    assert result.value is True
    assert result.rationale == "Valid output"


def test_databricks_model_handles_errors_gracefully(mock_databricks_rag_eval):
    class MockLLMResultInvalid:
        def __init__(self):
            invalid_text = "This is not valid JSON - maybe the model returned plain text"
            self.output = invalid_text
            self.output_json = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": invalid_text}}]}
            )

    class MockClientInvalid:
        def get_chat_completions_result(self, user_prompt, system_prompt, **kwargs):
            return MockLLMResultInvalid()

    class MockContextInvalid:
        def build_managed_rag_client(self):
            return MockClientInvalid()

    mock_databricks_rag_eval.get_context = lambda: MockContextInvalid()

    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ outputs }} is valid",
        feedback_value_type=str,
        model="databricks",
    )

    result = judge(outputs={"text": "test output"})
    assert isinstance(result, Feedback)
    assert result.error is not None
    # String errors are converted to AssessmentError objects
    assert isinstance(result.error, AssessmentError)
    assert "Invalid JSON response" in result.error.error_message

    class MockLLMResultMissingField:
        def __init__(self):
            data = {"rationale": "Some rationale but no result field"}
            self.output = json.dumps(data)
            self.output_json = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": json.dumps(data)}}]}
            )

    class MockClientMissingField:
        def get_chat_completions_result(self, user_prompt, system_prompt, **kwargs):
            return MockLLMResultMissingField()

    class MockContextMissingField:
        def build_managed_rag_client(self):
            return MockClientMissingField()

    mock_databricks_rag_eval.get_context = lambda: MockContextMissingField()

    result = judge(outputs={"text": "test output"})
    assert isinstance(result, Feedback)
    assert result.error is not None
    # String errors are converted to AssessmentError objects
    assert isinstance(result.error, AssessmentError)
    assert "Response missing 'result' field" in result.error.error_message

    class MockLLMResultNone:
        def __init__(self):
            self.output = None
            self.output_json = None

    class MockClientNone:
        def get_chat_completions_result(self, user_prompt, system_prompt, **kwargs):
            return MockLLMResultNone()

    class MockContextNone:
        def build_managed_rag_client(self):
            return MockClientNone()

    mock_databricks_rag_eval.get_context = lambda: MockContextNone()

    result = judge(outputs={"text": "test output"})
    assert isinstance(result, Feedback)
    assert result.error is not None
    # String errors are converted to AssessmentError objects
    assert isinstance(result.error, AssessmentError)
    assert "Empty response from Databricks judge" in result.error.error_message


def test_databricks_model_works_with_trace(mock_databricks_rag_eval):
    mock_databricks_rag_eval.get_context = lambda: mock_databricks_rag_eval.MockContext(
        expected_content="trace", response_data={"result": True, "rationale": "Trace looks good"}
    )

    judge = make_judge(
        name="trace_judge",
        instructions="Analyze {{ trace }} for errors",
        feedback_value_type=str,
        model="databricks",
    )
    assert judge.model == "databricks"


@pytest.mark.parametrize(
    ("instructions", "expected_vars"),
    [
        (
            "Check if {{ inputs }} is correct",
            {"inputs"},
        ),
        (
            "Check {{ outputs }} against expectations",
            {"outputs"},
        ),
        (
            "Validate {{ inputs }} and {{ outputs }}",
            {"inputs", "outputs"},
        ),
        (
            "Check {{ inputs }}, {{ outputs }}, and {{ expectations }}",
            {"inputs", "outputs", "expectations"},
        ),
        (
            "Analyze this {{ trace }}",
            {"trace"},
        ),
    ],
)
def test_template_variable_extraction(instructions, expected_vars):
    judge = make_judge(
        name="test_judge",
        instructions=instructions,
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    assert judge.template_variables == expected_vars


@pytest.mark.parametrize(
    ("instructions", "error_pattern"),
    [
        (
            "Check if {{ query }} is answered by {{ response }}",
            "Instructions template contains unsupported variables: {'query', 'response'}",
        ),
        (
            "Check {{ answer }} against {{ expected_answer }}",
            "Instructions template contains unsupported variables: {'answer', 'expected_answer'}",
        ),
        (
            "Validate {{ custom_field }}",
            "Instructions template contains unsupported variables: {'custom_field'}",
        ),
    ],
)
def test_custom_variables_rejected(instructions, error_pattern):
    with pytest.raises(
        MlflowException, match="Instructions template contains unsupported variables"
    ):
        make_judge(
            name="test_judge",
            instructions=instructions,
            feedback_value_type=str,
            model="openai:/gpt-4",
        )


@pytest.mark.parametrize(
    ("name", "instructions", "model", "error_pattern"),
    [
        ("", "Check {{ outputs }}", "openai:/gpt-4", "name must be a non-empty string"),
        ("test", "", "openai:/gpt-4", "instructions must be a non-empty string"),
        (
            "test",
            "Check response",
            "openai:/gpt-4",
            "Instructions template must contain at least one variable",
        ),
        (
            "test",
            "Check {{ outputs }}",
            "invalid-model",
            "Malformed model uri 'invalid-model'",
        ),
        ("test", "Check {{ outputs }}", "invalid:/", "Malformed model uri 'invalid:/'"),
        ("test", "Check {{ outputs }}", "openai:", "Malformed model uri 'openai:'"),
    ],
)
def test_validation_errors(name, instructions, model, error_pattern):
    with pytest.raises(MlflowException, match=error_pattern):
        make_judge(name=name, instructions=instructions, feedback_value_type=str, model=model)


@pytest.mark.parametrize(
    "model",
    [
        "databricks",
        "openai:/gpt-4",
        "anthropic:/claude-3",
        "endpoints:/my-endpoint",
        "bedrock:/claude-v1",
    ],
)
def test_valid_model_formats(monkeypatch, model):
    # Mock databricks.agents.evals modules for the databricks model case
    if model == "databricks":
        # Mock the parent module first to prevent ImportError
        mock_evals_module = types.ModuleType("databricks.agents.evals")
        monkeypatch.setitem(sys.modules, "databricks.agents.evals", mock_evals_module)

        # Then mock the judges submodule
        mock_judges_module = types.ModuleType("databricks.agents.evals.judges")
        monkeypatch.setitem(sys.modules, "databricks.agents.evals.judges", mock_judges_module)

    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ outputs }} is valid",
        feedback_value_type=str,
        model=model,
    )
    assert judge.model == model


@pytest.mark.parametrize(
    ("instructions", "model", "error_pattern"),
    [
        (
            "Analyze {{ trace }} and check {{ custom_field }}",
            "openai:/gpt-4",
            "Instructions template contains unsupported variables",
        ),
    ],
)
def test_trace_variable_restrictions(instructions, model, error_pattern):
    with pytest.raises(MlflowException, match=error_pattern):
        make_judge(
            name="test_judge",
            instructions=instructions,
            feedback_value_type=str,
            model=model,
        )


def test_trace_with_inputs_outputs_allowed():
    judge1 = make_judge(
        name="test_judge",
        instructions="Analyze {{ trace }} and {{ inputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )
    assert judge1.template_variables == {"trace", "inputs"}

    judge2 = make_judge(
        name="test_judge",
        instructions="Analyze {{ trace }} and {{ outputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )
    assert judge2.template_variables == {"trace", "outputs"}


def test_trace_with_expectations_allowed():
    judge = make_judge(
        name="test_judge",
        instructions="Analyze {{ trace }} against {{ expectations }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    assert judge is not None
    assert "trace" in judge.template_variables
    assert "expectations" in judge.template_variables


def test_call_with_trace_supported(mock_trace, monkeypatch):
    captured_args = {}

    def mock_invoke(
        model_uri,
        prompt,
        assessment_name,
        trace=None,
        num_retries=10,
        response_format=None,
        use_case=None,
        inference_params=None,
    ):
        captured_args.update(
            {
                "model_uri": model_uri,
                "prompt": prompt,
                "assessment_name": assessment_name,
                "trace": trace,
                "num_retries": num_retries,
                "response_format": response_format,
                "use_case": use_case,
                "inference_params": inference_params,
            }
        )
        return Feedback(name=assessment_name, value=True, rationale="Trace analyzed")

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", mock_invoke)

    judge = make_judge(
        name="test_judge",
        instructions="Analyze this {{ trace }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    result = judge(trace=mock_trace)

    assert isinstance(result, Feedback)
    assert captured_args["trace"] == mock_trace
    assert captured_args["model_uri"] == "openai:/gpt-4"
    assert captured_args["assessment_name"] == "test_judge"


def test_call_trace_based_judge_ignores_inputs_outputs(mock_trace, mock_invoke_judge_model):
    # Test that trace-based judges ignore inputs/outputs and work with trace only
    captured_args = mock_invoke_judge_model.captured_args

    judge = make_judge(
        name="test_judge",
        instructions="Analyze this {{ trace }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    # These should all work - trace-based judge ignores inputs/outputs
    result1 = judge(trace=mock_trace, inputs={"query": "test"})
    assert isinstance(result1, Feedback)
    assert captured_args["trace"] == mock_trace

    result2 = judge(trace=mock_trace, outputs={"answer": "test"})
    assert isinstance(result2, Feedback)
    assert captured_args["trace"] == mock_trace

    result3 = judge(trace=mock_trace, expectations={"expected": "test"})
    assert isinstance(result3, Feedback)
    assert captured_args["trace"] == mock_trace


def test_call_with_no_inputs_or_outputs():
    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ outputs }} is valid",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    with pytest.raises(
        MlflowException, match="Must specify 'outputs' - required by template variables"
    ):
        judge()


def test_call_with_valid_outputs_returns_feedback(mock_invoke_judge_model):
    judge = make_judge(
        name="formality_judge",
        instructions="Check if {{ outputs }} is formal",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    test_output = "Dear Sir/Madam, I am writing to inquire..."
    result = judge(outputs={"response": test_output})

    assert isinstance(result, Feedback)
    assert result.name == "formality_judge"
    assert result.value is True
    assert result.rationale == "The response is formal"

    # Verify the prompt contains the outputs value
    assert len(mock_invoke_judge_model.calls) == 1
    model_uri, prompt, assessment_name = mock_invoke_judge_model.calls[0]
    assert isinstance(prompt, list)
    assert len(prompt) == 2
    # Check that the user message contains the JSON-serialized outputs
    user_msg = prompt[1]
    expected_outputs_json = json.dumps({"response": test_output}, default=str, indent=2)
    assert expected_outputs_json in user_msg.content


def test_call_with_valid_inputs_returns_feedback(mock_invoke_judge_model):
    judge = make_judge(
        name="input_judge",
        instructions="Check if {{ inputs }} is valid",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    test_input = {"query": "What is MLflow?"}
    result = judge(inputs=test_input)

    assert isinstance(result, Feedback)
    assert result.name == "input_judge"
    assert result.value is True
    assert result.rationale == "The response is formal"

    # Verify the prompt contains the inputs value as JSON
    assert len(mock_invoke_judge_model.calls) == 1
    model_uri, prompt, assessment_name = mock_invoke_judge_model.calls[0]
    user_msg = prompt[1]

    expected_inputs_json = json.dumps(test_input, default=str, indent=2)
    assert expected_inputs_json in user_msg.content


def test_call_with_valid_inputs_and_outputs_returns_feedback(mock_invoke_judge_model):
    judge = make_judge(
        name="inputs_outputs_judge",
        instructions="Check if {{ outputs }} matches {{ inputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    test_input = {"query": "What is MLflow?"}
    test_output = {"response": "MLflow is an open source platform"}
    result = judge(inputs=test_input, outputs=test_output)

    assert isinstance(result, Feedback)
    assert result.name == "inputs_outputs_judge"
    assert result.value is True
    assert result.rationale == "The response is formal"

    # Verify the prompt contains both inputs and outputs values as JSON
    assert len(mock_invoke_judge_model.calls) == 1
    model_uri, prompt, assessment_name = mock_invoke_judge_model.calls[0]
    user_msg = prompt[1]

    expected_inputs_json = json.dumps(test_input, default=str, indent=2)
    expected_outputs_json = json.dumps(test_output, default=str, indent=2)
    assert expected_inputs_json in user_msg.content
    assert expected_outputs_json in user_msg.content


def test_call_with_expectations_as_json(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Check {{ outputs }} against {{ expectations }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    expectations = {"correct": True, "score": 100}
    judge(outputs={"answer": "42"}, expectations=expectations)

    # Check that we have a list of messages
    captured_messages = mock_invoke_judge_model.captured_args["prompt"]
    assert isinstance(captured_messages, list)
    assert len(captured_messages) == 2

    # Expectations should be in the user message as JSON
    user_msg = captured_messages[1]
    expected_expectations_json = json.dumps(expectations, default=str, indent=2)
    assert expected_expectations_json in user_msg.content


def test_call_with_reserved_variables(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ inputs }} meets {{ expectations }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    inputs_data = {"question": "What is AI?"}
    expectations_data = {"criteria": "technical accuracy"}
    result = judge(inputs=inputs_data, expectations=expectations_data)

    assert isinstance(result, Feedback)

    # Check that we have a list of messages
    captured_messages = mock_invoke_judge_model.captured_args["prompt"]
    assert isinstance(captured_messages, list)
    assert len(captured_messages) == 2

    # Check system message has the template
    system_msg = captured_messages[0]
    assert "Check if {{ inputs }} meets {{ expectations }}" in system_msg.content

    # Check user message has the JSON dumps of inputs and expectations
    user_msg = captured_messages[1]
    expected_inputs_json = json.dumps(inputs_data, default=str, indent=2)
    expected_expectations_json = json.dumps(expectations_data, default=str, indent=2)
    assert expected_inputs_json in user_msg.content
    assert expected_expectations_json in user_msg.content
    assert "technical accuracy" in user_msg.content
    assert "What is AI?" in user_msg.content


def test_instructions_property():
    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ outputs }} is formal",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    instructions = judge.instructions
    assert instructions == "Check if {{ outputs }} is formal"


def test_kind_property():
    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ outputs }} is valid",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    assert judge.kind == ScorerKind.INSTRUCTIONS


@pytest.mark.parametrize(
    ("inputs", "outputs", "expectations", "should_fail"),
    [
        ({"text": "hello", "result": "world"}, None, None, True),  # Missing outputs
        (
            {"text": "hello"},
            {"result": "world"},
            None,
            False,
        ),  # Valid: both inputs and outputs
        (
            {"text": "hello"},
            {"result": "world"},
            {"expected": "world"},
            False,
        ),  # Valid: all
        (None, {"text": "hello", "result": "world"}, None, True),  # Missing inputs
    ],
)
def test_call_with_various_input_combinations(
    mock_invoke_judge_model, inputs, outputs, expectations, should_fail
):
    judge = make_judge(
        name="test_judge",
        instructions="Check {{ inputs }} and {{ outputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    if should_fail:
        with pytest.raises(
            MlflowException, match="Must specify .* - required by template variables"
        ):
            judge(inputs=inputs, outputs=outputs, expectations=expectations)
    else:
        result = judge(inputs=inputs, outputs=outputs, expectations=expectations)
        assert isinstance(result, Feedback)


def test_prompt_formatting_with_all_reserved_variable_types(mock_invoke_judge_model):
    judge = make_judge(
        name="test",
        instructions=(
            "Inputs: {{ inputs }}, Outputs: {{ outputs }}, Expectations: {{ expectations }}"
        ),
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    inputs_data = {"query": "test", "context": "testing"}
    outputs_data = {"response": "answer", "score": 0.9}
    expectations_data = {"expected": "correct answer"}

    judge(inputs=inputs_data, outputs=outputs_data, expectations=expectations_data)

    # Check that we have a list of messages
    captured_messages = mock_invoke_judge_model.captured_args["prompt"]
    assert isinstance(captured_messages, list)
    assert len(captured_messages) == 2

    # Check system message has the template
    system_msg = captured_messages[0]
    expected_template = (
        "Inputs: {{ inputs }}, Outputs: {{ outputs }}, Expectations: {{ expectations }}"
    )
    assert expected_template in system_msg.content

    # Check user message has all the JSON-serialized values
    user_msg = captured_messages[1]
    expected_inputs_json = json.dumps(inputs_data, default=str, indent=2)
    expected_outputs_json = json.dumps(outputs_data, default=str, indent=2)
    expected_expectations_json = json.dumps(expectations_data, default=str, indent=2)
    assert expected_inputs_json in user_msg.content
    assert expected_outputs_json in user_msg.content
    assert expected_expectations_json in user_msg.content


def test_output_format_instructions_added(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ outputs }} is formal",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    result = judge(outputs={"text": "Hello there"})

    # Check that we have a list of messages
    captured_messages = mock_invoke_judge_model.captured_args["prompt"]
    assert isinstance(captured_messages, list)
    assert len(captured_messages) == 2

    # Check system message contains proper output format instructions
    system_msg = captured_messages[0]
    assert system_msg.role == "system"
    assert system_msg.content.startswith(JUDGE_BASE_PROMPT)
    assert "Check if {{ outputs }} is formal" in system_msg.content
    # Tighter assertion for output format instructions
    assert "Please provide your assessment in the following JSON format only" in system_msg.content
    assert '"result": "The evaluation rating/result"' in system_msg.content
    assert '"rationale": "Detailed explanation for the evaluation"' in system_msg.content

    assert result.value is True


def test_output_format_instructions_with_complex_template(mock_invoke_judge_model):
    judge = make_judge(
        name="complex_judge",
        instructions="Evaluate {{ outputs }} considering {{ inputs }} and {{ expectations }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    judge(
        inputs={"context": "formal business setting"},
        outputs={"response": "Hey what's up"},
        expectations={"criteria": "professionalism"},
    )

    # Check that we have a list of messages
    captured_messages = mock_invoke_judge_model.captured_args["prompt"]
    assert isinstance(captured_messages, list)
    assert len(captured_messages) == 2

    # Check system message contains proper output format instructions
    system_msg = captured_messages[0]
    assert system_msg.role == "system"
    assert system_msg.content.startswith(JUDGE_BASE_PROMPT)
    assert (
        "Evaluate {{ outputs }} considering {{ inputs }} and {{ expectations }}"
        in system_msg.content
    )
    # Tighter assertion for output format instructions
    assert "Please provide your assessment in the following JSON format only" in system_msg.content
    assert '"result": "The evaluation rating/result"' in system_msg.content
    assert '"rationale": "Detailed explanation for the evaluation"' in system_msg.content


def test_judge_registration_as_scorer(mock_invoke_judge_model):
    experiment = mlflow.create_experiment("test_judge_registration")

    original_instructions = "Evaluate if the {{ outputs }} is professional and formal."
    judge = make_judge(
        name="test_judge",
        instructions=original_instructions,
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    assert judge.instructions == original_instructions
    assert judge.model == "openai:/gpt-4"
    assert judge.template_variables == {"outputs"}

    serialized = judge.model_dump()
    assert "name" in serialized
    assert serialized["name"] == "test_judge"
    assert "instructions_judge_pydantic_data" in serialized
    assert serialized["instructions_judge_pydantic_data"]["instructions"] == original_instructions
    assert serialized["instructions_judge_pydantic_data"]["model"] == "openai:/gpt-4"

    store = _get_scorer_store()
    version = store.register_scorer(experiment, judge)
    assert version.scorer_version == 1

    retrieved_scorer = store.get_scorer(experiment, "test_judge", version.scorer_version)
    assert retrieved_scorer is not None
    assert isinstance(retrieved_scorer, InstructionsJudge)
    assert retrieved_scorer.name == "test_judge"
    assert retrieved_scorer.instructions == original_instructions
    assert retrieved_scorer.model == "openai:/gpt-4"
    assert retrieved_scorer.template_variables == {"outputs"}

    deserialized = Scorer.model_validate(serialized)
    assert isinstance(deserialized, InstructionsJudge)
    assert deserialized.name == judge.name
    assert deserialized.instructions == original_instructions
    assert deserialized.model == judge.model
    assert deserialized.template_variables == {"outputs"}

    test_output = {"response": "This output demonstrates professional communication."}
    result = retrieved_scorer(outputs=test_output)
    assert isinstance(result, Feedback)
    assert result.name == "test_judge"

    assert len(mock_invoke_judge_model.calls) == 1
    model_uri, prompt, assessment_name = mock_invoke_judge_model.calls[0]
    assert model_uri == "openai:/gpt-4"
    assert assessment_name == "test_judge"

    # Check that prompt is now a list of ChatMessage objects
    assert isinstance(prompt, list)
    assert len(prompt) == 2

    # Check system message
    assert prompt[0].role == "system"
    assert prompt[0].content.startswith(JUDGE_BASE_PROMPT)
    assert "Evaluate if the {{ outputs }} is professional and formal." in prompt[0].content
    assert "JSON format" in prompt[0].content

    # Check user message
    assert prompt[1].role == "user"
    assert "outputs:" in prompt[1].content
    assert "This output demonstrates professional communication." in prompt[1].content

    mock_invoke_judge_model.reset_mock()
    result2 = deserialized(outputs=test_output)
    assert isinstance(result2, Feedback)
    assert len(mock_invoke_judge_model.calls) == 1
    model_uri, prompt, assessment_name = mock_invoke_judge_model.calls[0]
    assert model_uri == "openai:/gpt-4"
    assert assessment_name == "test_judge"

    # Verify the same message structure for deserialized judge
    assert isinstance(prompt, list)
    assert len(prompt) == 2
    assert prompt[0].role == "system"
    assert prompt[1].role == "user"
    assert "outputs:" in prompt[1].content
    assert "This output demonstrates professional communication." in prompt[1].content

    v2_instructions = "Evaluate if the output {{ outputs }} is professional, formal, and concise."
    judge_v2 = make_judge(
        name="test_judge",
        instructions=v2_instructions,
        feedback_value_type=str,
        model="openai:/gpt-4o",
    )
    version2 = store.register_scorer(experiment, judge_v2)
    assert version2.scorer_version == 2

    versions = store.list_scorer_versions(experiment, "test_judge")
    assert len(versions) == 2

    v1_scorer, v1_num = versions[0]
    assert v1_num == 1
    assert isinstance(v1_scorer, InstructionsJudge)
    assert v1_scorer.instructions == original_instructions
    assert v1_scorer.model == "openai:/gpt-4"

    v2_scorer, v2_num = versions[1]
    assert v2_num == 2
    assert isinstance(v2_scorer, InstructionsJudge)
    assert v2_scorer.instructions == v2_instructions
    assert v2_scorer.model == "openai:/gpt-4o"

    latest = store.get_scorer(experiment, "test_judge")
    assert isinstance(latest, InstructionsJudge)
    assert latest.instructions == v2_instructions
    assert latest.model == "openai:/gpt-4o"


def test_judge_registration_with_reserved_variables(mock_invoke_judge_model):
    experiment = mlflow.create_experiment("test_reserved_vars")

    instructions_with_reserved = (
        "Check if {{ inputs }} is answered correctly by {{ outputs }} "
        "according to {{ expectations }}"
    )
    judge = make_judge(
        name="reserved_judge",
        instructions=instructions_with_reserved,
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    assert judge.template_variables == {"inputs", "outputs", "expectations"}

    store = _get_scorer_store()
    version = store.register_scorer(experiment, judge)
    assert version.scorer_version == 1

    retrieved_judge = store.get_scorer(experiment, "reserved_judge", version.scorer_version)
    assert isinstance(retrieved_judge, InstructionsJudge)
    assert retrieved_judge.instructions == instructions_with_reserved
    assert retrieved_judge.template_variables == {"inputs", "outputs", "expectations"}

    result = retrieved_judge(
        inputs={"query": "What is 2+2?", "context": "mathematical"},
        outputs={"response": "The answer is 4", "confidence": 0.95},
        expectations={"criteria": "mathematical accuracy", "threshold": "95%"},
    )
    assert isinstance(result, Feedback)
    assert result.name == "reserved_judge"

    assert len(mock_invoke_judge_model.calls) == 1
    model_uri, prompt, assessment_name = mock_invoke_judge_model.calls[0]
    assert model_uri == "openai:/gpt-4"
    assert assessment_name == "reserved_judge"

    # Check that prompt is now a list of ChatMessage objects
    assert isinstance(prompt, list)
    assert len(prompt) == 2

    # Check system message
    assert prompt[0].role == "system"
    assert prompt[0].content.startswith(JUDGE_BASE_PROMPT)
    assert "Check if {{ inputs }} is answered correctly by {{ outputs }}" in prompt[0].content
    assert "according to {{ expectations }}" in prompt[0].content
    assert "JSON format" in prompt[0].content

    # Check user message with all reserved variables as JSON
    assert prompt[1].role == "user"
    user_content = prompt[1].content
    assert "expectations:" in user_content
    assert "inputs:" in user_content
    assert "outputs:" in user_content
    # Verify the JSON contains the actual data
    assert "query" in user_content
    assert "What is 2+2?" in user_content
    assert "response" in user_content
    assert "The answer is 4" in user_content
    assert "mathematical accuracy" in user_content


def test_model_dump_comprehensive():
    basic_judge = make_judge(
        name="basic_judge",
        instructions="Check if {{ inputs }} is correct",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    serialized = basic_judge.model_dump()

    assert isinstance(serialized, dict)
    assert "name" in serialized
    assert serialized["name"] == "basic_judge"

    assert "mlflow_version" in serialized
    assert serialized["mlflow_version"] == mlflow.__version__
    assert "serialization_version" in serialized
    assert serialized["serialization_version"] == 1

    assert "aggregations" in serialized
    assert serialized["aggregations"] == []

    assert "instructions_judge_pydantic_data" in serialized
    assert isinstance(serialized["instructions_judge_pydantic_data"], dict)
    assert "instructions" in serialized["instructions_judge_pydantic_data"]
    assert (
        serialized["instructions_judge_pydantic_data"]["instructions"]
        == "Check if {{ inputs }} is correct"
    )
    assert "model" in serialized["instructions_judge_pydantic_data"]
    assert serialized["instructions_judge_pydantic_data"]["model"] == "openai:/gpt-4"

    assert "builtin_scorer_class" in serialized
    assert serialized["builtin_scorer_class"] is None
    assert "builtin_scorer_pydantic_data" in serialized
    assert serialized["builtin_scorer_pydantic_data"] is None
    assert "call_source" in serialized
    assert serialized["call_source"] is None
    assert "call_signature" in serialized
    assert serialized["call_signature"] is None
    assert "original_func_name" in serialized
    assert serialized["original_func_name"] is None

    complex_judge = make_judge(
        name="complex_judge",
        instructions="Check if {{ inputs }} matches {{ expectations }}",
        feedback_value_type=str,
        model="anthropic:/claude-3",
    )

    complex_serialized = complex_judge.model_dump()

    assert complex_serialized["instructions_judge_pydantic_data"]["instructions"] == (
        "Check if {{ inputs }} matches {{ expectations }}"
    )
    assert complex_serialized["instructions_judge_pydantic_data"]["model"] == "anthropic:/claude-3"

    default_model_judge = make_judge(
        name="default_judge",
        instructions="Evaluate {{ outputs }}",
        feedback_value_type=str,
    )

    default_serialized = default_model_judge.model_dump()
    assert default_serialized["instructions_judge_pydantic_data"]["model"] in [
        "databricks",
        "openai:/gpt-4.1-mini",
    ]

    for serialized_data in [serialized, complex_serialized, default_serialized]:
        deserialized = Scorer.model_validate(serialized_data)
        assert isinstance(deserialized, InstructionsJudge)
        assert deserialized.name == serialized_data["name"]
        raw_instructions = serialized_data["instructions_judge_pydantic_data"]["instructions"]
        assert deserialized.instructions == raw_instructions
        assert deserialized.model == serialized_data["instructions_judge_pydantic_data"]["model"]


def test_instructions_judge_deserialization_validation():
    invalid_data_missing_instructions = {
        "name": "test_judge",
        "aggregations": None,
        "mlflow_version": mlflow.__version__,
        "serialization_version": 1,
        "instructions_judge_pydantic_data": {"model": "openai:/gpt-4"},
        "builtin_scorer_class": None,
        "builtin_scorer_pydantic_data": None,
        "call_source": None,
        "call_signature": None,
        "original_func_name": None,
    }

    with pytest.raises(MlflowException, match="missing required field 'instructions'"):
        Scorer.model_validate(invalid_data_missing_instructions)

    invalid_data_missing_model = {
        "name": "test_judge",
        "aggregations": None,
        "mlflow_version": mlflow.__version__,
        "serialization_version": 1,
        "instructions_judge_pydantic_data": {"instructions": "Check {{ inputs }}"},
        "builtin_scorer_class": None,
        "builtin_scorer_pydantic_data": None,
        "call_source": None,
        "call_signature": None,
        "original_func_name": None,
    }

    with pytest.raises(MlflowException, match="missing required field 'model'"):
        Scorer.model_validate(invalid_data_missing_model)

    invalid_data_wrong_type = {
        "name": "test_judge",
        "aggregations": None,
        "mlflow_version": mlflow.__version__,
        "serialization_version": 1,
        "instructions_judge_pydantic_data": {
            "instructions": 123,
            "model": "openai:/gpt-4",
        },
        "builtin_scorer_class": None,
        "builtin_scorer_pydantic_data": None,
        "call_source": None,
        "call_signature": None,
        "original_func_name": None,
    }

    with pytest.raises(MlflowException, match="field 'instructions' must be str, got int"):
        Scorer.model_validate(invalid_data_wrong_type)


def test_model_dump_uses_serialized_scorer_dataclass():
    judge = make_judge(
        name="test_dataclass_judge",
        instructions="Evaluate {{ inputs }} and {{ outputs }}",
        feedback_value_type=str,
        model="openai:/gpt-3.5-turbo",
    )

    serialized = judge.model_dump()

    expected_scorer = SerializedScorer(
        name="test_dataclass_judge",
        aggregations=[],
        is_session_level_scorer=False,
        mlflow_version=mlflow.__version__,
        serialization_version=1,
        instructions_judge_pydantic_data={
            "feedback_value_type": {
                "type": "string",
                "title": "Result",
            },
            "instructions": "Evaluate {{ inputs }} and {{ outputs }}",
            "model": "openai:/gpt-3.5-turbo",
        },
        builtin_scorer_class=None,
        builtin_scorer_pydantic_data=None,
        call_source=None,
        call_signature=None,
        original_func_name=None,
    )

    expected_dict = asdict(expected_scorer)

    assert serialized == expected_dict

    assert set(serialized.keys()) == set(expected_dict.keys())


def test_model_dump_session_level_scorer():
    judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate the {{ conversation }} for coherence",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    # Verify it's a session-level scorer
    assert judge.is_session_level_scorer is True

    serialized = judge.model_dump()

    # Verify is_session_level_scorer is properly serialized
    assert serialized["is_session_level_scorer"] is True
    assert serialized["name"] == "conversation_judge"

    expected_scorer = SerializedScorer(
        name="conversation_judge",
        aggregations=[],
        is_session_level_scorer=True,
        mlflow_version=mlflow.__version__,
        serialization_version=1,
        instructions_judge_pydantic_data={
            "feedback_value_type": {
                "type": "string",
                "title": "Result",
            },
            "instructions": "Evaluate the {{ conversation }} for coherence",
            "model": "openai:/gpt-4",
        },
        builtin_scorer_class=None,
        builtin_scorer_pydantic_data=None,
        call_source=None,
        call_signature=None,
        original_func_name=None,
    )

    expected_dict = asdict(expected_scorer)
    assert serialized == expected_dict

    # Test deserialization preserves is_session_level_scorer
    deserialized = Scorer.model_validate(serialized)
    assert deserialized.is_session_level_scorer is True
    assert deserialized.name == "conversation_judge"


def test_instructions_judge_works_with_evaluate(mock_invoke_judge_model):
    judge = make_judge(
        name="response_quality",
        instructions="Evaluate if the {{ outputs }} is helpful given {{ inputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    assert judge.aggregations == []

    data = pd.DataFrame(
        {
            "inputs": [
                {"question": "What is MLflow?"},
                {"question": "How to track experiments?"},
            ],
            "outputs": [
                {"response": "MLflow is an open source platform for ML lifecycle."},
                {"response": "Use mlflow.start_run() to track experiments."},
            ],
        }
    )

    result = mlflow.genai.evaluate(data=data, scorers=[judge])

    assert "response_quality/mean" not in result.metrics
    assert "response_quality/value" in result.result_df.columns
    assert len(result.result_df["response_quality/value"]) == 2
    assert all(score is True for score in result.result_df["response_quality/value"])


@pytest.mark.parametrize(
    ("trace_inputs", "trace_outputs", "span_inputs", "span_outputs"),
    [
        (
            {"question": "What is MLflow?"},
            {"answer": "MLflow is a platform"},
            {"prompt": "Explain"},
            {"response": "MLflow helps"},
        ),
        ("What is 2+2?", "The answer is 4", {"query": "Solve this"}, {"result": "4"}),
        (
            {"question": "What is AI?"},
            "AI is intelligence",
            {"query": "Define AI"},
            {"response": "Artificial Intelligence"},
        ),
        (
            "Calculate 5+5",
            {"result": 10, "confidence": 0.99},
            {"task": "Simple math"},
            {"answer": 10},
        ),
        ({}, {}, {}, {}),
        (None, None, None, None),
        (
            {"user": {"id": 1, "question": "Help"}},
            {"response": {"text": "Sure!", "metadata": {"lang": "en"}}},
            {"context": [1, 2, 3]},
            {"output": [{"type": "text", "value": "response"}]},
        ),
        (42, True, {"number": 3.14}, {"result": False}),
        (["question1", "question2"], ["answer1", "answer2"], {"list": [1, 2]}, {"output": [3, 4]}),
    ],
)
def test_instructions_judge_works_with_evaluate_on_trace(
    mock_invoke_judge_model, trace_inputs, trace_outputs, span_inputs, span_outputs
):
    with mlflow.start_span(name="test", span_type=SpanType.CHAIN) as span:
        span.set_inputs(trace_inputs)
        span.set_outputs(trace_outputs)

        mlflow.update_current_trace(
            metadata={
                "mlflow.traceInputs": json.dumps(trace_inputs),
                "mlflow.traceOutputs": json.dumps(trace_outputs),
            }
        )

    trace = mlflow.get_trace(span.trace_id)
    judge = make_judge(
        name="trace_evaluator",
        instructions="Analyze this {{trace}} for quality and correctness",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )
    data = pd.DataFrame({"trace": [trace]})
    result = mlflow.genai.evaluate(data=data, scorers=[judge])

    assert "trace_evaluator/value" in result.result_df.columns
    assert len(result.result_df["trace_evaluator/value"]) == 1
    assert result.result_df["trace_evaluator/value"].iloc[0]


def test_trace_prompt_augmentation(mock_trace, monkeypatch):
    captured_prompt = None

    def mock_invoke(
        model_uri,
        prompt,
        assessment_name,
        trace=None,
        num_retries=10,
        response_format=None,
        use_case=None,
        inference_params=None,
    ):
        nonlocal captured_prompt
        captured_prompt = prompt
        return Feedback(name=assessment_name, value=True)

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", mock_invoke)

    judge = make_judge(
        name="test_judge",
        instructions="Analyze this {{ trace }} for quality",
        feedback_value_type=bool,
        model="openai:/gpt-4",
    )

    judge(trace=mock_trace)

    assert isinstance(captured_prompt, list)
    assert len(captured_prompt) == 2

    system_content = captured_prompt[0].content
    assert captured_prompt[0].role == "system"
    assert "expert judge" in system_content
    assert "step-by-step record" in system_content
    assert "provided to you" in system_content
    assert "Evaluation Rating Fields" in system_content
    assert "- result (bool): The evaluation rating/result" in system_content
    assert "- rationale (str): Detailed explanation for the evaluation" in system_content
    assert "Instructions" in system_content
    assert "Analyze this {{ trace }} for quality" in system_content


@pytest.mark.parametrize(
    ("test_value", "expect_json"),
    [
        ("simple string", True),
        (42, True),
        (3.14, True),
        (True, True),
        (False, True),
        (["item1", "item2"], True),
        ({"key": "value"}, True),
        ({"nested": {"data": [1, 2, 3]}}, True),
        ([], True),
        ({}, True),
        ("", True),
        (0, True),
        # Non-JSON-serializable objects that fall back to str()
        ({1, 2, 3}, False),
        (frozenset([4, 5, 6]), False),
        (lambda x: x + 1, False),
        (iter([1, 2, 3]), False),
        (range(3), False),
        # JSON object with non-serializable field - json.dumps works with default=str
        ({"valid_field": "ok", "bad_field": {1, 2}}, True),
    ],
)
def test_judge_accepts_various_input_output_data_types(
    mock_invoke_judge_model, test_value, expect_json
):
    judge = make_judge(
        name="test_judge",
        instructions="Compare {{inputs}} with {{outputs}}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    result = judge(inputs=test_value, outputs=test_value)
    assert isinstance(result, Feedback)

    # Verify both inputs and outputs values were serialized in the prompt
    captured_messages = mock_invoke_judge_model.captured_args["prompt"]
    user_msg = captured_messages[1]

    expected_value = (
        json.dumps(test_value, default=str, indent=2) if expect_json else str(test_value)
    )
    assert expected_value in user_msg.content
    # Should appear twice (once for inputs, once for outputs)
    assert user_msg.content.count(expected_value) == 2


def test_judge_rejects_scalar_expectations():
    judge = make_judge(
        name="test_judge",
        instructions="Compare {{ outputs }} to {{ expectations }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    with pytest.raises(MlflowException, match="'expectations' must be a dictionary, got str"):
        judge(outputs={"result": "test"}, expectations="expected value")

    with pytest.raises(MlflowException, match="'expectations' must be a dictionary, got tuple"):
        judge(outputs={"result": "test"}, expectations=("expected", "values"))


def test_judge_accepts_valid_dict_inputs(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ inputs }} and {{ outputs }} are valid",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    result = judge(
        inputs={"question": "What is MLflow?"},
        outputs={"answer": "MLflow is an open source platform"},
    )
    assert isinstance(result, Feedback)

    result = judge(inputs={}, outputs={})
    assert isinstance(result, Feedback)

    result = judge(
        inputs={"nested": {"key": "value"}},
        outputs={"response": {"status": "ok", "data": "result"}},
    )
    assert isinstance(result, Feedback)


def test_judge_rejects_invalid_trace():
    judge = make_judge(
        name="test_judge",
        instructions="Analyze this {{ trace }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    with pytest.raises(MlflowException, match="'trace' must be a Trace object, got str"):
        judge(trace="not a trace")

    with pytest.raises(MlflowException, match="'trace' must be a Trace object, got dict"):
        judge(trace={"trace_data": "invalid"})

    inputs_judge = make_judge(
        name="inputs_judge",
        instructions="Check {{ inputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )
    with pytest.raises(MlflowException, match="Must specify 'inputs'"):
        inputs_judge(trace=None)


def test_judge_accepts_valid_trace(mock_trace, mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Analyze this {{ trace }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    result = judge(trace=mock_trace)
    assert isinstance(result, Feedback)
    assert mock_invoke_judge_model.captured_args["trace"] == mock_trace


def test_instructions_judge_with_chat_messages():
    captured_args = {}

    def capture_invoke(*args, **kwargs):
        captured_args.update(kwargs)
        captured_args["args"] = args
        return Feedback(
            name=kwargs.get("assessment_name", "test"),
            value=True,
            rationale="Test passed",
        )

    messages = [
        ChatMessage(role="system", content="You are an evaluation assistant."),
        ChatMessage(role="user", content="Is this response helpful for the question?"),
    ]

    with mock.patch("mlflow.genai.judges.utils.invoke_judge_model", side_effect=capture_invoke):
        from mlflow.genai.judges.utils import invoke_judge_model

        result = invoke_judge_model(
            model_uri="openai:/gpt-4",
            prompt=messages,
            assessment_name="test_assessment",
        )

    prompt_arg = captured_args.get("prompt")
    assert prompt_arg is not None
    assert prompt_arg == messages

    judge = make_judge(
        name="response_quality",
        instructions="Evaluate if the {{ outputs }} is helpful given {{ inputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    captured_args.clear()

    with mock.patch(
        "mlflow.genai.judges.instructions_judge.invoke_judge_model",
        side_effect=capture_invoke,
    ):
        result = judge(
            inputs={"question": "What is MLflow?"},
            outputs={"response": "MLflow is great"},
        )

    assert result.value is True
    assert result.rationale == "Test passed"

    prompt_sent = captured_args.get("prompt")
    assert isinstance(prompt_sent, list)
    assert len(prompt_sent) == 2
    assert all(isinstance(msg, ChatMessage) for msg in prompt_sent)
    assert prompt_sent[0].role == "system"
    assert prompt_sent[1].role == "user"


def test_trace_field_extraction_for_inputs_outputs_template(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate if {{ outputs }} correctly answers {{ inputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace_inputs = {"question": "What is MLflow?"}
    trace_outputs = {"answer": "MLflow is an open source platform"}

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(trace_inputs)
        span.set_outputs(trace_outputs)

    trace = mlflow.get_trace(span.trace_id)
    judge(trace=trace)


@pytest.mark.parametrize(
    ("instructions", "provided_params", "expected_warning"),
    [
        (
            "Evaluate if {{ outputs }} is correct",
            {"outputs": {"answer": "42"}, "inputs": {"question": "What is life?"}},
            "'inputs'",
        ),
        (
            "Check {{ inputs }}",
            {"inputs": {"q": "test"}, "outputs": {"a": "result"}, "expectations": {"e": "42"}},
            "'outputs', 'expectations'",
        ),
        (
            "Evaluate {{ trace }}",
            {"inputs": {"q": "test"}, "outputs": {"a": "result"}},
            "'inputs', 'outputs'",
        ),
    ],
)
def test_unused_parameters_warning(
    instructions, provided_params, expected_warning, mock_invoke_judge_model
):
    judge = make_judge(
        name="test_judge",
        instructions=instructions,
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    if "{{ trace }}" in instructions:
        trace = Trace(
            info=TraceInfo(
                trace_id="test-trace-id",
                trace_location=TraceLocation.from_experiment_id("0"),
                request_time=1234567890,
                execution_duration=1000,
                state=TraceState.OK,
                trace_metadata={},
            ),
            data=TraceData(spans=[]),
        )
        provided_params = {"trace": trace, **provided_params}

    with patch("mlflow.genai.judges.instructions_judge._logger") as mock_logger:
        judge(**provided_params)

        if "{{ trace }}" in instructions:
            assert not mock_logger.warning.called
        else:
            assert mock_logger.warning.called

            warning_call_args = mock_logger.warning.call_args
            assert warning_call_args is not None

            warning_msg = warning_call_args[0][0]

            assert "parameters were provided but are not used" in warning_msg
            assert expected_warning in warning_msg


def test_context_labels_added_to_interpolated_values(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate if {{outputs}} answers {{inputs}} per {{expectations}}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    test_inputs = {"question": "What is MLflow?"}
    test_outputs = {"answer": "MLflow is an open source platform"}
    test_expectations = {"criteria": "Must mention open source"}

    judge(inputs=test_inputs, outputs=test_outputs, expectations=test_expectations)

    assert len(mock_invoke_judge_model.calls) == 1
    _, prompt, _ = mock_invoke_judge_model.calls[0]

    user_msg = prompt[1]
    user_content = user_msg.content

    assert "inputs:" in user_content, "Missing 'inputs:' label"
    assert "outputs:" in user_content, "Missing 'outputs:' label"
    assert "expectations:" in user_content, "Missing 'expectations:' label"

    expected_inputs_json = json.dumps(test_inputs, default=str, indent=2)
    expected_outputs_json = json.dumps(test_outputs, default=str, indent=2)
    expected_expectations_json = json.dumps(test_expectations, default=str, indent=2)

    assert f"inputs: {expected_inputs_json}" in user_content
    assert f"outputs: {expected_outputs_json}" in user_content
    assert f"expectations: {expected_expectations_json}" in user_content

    expectations_pos = user_content.index("expectations:")
    inputs_pos = user_content.index("inputs:")
    outputs_pos = user_content.index("outputs:")

    assert outputs_pos < inputs_pos < expectations_pos


def test_trace_field_extraction_with_non_dict_values(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate if {{ outputs }} correctly answers {{ inputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace_inputs = "What is MLflow?"
    trace_outputs = "MLflow is an open source platform"

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(trace_inputs)
        span.set_outputs(trace_outputs)

    trace = mlflow.get_trace(span.trace_id)
    judge(trace=trace)

    assert len(mock_invoke_judge_model.calls) == 1
    _, prompt, _ = mock_invoke_judge_model.calls[0]

    user_msg = prompt[1]
    user_content = user_msg.content

    expected_inputs_json = json.dumps(trace_inputs, default=str, indent=2)
    expected_outputs_json = json.dumps(trace_outputs, default=str, indent=2)

    assert f"inputs: {expected_inputs_json}" in user_content
    assert f"outputs: {expected_outputs_json}" in user_content


def test_trace_field_extraction_with_expectations(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate if {{ outputs }} meets {{ expectations }} for {{ inputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace_inputs = {"question": "What is MLflow?"}
    trace_outputs = {"answer": "MLflow is an open source platform"}
    expected_answer = {"expected": "MLflow is an open source platform for managing ML lifecycle"}

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(trace_inputs)
        span.set_outputs(trace_outputs)

    trace = mlflow.get_trace(span.trace_id)

    expectation = Expectation(
        name="expected_answer",
        value=expected_answer,
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )
    mlflow.log_assessment(trace_id=span.trace_id, assessment=expectation)

    trace = mlflow.get_trace(span.trace_id)
    judge(trace=trace)

    assert len(mock_invoke_judge_model.calls) == 1
    _, prompt, _ = mock_invoke_judge_model.calls[0]

    user_msg = prompt[1]
    user_content = user_msg.content

    expected_inputs_json = json.dumps(trace_inputs, default=str, indent=2)
    expected_outputs_json = json.dumps(trace_outputs, default=str, indent=2)
    expected_expectations_json = json.dumps(
        {"expected_answer": expected_answer}, default=str, indent=2
    )

    assert f"inputs: {expected_inputs_json}" in user_content
    assert f"outputs: {expected_outputs_json}" in user_content
    assert f"expectations: {expected_expectations_json}" in user_content


def test_trace_field_extraction_with_multiple_expectations(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate if {{ outputs }} meets {{ expectations }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace_outputs = {"answer": "MLflow is an open source platform"}

    with mlflow.start_span(name="test_span") as span:
        span.set_outputs(trace_outputs)

    trace = mlflow.get_trace(span.trace_id)

    expectation1 = Expectation(
        name="format",
        value="Should be a complete sentence",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )
    expectation2 = Expectation(
        name="content",
        value="Should mention open source",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )

    mlflow.log_assessment(trace_id=span.trace_id, assessment=expectation1)
    mlflow.log_assessment(trace_id=span.trace_id, assessment=expectation2)

    trace = mlflow.get_trace(span.trace_id)
    judge(trace=trace)

    assert len(mock_invoke_judge_model.calls) == 1
    _, prompt, _ = mock_invoke_judge_model.calls[0]

    user_msg = prompt[1]
    user_content = user_msg.content

    expected_expectations = {
        "format": "Should be a complete sentence",
        "content": "Should mention open source",
    }
    expected_expectations_json = json.dumps(expected_expectations, default=str, indent=2)

    assert f"expectations: {expected_expectations_json}" in user_content


def test_trace_field_extraction_filters_non_human_expectations(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate if {{ outputs }} meets {{ expectations }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace_outputs = {"answer": "MLflow is an open source platform"}

    with mlflow.start_span(name="test_span") as span:
        span.set_outputs(trace_outputs)

    trace = mlflow.get_trace(span.trace_id)

    human_expectation = Expectation(
        name="ground_truth",
        value="Expected from human",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )
    llm_expectation = Expectation(
        name="llm_prediction",
        value="Expected from LLM",
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE),
    )
    code_expectation = Expectation(
        name="code_prediction",
        value="Expected from code",
        source=AssessmentSource(source_type=AssessmentSourceType.CODE),
    )

    mlflow.log_assessment(trace_id=span.trace_id, assessment=human_expectation)
    mlflow.log_assessment(trace_id=span.trace_id, assessment=llm_expectation)
    mlflow.log_assessment(trace_id=span.trace_id, assessment=code_expectation)

    trace = mlflow.get_trace(span.trace_id)
    judge(trace=trace)

    assert len(mock_invoke_judge_model.calls) == 1
    _, prompt, _ = mock_invoke_judge_model.calls[0]

    user_msg = prompt[1]
    user_content = user_msg.content

    assert "Expected from human" in user_content
    assert "Expected from LLM" not in user_content
    assert "Expected from code" not in user_content


def test_trace_with_trace_template_ignores_extraction(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate the {{ trace }} for quality",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"question": "What is MLflow?"})
        span.set_outputs({"answer": "MLflow is an open source platform"})

    trace = mlflow.get_trace(span.trace_id)
    judge(trace=trace)

    assert len(mock_invoke_judge_model.calls) == 1
    _, prompt, _ = mock_invoke_judge_model.calls[0]

    # Now prompt is a list of ChatMessages
    assert isinstance(prompt, list)
    assert len(prompt) == 2
    assert prompt[0].role == "system"
    assert "analyze a trace" in prompt[0].content.lower()


def test_field_based_template_with_trace_and_explicit_inputs(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate if {{ inputs }} matches {{ outputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace_inputs = {"question": "What is in the trace?"}
    trace_outputs = {"answer": "Trace answer"}
    explicit_inputs = {"question": "What is explicitly provided?"}
    explicit_outputs = {"answer": "Explicit answer"}

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(trace_inputs)
        span.set_outputs(trace_outputs)

    trace = mlflow.get_trace(span.trace_id)

    judge(trace=trace, inputs=explicit_inputs, outputs=explicit_outputs)

    assert len(mock_invoke_judge_model.calls) == 1
    model_uri, prompt, assessment_name = mock_invoke_judge_model.calls[0]
    messages = prompt

    assert isinstance(messages, list)
    assert len(messages) == 2

    user_message = messages[1].content
    assert "explicitly provided" in user_message
    assert "Explicit answer" in user_message
    assert "in the trace" not in user_message
    assert "Trace answer" not in user_message


def test_field_based_template_extracts_missing_fields_from_trace(
    mock_invoke_judge_model,
):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate if {{ inputs }} matches {{ outputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace_inputs = {"question": "From trace"}
    trace_outputs = {"answer": "Trace output"}
    explicit_inputs = {"question": "Explicitly provided"}

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(trace_inputs)
        span.set_outputs(trace_outputs)

    trace = mlflow.get_trace(span.trace_id)

    judge(trace=trace, inputs=explicit_inputs)

    assert len(mock_invoke_judge_model.calls) == 1
    _, prompt, _ = mock_invoke_judge_model.calls[0]
    messages = prompt

    user_message = messages[1].content
    assert "Explicitly provided" in user_message
    assert "Trace output" in user_message


def test_trace_based_template_with_additional_inputs(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate the {{ trace }} considering the reference {{ inputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    additional_inputs = {"reference": "This is the expected behavior"}

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"question": "What is MLflow?"})
        span.set_outputs({"answer": "MLflow is an ML platform"})

    trace = mlflow.get_trace(span.trace_id)

    judge(trace=trace, inputs=additional_inputs)

    assert len(mock_invoke_judge_model.calls) == 1
    _, prompt, _ = mock_invoke_judge_model.calls[0]

    # Now prompt is a list of ChatMessages
    assert isinstance(prompt, list)
    assert len(prompt) == 2
    assert prompt[0].role == "system"
    assert "analyze a trace" in prompt[0].content.lower()

    # Check that inputs are in the user message
    user_content = prompt[1].content
    assert prompt[1].role == "user"
    expected_inputs_json = json.dumps(additional_inputs, default=str, indent=2)
    assert expected_inputs_json in user_content
    assert "reference" in user_content
    assert "This is the expected behavior" in user_content

    # Template variable should still be in system message
    assert "{{ inputs }}" in prompt[0].content


def test_mixed_template_validation_allows_trace_with_fields():
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate {{ trace }} against {{ inputs }} and {{ outputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    assert judge.template_variables == {"trace", "inputs", "outputs"}


def test_mixed_trace_and_fields_template_comprehensive(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions=(
            "Evaluate the {{ trace }} considering the reference {{ inputs }}, "
            "expected {{ outputs }}, and ground truth {{ expectations }}"
        ),
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    assert judge.template_variables == {"trace", "inputs", "outputs", "expectations"}

    trace_inputs = {"question": "What is MLflow?"}
    trace_outputs = {"answer": "MLflow is an open source platform"}

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(trace_inputs)
        span.set_outputs(trace_outputs)

    trace = mlflow.get_trace(span.trace_id)

    additional_inputs = {"reference": "This is the expected input format"}
    additional_outputs = {"expected_format": "JSON with answer field"}
    additional_expectations = {"criteria": "Answer must mention ML lifecycle"}

    judge(
        trace=trace,
        inputs=additional_inputs,
        outputs=additional_outputs,
        expectations=additional_expectations,
    )

    assert len(mock_invoke_judge_model.calls) == 1
    _, prompt, _ = mock_invoke_judge_model.calls[0]

    # Now prompt is a list of ChatMessages
    assert isinstance(prompt, list)
    assert len(prompt) == 2
    assert prompt[0].role == "system"
    assert "analyze a trace" in prompt[0].content.lower()

    # Check that all field values are in the user message
    user_content = prompt[1].content
    assert prompt[1].role == "user"

    expected_inputs_json = json.dumps(additional_inputs, default=str, indent=2)
    expected_outputs_json = json.dumps(additional_outputs, default=str, indent=2)
    expected_expectations_json = json.dumps(additional_expectations, default=str, indent=2)

    assert expected_inputs_json in user_content
    assert expected_outputs_json in user_content
    assert expected_expectations_json in user_content

    assert "reference" in user_content
    assert "This is the expected input format" in user_content
    assert "expected_format" in user_content
    assert "JSON with answer field" in user_content
    assert "criteria" in user_content
    assert "Answer must mention ML lifecycle" in user_content

    # Template variables should be in system message
    assert "{{ inputs }}" in prompt[0].content
    assert "{{ outputs }}" in prompt[0].content
    assert "{{ expectations }}" in prompt[0].content
    assert "{{ trace }}" in prompt[0].content


@pytest.mark.parametrize(
    "exception",
    [
        litellm.ContextWindowExceededError("Context exceeded", "gpt-4", "openai"),
        litellm.BadRequestError("maximum context length is exceeded", "gpt-4", "openai"),
    ],
)
def test_context_window_error_removes_tool_calls_and_retries(exception, monkeypatch, mock_trace):
    exception_raised = False
    captured_error_messages = None
    captured_retry_messages = None

    def mock_completion(**kwargs):
        nonlocal exception_raised
        nonlocal captured_error_messages
        nonlocal captured_retry_messages

        if len(kwargs["messages"]) >= 8 and not exception_raised:
            captured_error_messages = kwargs["messages"]
            exception_raised = True
            raise exception

        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock()]
        if exception_raised:
            captured_retry_messages = kwargs["messages"]
            mock_response.choices[0].message = litellm.Message(
                role="assistant",
                content='{"result": "pass", "rationale": "Test passed"}',
                tool_calls=None,
            )
            mock_response._hidden_params = {"response_cost": 0.05}
        else:
            call_id = f"call_{len(kwargs['messages'])}"
            mock_response.choices[0].message = litellm.Message(
                role="assistant",
                content=None,
                tool_calls=[{"id": call_id, "function": {"name": "get_span", "arguments": "{}"}}],
            )
            mock_response._hidden_params = {"response_cost": 0.05}
        return mock_response

    monkeypatch.setattr("litellm.completion", mock_completion)
    monkeypatch.setattr("litellm.token_counter", lambda model, messages: len(messages) * 20)
    monkeypatch.setattr("litellm.get_max_tokens", lambda model: 120)

    judge = make_judge(
        name="test", instructions="test {{inputs}}", feedback_value_type=str, model="openai:/gpt-4"
    )
    judge(inputs={"input": "test"}, outputs={"output": "test"}, trace=mock_trace)

    # Verify pruning happened; we expect that 2 messages were removed (one tool call pair consisting
    # of 1. assistant message and 2. tool call result message)
    assert captured_retry_messages == captured_error_messages[:2] + captured_error_messages[4:8]


def test_non_context_error_does_not_trigger_pruning(monkeypatch):
    def mock_completion(**kwargs):
        raise Exception("some other error")

    monkeypatch.setattr("litellm.completion", mock_completion)

    judge = make_judge(
        name="test_judge",
        instructions="Check if {{inputs}} is correct",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )
    with pytest.raises(MlflowException, match="some other error"):
        judge(inputs={"input": "test"}, outputs={"output": "test"})


def test_trace_template_with_expectations_extracts_correctly(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Analyze the {{ trace }} to see if it meets {{ expectations }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace_inputs = {"question": "What is MLflow?"}
    trace_outputs = {"answer": "MLflow is an open source platform"}

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(trace_inputs)
        span.set_outputs(trace_outputs)

    trace = mlflow.get_trace(span.trace_id)

    expectation = Expectation(
        name="accuracy",
        value="Should mention ML lifecycle management",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )
    mlflow.log_assessment(trace_id=span.trace_id, assessment=expectation)

    trace = mlflow.get_trace(span.trace_id)

    result = judge(trace=trace)

    assert result is not None
    assert mock_invoke_judge_model.captured_args["trace"] == trace

    prompt = mock_invoke_judge_model.captured_args["prompt"]
    assert isinstance(prompt, list)
    assert len(prompt) == 2

    system_msg = prompt[0]
    assert system_msg.role == "system"
    assert "{{ expectations }}" in system_msg.content
    assert "Analyze the {{ trace }} to see if it meets {{ expectations }}" in system_msg.content

    user_msg = prompt[1]
    assert user_msg.role == "user"
    assert "expectations:" in user_msg.content
    assert "Should mention ML lifecycle management" in user_msg.content


def test_trace_template_with_outputs_not_interpolated(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions=(
            "Check the {{ trace }} and ensure {{ outputs }} is valid. REPEAT: {{ outputs }}"
        ),
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"q": "test"})
        span.set_outputs({"a": "test"})

    trace = mlflow.get_trace(span.trace_id)

    explicit_outputs = {"result": "test output with special chars: {}, []"}
    judge(trace=trace, outputs=explicit_outputs)

    prompt = mock_invoke_judge_model.captured_args["prompt"]
    assert isinstance(prompt, list)
    assert len(prompt) == 2

    system_msg = prompt[0]
    assert system_msg.role == "system"
    assert "{{ outputs }}" in system_msg.content
    assert (
        "Check the {{ trace }} and ensure {{ outputs }} is valid. REPEAT: {{ outputs }}"
        in system_msg.content
    )
    assert (
        "Check the {{ trace }} and ensure test output with special chars" not in system_msg.content
    )

    user_msg = prompt[1]
    assert user_msg.role == "user"
    assert "outputs:" in user_msg.content
    assert "test output with special chars: {}, []" in user_msg.content


def test_trace_template_field_values_appended_not_interpolated(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Analyze {{ trace }} with {{ inputs }}, {{ outputs }}, and {{ expectations }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"q": "from trace"})
        span.set_outputs({"a": "from trace"})

    trace = mlflow.get_trace(span.trace_id)

    expectation = Expectation(
        name="test_exp",
        value="expected value",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )
    mlflow.log_assessment(trace_id=span.trace_id, assessment=expectation)

    trace = mlflow.get_trace(span.trace_id)

    explicit_inputs = {"custom": "explicit input"}
    judge(trace=trace, inputs=explicit_inputs)

    prompt = mock_invoke_judge_model.captured_args["prompt"]
    assert isinstance(prompt, list)
    assert len(prompt) == 2

    system_msg = prompt[0]
    assert system_msg.role == "system"
    assert "{{ trace }}" in system_msg.content
    assert "{{ inputs }}" in system_msg.content
    assert "{{ outputs }}" in system_msg.content
    assert "{{ expectations }}" in system_msg.content
    assert (
        "Analyze {{ trace }} with {{ inputs }}, {{ outputs }}, and {{ expectations }}"
        in system_msg.content
    )

    user_msg = prompt[1]
    assert user_msg.role == "user"
    assert "inputs:" in user_msg.content
    assert "explicit input" in user_msg.content
    assert "outputs:" in user_msg.content
    assert "from trace" in user_msg.content
    assert "expectations:" in user_msg.content
    assert "expected value" in user_msg.content


def test_trace_template_with_all_fields_extraction(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate {{ trace }} against {{ inputs }}, {{ outputs }}, {{ expectations }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace_inputs = {"question": "What is AI?"}
    trace_outputs = {"answer": "Artificial Intelligence"}

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(trace_inputs)
        span.set_outputs(trace_outputs)

    trace = mlflow.get_trace(span.trace_id)

    exp1 = Expectation(
        name="clarity",
        value="Should be clear",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )
    exp2 = Expectation(
        name="accuracy",
        value="Should be accurate",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
    )
    mlflow.log_assessment(trace_id=span.trace_id, assessment=exp1)
    mlflow.log_assessment(trace_id=span.trace_id, assessment=exp2)

    trace = mlflow.get_trace(span.trace_id)

    judge(trace=trace)

    prompt = mock_invoke_judge_model.captured_args["prompt"]
    assert isinstance(prompt, list)
    assert len(prompt) == 2

    system_msg = prompt[0]
    assert system_msg.role == "system"
    assert (
        "Evaluate {{ trace }} against {{ inputs }}, {{ outputs }}, {{ expectations }}"
        in system_msg.content
    )

    user_msg = prompt[1]
    assert user_msg.role == "user"
    assert "What is AI?" in user_msg.content
    assert "Artificial Intelligence" in user_msg.content
    assert "Should be clear" in user_msg.content
    assert "Should be accurate" in user_msg.content
    assert "inputs:" in user_msg.content
    assert "outputs:" in user_msg.content
    assert "expectations:" in user_msg.content


def test_trace_only_template_uses_two_messages_with_empty_user(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Analyze this {{ trace }} for quality",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"q": "test"})
        span.set_outputs({"a": "test"})

    trace = mlflow.get_trace(span.trace_id)

    judge(trace=trace)

    prompt = mock_invoke_judge_model.captured_args["prompt"]

    assert isinstance(prompt, list)
    assert len(prompt) == 2

    system_msg = prompt[0]
    assert system_msg.role == "system"
    assert "Analyze this {{ trace }} for quality" in system_msg.content
    assert "expert judge" in system_msg.content

    user_msg = prompt[1]
    assert user_msg.role == "user"
    assert (
        user_msg.content == "Follow the instructions from the first message"
    )  # Placeholder user message for trace-only


def test_no_warning_when_extracting_fields_from_trace(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate {{ inputs }} and {{ outputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"question": "What is AI?"})
        span.set_outputs({"answer": "Artificial Intelligence"})

    trace = mlflow.get_trace(span.trace_id)

    # Call judge with only trace - should extract inputs/outputs from it
    with mock.patch("mlflow.genai.judges.instructions_judge._logger.warning") as mock_warning:
        judge(trace=trace)

        # Should NOT warn about trace being unused - it's used for extraction
        mock_warning.assert_not_called()

    # Verify the extraction worked
    prompt = mock_invoke_judge_model.captured_args["prompt"]
    assert "What is AI?" in prompt[1].content
    assert "Artificial Intelligence" in prompt[1].content


def test_warning_shown_for_explicitly_provided_unused_fields(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate {{ inputs }} only",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    with mock.patch("mlflow.genai.judges.instructions_judge._logger.warning") as mock_warning:
        judge(inputs="What is AI?", outputs="This output is not used by the template")

        mock_warning.assert_called_once()
        warning_message = mock_warning.call_args[0][0]
        assert "outputs" in warning_message
        assert "not used by this judge" in warning_message


def test_no_warning_for_trace_based_judge_with_extra_fields(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate {{ trace }} for quality",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    span_mock = Span(
        OTelReadableSpan(
            name="test_span",
            context=build_otel_context(
                trace_id=12345678,
                span_id=12345678,
            ),
        )
    )
    trace = Trace(
        info=TraceInfo(
            trace_id="test_trace",
            trace_location=TraceLocation.from_experiment_id("0"),
            request_time=1234567890,
            execution_duration=100,
            state=TraceState.OK,
            trace_metadata={},
            tags={},
        ),
        data=TraceData(spans=[span_mock]),
    )

    with mock.patch("mlflow.genai.judges.instructions_judge._logger.warning") as mock_warning:
        judge(
            trace=trace,
            inputs="These inputs are extracted from trace",
            outputs="These outputs are extracted from trace",
            expectations={"ground_truth": "These expectations are extracted from trace"},
        )

        mock_warning.assert_not_called()


def test_no_duplicate_output_fields_in_system_message():
    field_judge = make_judge(
        name="field_judge",
        instructions="Evaluate {{ inputs }} and {{ outputs }} for quality",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    field_system_msg = field_judge._build_system_message(is_trace_based=False)

    assert field_system_msg.count('"result"') == 1
    assert field_system_msg.count('"rationale"') == 1

    assert (
        field_system_msg.count("Please provide your assessment in the following JSON format") == 1
    )

    trace_judge = make_judge(
        name="trace_judge",
        instructions="Evaluate {{ trace }} for quality",
        feedback_value_type=Literal["good", "bad", "neutral"],
        model="openai:/gpt-4",
    )

    trace_system_msg = trace_judge._build_system_message(is_trace_based=True)

    assert trace_system_msg.count("- result (Literal['good', 'bad', 'neutral'])") == 1
    assert trace_system_msg.count("- rationale (str):") == 1

    assert "Please provide your assessment in the following JSON format" not in trace_system_msg


def test_instructions_judge_repr():
    # Test short instructions that fit within display limit
    short_instructions = "Check {{ outputs }}"
    judge = make_judge(
        name="test_judge",
        instructions=short_instructions,
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    repr_str = repr(judge)
    assert "InstructionsJudge" in repr_str
    assert "name='test_judge'" in repr_str
    assert "model='openai:/gpt-4'" in repr_str
    assert f"instructions='{short_instructions}'" in repr_str
    assert "template_variables=['outputs']" in repr_str

    # Test long instructions that exceed PROMPT_TEXT_DISPLAY_LIMIT (30 chars)
    long_instructions = (
        "This is a very long instruction that will be truncated {{ inputs }} and {{ outputs }}"
    )
    judge_long = make_judge(
        name="long_judge",
        instructions=long_instructions,
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    repr_long = repr(judge_long)
    assert "InstructionsJudge" in repr_long
    assert "name='long_judge'" in repr_long
    assert "model='openai:/gpt-4'" in repr_long
    # Should show first 30 characters + "..."
    assert "instructions='This is a very long instructio..." in repr_long
    assert "template_variables=['inputs', 'outputs']" in repr_long


def test_make_judge_with_feedback_value_type(monkeypatch):
    captured_response_format = None

    def mock_litellm_completion(**kwargs):
        nonlocal captured_response_format
        captured_response_format = kwargs.get("response_format")

        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock()]
        mock_response.choices[0].message = litellm.Message(
            role="assistant",
            content='{"result": 5, "rationale": "Excellent quality work"}',
            tool_calls=None,
        )
        mock_response._hidden_params = None
        return mock_response

    monkeypatch.setattr("litellm.completion", mock_litellm_completion)

    judge = make_judge(
        name="test_judge",
        instructions="Rate the quality of {{ outputs }} on a scale of 1-5",
        model="openai:/gpt-4",
        feedback_value_type=int,
    )

    result = judge(outputs={"text": "Great work!"})

    # Verify response_format was correctly captured by litellm.completion
    assert captured_response_format is not None
    assert issubclass(captured_response_format, pydantic.BaseModel)

    model_fields = captured_response_format.model_fields
    assert "result" in model_fields
    assert "rationale" in model_fields

    assert model_fields["result"].annotation == int
    assert model_fields["rationale"].annotation == str

    assert result.value == 5
    assert result.rationale == "Excellent quality work"


def test_make_judge_serialization_with_feedback_value_type():
    # Test with int type
    judge_int = make_judge(
        name="int_judge",
        instructions="Rate {{ outputs }} from 1-10",
        model="openai:/gpt-4",
        feedback_value_type=int,
    )

    serialized = judge_int.model_dump()
    assert "instructions_judge_pydantic_data" in serialized
    assert "feedback_value_type" in serialized["instructions_judge_pydantic_data"]
    assert serialized["instructions_judge_pydantic_data"]["feedback_value_type"] == {
        "type": "integer",
        "title": "Result",
    }

    restored_judge = Scorer.model_validate(serialized)
    assert isinstance(restored_judge, InstructionsJudge)
    assert restored_judge.name == "int_judge"
    assert restored_judge._feedback_value_type == int

    # Test with bool type
    judge_bool = make_judge(
        name="bool_judge",
        instructions="Is {{ outputs }} correct?",
        model="openai:/gpt-4",
        feedback_value_type=bool,
    )

    serialized_bool = judge_bool.model_dump()
    assert serialized_bool["instructions_judge_pydantic_data"]["feedback_value_type"] == {
        "type": "boolean",
        "title": "Result",
    }

    restored_bool = Scorer.model_validate(serialized_bool)
    assert restored_bool._feedback_value_type == bool

    # Test with Literal type
    judge_literal = make_judge(
        name="literal_judge",
        instructions="Rate {{ outputs }} quality",
        model="openai:/gpt-4",
        feedback_value_type=Literal["good", "bad", "neutral"],
    )

    serialized_literal = judge_literal.model_dump()
    assert serialized_literal["instructions_judge_pydantic_data"]["feedback_value_type"] == {
        "type": "string",
        "enum": ["good", "bad", "neutral"],
        "title": "Result",
    }

    restored_literal = Scorer.model_validate(serialized_literal)
    assert typing.get_origin(restored_literal._feedback_value_type) is Literal
    assert set(typing.get_args(restored_literal._feedback_value_type)) == {"good", "bad", "neutral"}

    # Test with dict[str, int] type
    judge_dict = make_judge(
        name="dict_judge",
        instructions="Rate {{ outputs }} with scores",
        model="openai:/gpt-4",
        feedback_value_type=dict[str, int],
    )

    serialized_dict = judge_dict.model_dump()
    assert serialized_dict["instructions_judge_pydantic_data"]["feedback_value_type"] == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
        "title": "Result",
    }

    restored_dict = Scorer.model_validate(serialized_dict)
    assert typing.get_origin(restored_dict._feedback_value_type) is dict
    assert typing.get_args(restored_dict._feedback_value_type) == (str, int)

    # Test with list[str] type
    judge_list = make_judge(
        name="list_judge",
        instructions="List issues in {{ outputs }}",
        model="openai:/gpt-4",
        feedback_value_type=list[str],
    )

    serialized_list = judge_list.model_dump()
    assert serialized_list["instructions_judge_pydantic_data"]["feedback_value_type"] == {
        "type": "array",
        "items": {"type": "string"},
        "title": "Result",
    }

    restored_list = Scorer.model_validate(serialized_list)
    assert typing.get_origin(restored_list._feedback_value_type) is list
    assert typing.get_args(restored_list._feedback_value_type) == (str,)


def test_judge_with_literal_type_serialization():
    literal_type = Literal["good", "bad"]
    judge = make_judge(
        name="test_judge",
        instructions="Rate the response as {{ inputs }}",
        feedback_value_type=literal_type,
        model="databricks:/databricks-meta-llama-3-1-70b-instruct",
    )

    # Test serialization
    serialized = InstructionsJudge._serialize_feedback_value_type(literal_type)
    assert "enum" in serialized
    assert serialized["enum"] == ["good", "bad"]

    # Test model validate
    dumped = judge.model_dump()
    restored = Scorer.model_validate(dumped)
    assert restored.name == "test_judge"
    assert restored._feedback_value_type is not None

    # Test register
    registered = judge.register()
    assert registered.name == "test_judge"
    assert registered._feedback_value_type is not None


def test_make_judge_validates_feedback_value_type():
    # Valid types should work
    make_judge(
        name="int_judge",
        instructions="Rate {{ outputs }}",
        model="openai:/gpt-4",
        feedback_value_type=int,
    )
    make_judge(
        name="str_judge",
        instructions="Evaluate {{ outputs }}",
        model="openai:/gpt-4",
        feedback_value_type=str,
    )
    make_judge(
        name="dict_judge",
        instructions="Rate {{ outputs }}",
        model="openai:/gpt-4",
        feedback_value_type=dict[str, int],
    )
    make_judge(
        name="list_judge",
        instructions="List {{ outputs }}",
        model="openai:/gpt-4",
        feedback_value_type=list[str],
    )

    # Unsupported types should be rejected
    class CustomModel(pydantic.BaseModel):
        score: int

    with pytest.raises(
        MlflowException,
        match=r"Unsupported feedback_value_type: .+CustomModel",
    ):
        make_judge(
            name="invalid_judge",
            instructions="Rate {{ outputs }}",
            model="openai:/gpt-4",
            feedback_value_type=CustomModel,
        )

    with pytest.raises(
        MlflowException,
        match=r"The `feedback_value_type` argument does not support a dict type",
    ):
        make_judge(
            name="invalid_judge",
            instructions="Rate {{ outputs }}",
            model="openai:/gpt-4",
            feedback_value_type=dict[str, CustomModel],
        )

    with pytest.raises(
        MlflowException,
        match=r"The `feedback_value_type` argument does not support a list type",
    ):
        make_judge(
            name="invalid_judge",
            instructions="Rate {{ outputs }}",
            model="openai:/gpt-4",
            feedback_value_type=list[CustomModel],
        )


def test_make_judge_with_default_feedback_value_type(monkeypatch):
    # Test that feedback_value_type defaults to str when omitted
    captured_response_format = None

    def mock_litellm_completion(**kwargs):
        nonlocal captured_response_format
        captured_response_format = kwargs.get("response_format")

        mock_response = mock.Mock()
        mock_response.choices = [mock.Mock()]
        mock_response.choices[0].message = litellm.Message(
            role="assistant",
            content='{"result": "Good quality", "rationale": "The response is clear and accurate"}',
            tool_calls=None,
        )
        mock_response._hidden_params = None
        return mock_response

    monkeypatch.setattr("litellm.completion", mock_litellm_completion)

    judge = make_judge(
        name="default_judge",
        instructions="Evaluate {{ outputs }}",
        model="openai:/gpt-4",
    )

    # Verify serialization includes the default str type
    serialized = judge.model_dump()
    assert "instructions_judge_pydantic_data" in serialized
    assert "feedback_value_type" in serialized["instructions_judge_pydantic_data"]
    assert serialized["instructions_judge_pydantic_data"]["feedback_value_type"] == {
        "type": "string",
        "title": "Result",
    }

    # Verify execution with default str type
    result = judge(outputs={"text": "Great work!"})

    assert captured_response_format is not None
    assert issubclass(captured_response_format, pydantic.BaseModel)

    model_fields = captured_response_format.model_fields
    assert "result" in model_fields
    assert "rationale" in model_fields

    assert model_fields["result"].annotation == str
    assert model_fields["rationale"].annotation == str

    assert result.value == "Good quality"
    assert result.rationale == "The response is clear and accurate"


def test_conversation_template_variable_extraction():
    judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate the {{ conversation }} for quality",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    assert judge.template_variables == {"conversation"}


def test_is_session_level_scorer_property():
    """Test that is_session_level_scorer property returns True when conversation is in template
    variables.
    """
    conversation_judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate {{ conversation }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    assert conversation_judge.is_session_level_scorer is True

    regular_judge = make_judge(
        name="regular_judge",
        instructions="Evaluate {{ outputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    assert regular_judge.is_session_level_scorer is False


def test_conversation_with_expectations_allowed():
    judge = make_judge(
        name="conversation_expectations_judge",
        instructions="Evaluate {{ conversation }} against {{ expectations }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    assert judge.template_variables == {"conversation", "expectations"}


def test_conversation_with_other_variables_rejected():
    with pytest.raises(
        MlflowException,
        match=(
            "Instructions template must not contain any template variables "
            "other than {{ expectations }} if {{ conversation }} is provided"
        ),
    ):
        make_judge(
            name="invalid_judge",
            instructions="Evaluate {{ conversation }} and {{ inputs }}",
            feedback_value_type=str,
            model="openai:/gpt-4",
        )

    with pytest.raises(
        MlflowException,
        match=(
            "Instructions template must not contain any template variables "
            "other than {{ expectations }} if {{ conversation }} is provided"
        ),
    ):
        make_judge(
            name="invalid_judge",
            instructions="Evaluate {{ conversation }} and {{ outputs }}",
            feedback_value_type=str,
            model="openai:/gpt-4",
        )

    with pytest.raises(
        MlflowException,
        match=(
            "Instructions template must not contain any template variables "
            "other than {{ expectations }} if {{ conversation }} is provided"
        ),
    ):
        make_judge(
            name="invalid_judge",
            instructions="Evaluate {{ conversation }} and {{ trace }}",
            feedback_value_type=str,
            model="openai:/gpt-4",
        )


def test_session_validation_type_error():
    judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate {{ conversation }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    with pytest.raises(MlflowException, match="'session' must be a list of Trace objects, got str"):
        judge(session="not a list")


def test_session_validation_not_all_traces():
    judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate {{ conversation }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    with pytest.raises(MlflowException, match="All elements in 'session' must be Trace objects"):
        judge(session=["not a trace", "also not a trace"])


def create_trace_with_session(
    trace_id: str,
    session_id: str | None = None,
    inputs: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    timestamp_ms: int = 1000,
):
    """Helper function to create a trace, optionally with a session ID."""
    trace_metadata = {
        "mlflow.trace_schema.version": "2",
        "mlflow.traceInputs": json.dumps(inputs or {}),
        "mlflow.traceOutputs": json.dumps(outputs or {}),
    }
    if session_id is not None:
        trace_metadata[TraceMetadataKey.TRACE_SESSION] = session_id

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=timestamp_ms,  # timestamp_ms property returns request_time
        execution_duration=1000,
        state=TraceState.OK,
        trace_metadata=trace_metadata,
        tags={
            "mlflow.traceName": "test_trace",
            "mlflow.source.name": "test",
            "mlflow.source.type": "LOCAL",
        },
    )
    spans = [
        create_test_span(
            span_id=1,
            parent_id=None,
            name="root_span",
            inputs=inputs or {},
            outputs=outputs or {},
            span_type=SpanType.CHAIN,
        ),
    ]
    trace_data = TraceData(spans=spans)
    return Trace(info=trace_info, data=trace_data)


def test_validate_session_missing_session_id():
    judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate {{ conversation }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace_without_session = create_trace_with_session("trace-1", session_id=None)

    with pytest.raises(
        MlflowException,
        match="All traces in 'session' must have a session_id",
    ):
        judge._validate_session([trace_without_session])


def test_validate_session_different_sessions():
    """Test that _validate_session raises error and shows trace_ids grouped by session_id
    when traces belong to different sessions. Also verifies truncation when there are more than 3
    traces.
    """
    judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate {{ conversation }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    # Create traces: session-1 has 5 traces (will be truncated),
    # session-2 has 2 traces, session-3 has 1 trace
    trace1 = create_trace_with_session("trace-1", "session-1")
    trace2 = create_trace_with_session("trace-2", "session-1")
    trace3 = create_trace_with_session("trace-3", "session-1")
    trace4 = create_trace_with_session("trace-4", "session-1")
    trace5 = create_trace_with_session("trace-5", "session-1")
    trace6 = create_trace_with_session("trace-6", "session-2")
    trace7 = create_trace_with_session("trace-7", "session-2")
    trace8 = create_trace_with_session("trace-8", "session-3")

    with pytest.raises(
        MlflowException,
        match="All traces in 'session' must belong to the same session",
    ) as exception_info:
        judge._validate_session([trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8])

    # Verify the exception message includes trace_ids grouped by session_id and truncates when >3
    error_message = str(exception_info.value)
    expected_message = (
        "All traces in 'session' must belong to the same session. "
        "Found 3 different session(s):\n"
        "session_id 'session-1': trace_ids ['trace-1', 'trace-2', 'trace-3'] and 2 more traces\n"
        "session_id 'session-2': trace_ids ['trace-6', 'trace-7']\n"
        "session_id 'session-3': trace_ids ['trace-8']"
    )
    assert error_message == expected_message


def test_validate_session_same_session():
    judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate {{ conversation }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace1 = create_trace_with_session("trace-1", "session-1")
    trace2 = create_trace_with_session("trace-2", "session-1")

    # Should not raise
    judge._validate_session([trace1, trace2])


def test_conversation_extraction_from_session(mock_invoke_judge_model):
    judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate {{ conversation }} for quality",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace1 = create_trace_with_session(
        "trace-1",
        "session-1",
        inputs={"question": "What is MLflow?"},
        outputs={"answer": "MLflow is an open source platform"},
        timestamp_ms=1000,
    )
    trace2 = create_trace_with_session(
        "trace-2",
        "session-1",
        inputs={"question": "How do I use it?"},
        outputs={"answer": "You can use mlflow.start_run()"},
        timestamp_ms=2000,
    )

    result = judge(session=[trace1, trace2])

    assert isinstance(result, Feedback)
    assert len(mock_invoke_judge_model.calls) == 1
    _, prompt, _ = mock_invoke_judge_model.calls[0]

    # Check that conversation is in the user message
    user_msg = prompt[1]
    expected_content = """conversation: [
  {
    "role": "user",
    "content": "{'question': 'What is MLflow?'}"
  },
  {
    "role": "assistant",
    "content": "{\\"answer\\": \\"MLflow is an open source platform\\"}"
  },
  {
    "role": "user",
    "content": "{'question': 'How do I use it?'}"
  },
  {
    "role": "assistant",
    "content": "{\\"answer\\": \\"You can use mlflow.start_run()\\"}"
  }
]"""
    assert user_msg.content == expected_content


def test_conversation_extraction_chronological_order(mock_invoke_judge_model):
    judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate {{ conversation }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    # Create traces out of order
    trace2 = create_trace_with_session(
        "trace-2",
        "session-1",
        inputs={"question": "Second question"},
        outputs={"answer": "Second answer"},
        timestamp_ms=2000,
    )
    trace1 = create_trace_with_session(
        "trace-1",
        "session-1",
        inputs={"question": "First question"},
        outputs={"answer": "First answer"},
        timestamp_ms=1000,
    )

    judge(session=[trace2, trace1])  # Pass in reverse order

    _, prompt, _ = mock_invoke_judge_model.calls[0]
    user_msg = prompt[1]
    content = user_msg.content

    # Check that messages are in chronological order (first question before second)
    first_pos = content.find("First question")
    second_pos = content.find("Second question")
    assert first_pos < second_pos


def test_conversation_with_expectations(mock_invoke_judge_model):
    judge = make_judge(
        name="conversation_expectations_judge",
        instructions="Evaluate {{ conversation }} against {{ expectations }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace1 = create_trace_with_session(
        "trace-1",
        "session-1",
        inputs={"question": "What is MLflow?"},
        outputs={"answer": "MLflow is a platform"},
        timestamp_ms=1000,
    )

    expectations = {"criteria": "Should be accurate and helpful"}

    result = judge(session=[trace1], expectations=expectations)

    assert isinstance(result, Feedback)
    _, prompt, _ = mock_invoke_judge_model.calls[0]
    user_msg = prompt[1]

    expected_content = """conversation: [
  {
    "role": "user",
    "content": "{'question': 'What is MLflow?'}"
  },
  {
    "role": "assistant",
    "content": "{\\"answer\\": \\"MLflow is a platform\\"}"
  }
]
expectations: {
  "criteria": "Should be accurate and helpful"
}"""
    assert user_msg.content == expected_content


def test_conversation_with_session_level_expectations(mock_invoke_judge_model):
    judge = make_judge(
        name="conversation_expectations_judge",
        instructions="Evaluate {{ conversation }} against {{ expectations }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    session_id = "test-session"

    with mlflow.start_span(name="turn_0") as span:
        span.set_inputs({"question": "What is MLflow?"})
        span.set_outputs({"answer": "MLflow is a platform"})
        mlflow.update_current_trace(metadata={TraceMetadataKey.TRACE_SESSION: session_id})

    trace_id = span.trace_id

    expectation = Expectation(
        name="accuracy",
        value="Should provide accurate information",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN),
        metadata={TraceMetadataKey.TRACE_SESSION: session_id},
    )
    mlflow.log_assessment(trace_id=trace_id, assessment=expectation)

    trace = mlflow.get_trace(trace_id)

    result = judge(session=[trace])

    assert isinstance(result, Feedback)
    _, prompt, _ = mock_invoke_judge_model.calls[0]
    user_msg = prompt[1]

    expected_content = """conversation: [
  {
    "role": "user",
    "content": "{'question': 'What is MLflow?'}"
  },
  {
    "role": "assistant",
    "content": "{\\"answer\\": \\"MLflow is a platform\\"}"
  }
]
expectations: {
  "accuracy": "Should provide accurate information"
}"""
    assert user_msg.content == expected_content


def test_conversation_missing_session():
    judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate {{ conversation }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    with pytest.raises(
        MlflowException, match="Must specify 'session' - required by template variables"
    ):
        judge()


def test_conversation_empty_session():
    judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate {{ conversation }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    with pytest.raises(
        MlflowException, match="Must specify 'session' - required by template variables"
    ):
        judge(session=[])


def test_conversation_with_empty_inputs_or_outputs(mock_invoke_judge_model):
    judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate {{ conversation }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace1 = create_trace_with_session(
        "trace-1",
        "session-1",
        inputs={},  # Empty inputs
        outputs={"answer": "Valid answer"},
        timestamp_ms=1000,
    )
    trace2 = create_trace_with_session(
        "trace-2",
        "session-1",
        inputs={"question": "Valid question"},
        outputs={},  # Empty outputs
        timestamp_ms=2000,
    )

    judge(session=[trace1, trace2])

    _, prompt, _ = mock_invoke_judge_model.calls[0]
    user_msg = prompt[1]

    # Should only contain non-empty messages
    expected_content = """conversation: [
  {
    "role": "assistant",
    "content": "{\\"answer\\": \\"Valid answer\\"}"
  },
  {
    "role": "user",
    "content": "{'question': 'Valid question'}"
  }
]"""
    assert user_msg.content == expected_content


def test_conversation_unused_parameter_warning(mock_invoke_judge_model):
    judge = make_judge(
        name="outputs_judge",
        instructions="Evaluate {{ outputs }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace1 = create_trace_with_session(
        "trace-1",
        "session-1",
        inputs={"question": "Test"},
        outputs={"answer": "Test answer"},
    )

    with patch("mlflow.genai.judges.instructions_judge._logger") as mock_logger:
        judge(outputs={"answer": "Test"}, session=[trace1])

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "conversation" in warning_msg or "session" in warning_msg
        assert "not used by this judge" in warning_msg


def test_conversation_no_warning_when_used(mock_invoke_judge_model):
    judge = make_judge(
        name="conversation_judge",
        instructions="Evaluate {{ conversation }}",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    trace1 = create_trace_with_session(
        "trace-1",
        "session-1",
        inputs={"question": "Test"},
        outputs={"answer": "Test answer"},
    )

    with patch("mlflow.genai.judges.instructions_judge._logger") as mock_logger:
        judge(session=[trace1])

        # Should not warn about conversation being unused
        # Check that no warnings were called, or if they were, they're not about conversation
        if mock_logger.warning.called:
            for call in mock_logger.warning.call_args_list:
                if call and call[0]:
                    warning_msg = call[0][0]
                    # Should not contain both "conversation" and "not used" together
                    if "conversation" in warning_msg.lower():
                        assert "not used" not in warning_msg.lower()


def test_instructions_judge_generate_rationale_first():
    # Test with generate_rationale_first=False (default)
    judge_default = InstructionsJudge(
        name="test_judge",
        instructions="Evaluate {{ outputs }}",
        model="openai:/gpt-4",
        feedback_value_type=str,
        generate_rationale_first=False,
    )

    # Check output fields order (default: result first, then rationale)
    output_fields_default = judge_default.get_output_fields()
    assert len(output_fields_default) == 2
    assert output_fields_default[0].name == "result"
    assert output_fields_default[1].name == "rationale"

    # Check response format field order (default: result first)
    response_format_default = judge_default._create_response_format_model()
    field_names_default = list(response_format_default.model_fields.keys())
    assert field_names_default == ["result", "rationale"]

    # Test with generate_rationale_first=True
    judge_rationale_first = InstructionsJudge(
        name="test_judge_rationale_first",
        instructions="Evaluate {{ outputs }}",
        model="openai:/gpt-4",
        feedback_value_type=Literal["good", "bad"],
        generate_rationale_first=True,
    )

    # Check output fields order (rationale first, then result)
    output_fields_rationale_first = judge_rationale_first.get_output_fields()
    assert len(output_fields_rationale_first) == 2
    assert output_fields_rationale_first[0].name == "rationale"
    assert output_fields_rationale_first[1].name == "result"

    # Check response format field order (rationale first)
    response_format_rationale_first = judge_rationale_first._create_response_format_model()
    field_names_rationale_first = list(response_format_rationale_first.model_fields.keys())
    assert field_names_rationale_first == ["rationale", "result"]

    # Verify field descriptions are correct regardless of order
    assert output_fields_default[0].value_type == str
    assert output_fields_default[1].value_type == str
    assert output_fields_rationale_first[0].value_type == str  # rationale
    assert output_fields_rationale_first[1].value_type == Literal["good", "bad"]  # result


@pytest.mark.parametrize(
    "description",
    [
        "Evaluates the conciseness of the response",  # With custom description
        None,  # Without description
    ],
)
def test_response_format_uses_generic_field_description(description):
    judge = InstructionsJudge(
        name="Conciseness" if description else "TestJudge",
        instructions="Evaluate if the output {{ outputs }} is concise",
        description=description,
        model="openai:/gpt-4",
    )

    response_format_model = judge._create_response_format_model()
    schema = response_format_model.model_json_schema()

    # The result field description should be the generic description,
    # NOT the scorer's description
    result_description = schema["properties"]["result"]["description"]
    assert result_description == _RESULT_FIELD_DESCRIPTION

    # Verify rationale field uses its own description
    rationale_description = schema["properties"]["rationale"]["description"]
    assert rationale_description == "Detailed explanation for the evaluation"

    # Also verify get_output_fields() uses generic description (used in system prompt)
    output_fields = judge.get_output_fields()
    result_field = next(f for f in output_fields if f.name == "result")
    assert result_field.description == _RESULT_FIELD_DESCRIPTION


@pytest.mark.parametrize(
    "inference_params",
    [
        {"temperature": 0.0},
        {"temperature": 1.0},
        {"max_tokens": 100},
        {"top_p": 0.95},
        {"temperature": 0.5, "max_tokens": 200, "top_p": 0.9},
    ],
)
def test_make_judge_with_inference_params(inference_params):
    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ outputs }} is formal",
        model="openai:/gpt-4",
        inference_params=inference_params,
    )

    assert judge.inference_params == inference_params
    assert judge._inference_params == inference_params

    # Verify repr includes inference_params
    repr_str = repr(judge)
    assert "inference_params=" in repr_str

    # Verify serialization includes inference_params
    dumped = judge.model_dump()
    pydantic_data = dumped["instructions_judge_pydantic_data"]
    assert pydantic_data["inference_params"] == inference_params


def test_make_judge_without_inference_params():
    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ outputs }} is formal",
        model="openai:/gpt-4",
    )

    assert judge.inference_params is None
    assert judge._inference_params is None

    # Verify repr does not include inference_params
    repr_str = repr(judge)
    assert "inference_params" not in repr_str

    # Verify serialization does not include inference_params
    dumped = judge.model_dump()
    pydantic_data = dumped["instructions_judge_pydantic_data"]
    assert "inference_params" not in pydantic_data


def test_inference_params_passed_to_invoke_judge_model(mock_invoke_judge_model):
    inference_params = {"temperature": 0.1}
    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ outputs }} is good",
        model="openai:/gpt-4",
        inference_params=inference_params,
    )

    judge(outputs="test output")

    assert mock_invoke_judge_model.captured_args.get("inference_params") == inference_params
