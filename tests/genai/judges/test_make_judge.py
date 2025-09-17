import json
import sys
import types
from dataclasses import asdict
from unittest import mock
from unittest.mock import patch

import litellm
import pandas as pd
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
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai import make_judge
from mlflow.genai.judges.instructions_judge import InstructionsJudge
from mlflow.genai.judges.instructions_judge.constants import JUDGE_BASE_PROMPT
from mlflow.genai.judges.utils import _LITELLM_PROVIDERS, _NATIVE_PROVIDERS, validate_judge_model
from mlflow.genai.scorers.base import Scorer, ScorerKind, SerializedScorer
from mlflow.genai.scorers.registry import _get_scorer_store
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
            self.output = json.dumps(output_data or {"result": True, "rationale": "Test passed"})
            self.error_message = None

    class MockManagedRAGClient:
        def __init__(self, expected_content=None, response_data=None):
            self.expected_content = expected_content
            self.response_data = response_data

        def get_chat_completions_result(self, user_prompt, system_prompt):
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

    def _mock(model_uri, prompt, assessment_name, trace=None):
        # Store call details in list format (for backward compatibility)
        calls.append((model_uri, prompt, assessment_name))

        # Store latest call details in dict format
        captured_args.update(
            {
                "model_uri": model_uri,
                "prompt": prompt,
                "assessment_name": assessment_name,
                "trace": trace,
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
        name="test_judge", instructions="Check if {{ outputs }} is formal", model="openai:/gpt-4"
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

    judge = make_judge(name="test_judge", instructions="Check if {{ outputs }} is accurate")

    assert judge.model == expected_model


def test_make_judge_with_databricks_default(monkeypatch):
    # Mock the parent module first to prevent ImportError
    mock_evals_module = types.ModuleType("databricks.agents.evals")
    monkeypatch.setitem(sys.modules, "databricks.agents.evals", mock_evals_module)

    # Then mock the judges submodule
    mock_judges_module = types.ModuleType("databricks.agents.evals.judges")
    monkeypatch.setitem(sys.modules, "databricks.agents.evals.judges", mock_judges_module)

    monkeypatch.setattr("mlflow.genai.judges.utils.is_databricks_uri", lambda x: True)

    judge = make_judge(name="test_judge", instructions="Check if {{ outputs }} is valid")

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
            name="test_judge", instructions="Check if {{ outputs }} is valid", model="databricks"
        )


@pytest.mark.parametrize("provider", _LITELLM_PROVIDERS)
def test_litellm_provider_requires_litellm(monkeypatch, provider):
    monkeypatch.setitem(sys.modules, "litellm", None)

    with pytest.raises(
        MlflowException,
        match=f"LiteLLM is required for using '{provider}' as a provider",
    ):
        make_judge(
            name="test_judge",
            instructions="Check if {{ outputs }} is valid",
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
        name="test_judge", instructions="Check if {{ outputs }} is valid", model="databricks"
    )

    result = judge(outputs={"text": "test output"})
    assert isinstance(result, Feedback)
    assert result.value is True
    assert result.rationale == "Valid output"


def test_databricks_model_handles_errors_gracefully(mock_databricks_rag_eval):
    class MockLLMResultInvalid:
        def __init__(self):
            self.output = "This is not valid JSON - maybe the model returned plain text"

    class MockClientInvalid:
        def get_chat_completions_result(self, user_prompt, system_prompt):
            return MockLLMResultInvalid()

    class MockContextInvalid:
        def build_managed_rag_client(self):
            return MockClientInvalid()

    mock_databricks_rag_eval.get_context = lambda: MockContextInvalid()

    judge = make_judge(
        name="test_judge", instructions="Check if {{ outputs }} is valid", model="databricks"
    )

    result = judge(outputs={"text": "test output"})
    assert isinstance(result, Feedback)
    assert result.error is not None
    assert "Invalid JSON response" in result.error  # NB: Non-JSON response error

    class MockLLMResultMissingField:
        def __init__(self):
            self.output = json.dumps({"rationale": "Some rationale but no result field"})

    class MockClientMissingField:
        def get_chat_completions_result(self, user_prompt, system_prompt):
            return MockLLMResultMissingField()

    class MockContextMissingField:
        def build_managed_rag_client(self):
            return MockClientMissingField()

    mock_databricks_rag_eval.get_context = lambda: MockContextMissingField()

    result = judge(outputs={"text": "test output"})
    assert isinstance(result, Feedback)
    assert result.error is not None
    assert "Response missing 'result' field" in result.error  # NB: Missing result field error

    class MockLLMResultNone:
        output = None

    class MockClientNone:
        def get_chat_completions_result(self, user_prompt, system_prompt):
            return MockLLMResultNone()

    class MockContextNone:
        def build_managed_rag_client(self):
            return MockClientNone()

    mock_databricks_rag_eval.get_context = lambda: MockContextNone()

    result = judge(outputs={"text": "test output"})
    assert isinstance(result, Feedback)
    assert result.error is not None
    assert "Empty response from Databricks judge" in result.error  # NB: None/empty response error


def test_databricks_model_works_with_trace(mock_databricks_rag_eval):
    mock_databricks_rag_eval.get_context = lambda: mock_databricks_rag_eval.MockContext(
        expected_content="trace", response_data={"result": True, "rationale": "Trace looks good"}
    )

    judge = make_judge(
        name="trace_judge", instructions="Analyze {{ trace }} for errors", model="databricks"
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
    judge = make_judge(name="test_judge", instructions=instructions, model="openai:/gpt-4")

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
        make_judge(name="test_judge", instructions=instructions, model="openai:/gpt-4")


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
        make_judge(name=name, instructions=instructions, model=model)


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
        name="test_judge", instructions="Check if {{ outputs }} is valid", model=model
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
        make_judge(name="test_judge", instructions=instructions, model=model)


def test_trace_with_inputs_outputs_allowed():
    judge1 = make_judge(
        name="test_judge",
        instructions="Analyze {{ trace }} and {{ inputs }}",
        model="openai:/gpt-4",
    )
    assert judge1.template_variables == {"trace", "inputs"}

    judge2 = make_judge(
        name="test_judge",
        instructions="Analyze {{ trace }} and {{ outputs }}",
        model="openai:/gpt-4",
    )
    assert judge2.template_variables == {"trace", "outputs"}


def test_trace_with_expectations_allowed():
    judge = make_judge(
        name="test_judge",
        instructions="Analyze {{ trace }} against {{ expectations }}",
        model="openai:/gpt-4",
    )

    assert judge is not None
    assert "trace" in judge.template_variables
    assert "expectations" in judge.template_variables


def test_call_with_trace_supported(mock_trace, monkeypatch):
    captured_args = {}

    def mock_invoke(model_uri, prompt, assessment_name, trace=None):
        captured_args.update(
            {
                "model_uri": model_uri,
                "prompt": prompt,
                "assessment_name": assessment_name,
                "trace": trace,
            }
        )
        return Feedback(name=assessment_name, value=True, rationale="Trace analyzed")

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", mock_invoke)

    judge = make_judge(
        name="test_judge", instructions="Analyze this {{ trace }}", model="openai:/gpt-4"
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
        name="test_judge", instructions="Analyze this {{ trace }}", model="openai:/gpt-4"
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
        name="test_judge", instructions="Check if {{ outputs }} is valid", model="openai:/gpt-4"
    )

    with pytest.raises(
        MlflowException, match="Must specify 'outputs' - required by template variables"
    ):
        judge()


def test_call_with_valid_outputs_returns_feedback(mock_invoke_judge_model):
    judge = make_judge(
        name="formality_judge",
        instructions="Check if {{ outputs }} is formal",
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
        name="test_judge", instructions="Check if {{ outputs }} is formal", model="openai:/gpt-4"
    )

    instructions = judge.instructions
    assert instructions == "Check if {{ outputs }} is formal"


def test_kind_property():
    judge = make_judge(
        name="test_judge", instructions="Check if {{ outputs }} is valid", model="openai:/gpt-4"
    )

    assert judge.kind == ScorerKind.CLASS


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
    assert version == 1

    retrieved_scorer = store.get_scorer(experiment, "test_judge", version)
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
        model="openai:/gpt-4o",
    )
    version2 = store.register_scorer(experiment, judge_v2)
    assert version2 == 2

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
        model="openai:/gpt-4",
    )

    assert judge.template_variables == {"inputs", "outputs", "expectations"}

    store = _get_scorer_store()
    version = store.register_scorer(experiment, judge)
    assert version == 1

    retrieved_judge = store.get_scorer(experiment, "reserved_judge", version)
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
        model="openai:/gpt-3.5-turbo",
    )

    serialized = judge.model_dump()

    expected_scorer = SerializedScorer(
        name="test_dataclass_judge",
        aggregations=[],
        mlflow_version=mlflow.__version__,
        serialization_version=1,
        instructions_judge_pydantic_data={
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


def test_instructions_judge_works_with_evaluate(mock_invoke_judge_model):
    judge = make_judge(
        name="response_quality",
        instructions="Evaluate if the {{ outputs }} is helpful given {{ inputs }}",
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
    trace_info = TraceInfo(
        trace_id="test-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=1234567890,
        execution_duration=1000,
        state=TraceState.OK,
        trace_metadata={
            "mlflow.trace_schema.version": "2",
            "mlflow.traceInputs": json.dumps(trace_inputs),
            "mlflow.traceOutputs": json.dumps(trace_outputs),
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
            name="test_span",
            inputs=span_inputs,
            outputs=span_outputs,
            span_type=SpanType.CHAIN,
        ),
    ]
    trace = Trace(info=trace_info, data=TraceData(spans=spans))
    judge = make_judge(
        name="trace_evaluator",
        instructions="Analyze this {{trace}} for quality and correctness",
        model="openai:/gpt-4",
    )
    data = pd.DataFrame({"trace": [trace]})
    result = mlflow.genai.evaluate(data=data, scorers=[judge])

    assert "trace_evaluator/value" in result.result_df.columns
    assert len(result.result_df["trace_evaluator/value"]) == 1
    assert result.result_df["trace_evaluator/value"].iloc[0]


def test_trace_prompt_augmentation(mock_trace, monkeypatch):
    captured_prompt = None

    def mock_invoke(model_uri, prompt, assessment_name, trace=None):
        nonlocal captured_prompt
        captured_prompt = prompt
        return Feedback(name=assessment_name, value=True)

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", mock_invoke)

    judge = make_judge(
        name="test_judge",
        instructions="Analyze this {{ trace }} for quality",
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
    assert "- result: The evaluation rating/result" in system_content
    assert "- rationale: Detailed explanation for the evaluation" in system_content
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
        model="openai:/gpt-4",
    )

    with pytest.raises(MlflowException, match="'trace' must be a Trace object, got str"):
        judge(trace="not a trace")

    with pytest.raises(MlflowException, match="'trace' must be a Trace object, got dict"):
        judge(trace={"trace_data": "invalid"})

    inputs_judge = make_judge(
        name="inputs_judge",
        instructions="Check {{ inputs }}",
        model="openai:/gpt-4",
    )
    with pytest.raises(MlflowException, match="Must specify 'inputs'"):
        inputs_judge(trace=None)


def test_judge_accepts_valid_trace(mock_trace, mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Analyze this {{ trace }}",
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
        else:
            call_id = f"call_{len(kwargs['messages'])}"
            mock_response.choices[0].message = litellm.Message(
                role="assistant",
                content=None,
                tool_calls=[{"id": call_id, "function": {"name": "get_span", "arguments": "{}"}}],
            )
        return mock_response

    monkeypatch.setattr("litellm.completion", mock_completion)
    monkeypatch.setattr("litellm.token_counter", lambda model, messages: len(messages) * 20)
    monkeypatch.setattr("litellm.get_max_tokens", lambda model: 120)

    judge = make_judge(name="test", instructions="test {{inputs}}", model="openai:/gpt-4")
    judge(inputs={"input": "test"}, outputs={"output": "test"}, trace=mock_trace)

    # Verify pruning happened; we expect that 2 messages were removed (one tool call pair consisting
    # of 1. assistant message and 2. tool call result message)
    assert captured_retry_messages == captured_error_messages[:2] + captured_error_messages[4:8]


def test_non_context_error_does_not_trigger_pruning(monkeypatch):
    def mock_completion(**kwargs):
        raise Exception("some other error")

    monkeypatch.setattr("litellm.completion", mock_completion)

    judge = make_judge(
        name="test_judge", instructions="Check if {{inputs}} is correct", model="openai:/gpt-4"
    )
    with pytest.raises(MlflowException, match="some other error"):
        judge(inputs={"input": "test"}, outputs={"output": "test"})


def test_trace_template_with_expectations_extracts_correctly(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Analyze the {{ trace }} to see if it meets {{ expectations }}",
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
    assert user_msg.content == ""  # Empty user message for trace-only


def test_no_warning_when_extracting_fields_from_trace(mock_invoke_judge_model):
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate {{ inputs }} and {{ outputs }}",
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
        model="openai:/gpt-4",
    )

    trace_system_msg = trace_judge._build_system_message(is_trace_based=True)

    assert trace_system_msg.count("- result:") == 1
    assert trace_system_msg.count("- rationale:") == 1

    assert "Please provide your assessment in the following JSON format" not in trace_system_msg


def test_instructions_judge_repr():
    # Test short instructions that fit within display limit
    short_instructions = "Check {{ outputs }}"
    judge = make_judge(name="test_judge", instructions=short_instructions, model="openai:/gpt-4")

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
        name="long_judge", instructions=long_instructions, model="openai:/gpt-4"
    )

    repr_long = repr(judge_long)
    assert "InstructionsJudge" in repr_long
    assert "name='long_judge'" in repr_long
    assert "model='openai:/gpt-4'" in repr_long
    # Should show first 30 characters + "..."
    assert "instructions='This is a very long instructio..." in repr_long
    assert "template_variables=['inputs', 'outputs']" in repr_long
