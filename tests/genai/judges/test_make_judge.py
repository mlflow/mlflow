import json
from dataclasses import asdict
from unittest import mock

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
from mlflow.genai.scorers.base import Scorer, ScorerKind, SerializedScorer
from mlflow.genai.scorers.registry import _get_scorer_store
from mlflow.tracing.utils import build_otel_context
from mlflow.types.llm import ChatMessage


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
        name="test_judge",
        instructions="Check if {{ outputs }} is formal",
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

    judge = make_judge(name="test_judge", instructions="Check if {{ outputs }} is accurate")

    assert judge.model == expected_model


def test_make_judge_with_databricks_default(monkeypatch):
    monkeypatch.setattr("mlflow.genai.judges.utils.is_databricks_uri", lambda x: True)

    judge = make_judge(name="test_judge", instructions="Check if {{ outputs }} is valid")

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
def test_valid_model_formats(model):
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
        # These are now ALLOWED - mixing trace with inputs/outputs is supported
        # (
        #     "Analyze {{ trace }} and {{ inputs }}",
        #     "openai:/gpt-4",
        #     "Instructions template cannot contain both 'trace' and 'inputs'/'outputs'",
        # ),
        # (
        #     "Analyze {{ trace }} and {{ outputs }}",
        #     "openai:/gpt-4",
        #     "Instructions template cannot contain both 'trace' and 'inputs'/'outputs'",
        # ),
        (
            "Analyze {{ trace }} for errors",
            "databricks",
            "Model cannot be 'databricks' when using 'trace' variable",
        ),
    ],
)
def test_trace_variable_restrictions(instructions, model, error_pattern):
    with pytest.raises(MlflowException, match=error_pattern):
        make_judge(name="test_judge", instructions=instructions, model=model)


def test_trace_with_inputs_outputs_allowed():
    # These combinations are now ALLOWED - we support mixed templates
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
    # Test that mixing trace with expectations is now allowed
    judge = make_judge(
        name="test_judge",
        instructions="Analyze {{ trace }} against {{ expectations }}",
        model="openai:/gpt-4",
    )

    # Verify the judge was created successfully and has both template variables
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
        name="test_judge",
        instructions="Analyze this {{ trace }}",
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
        name="test_judge",
        instructions="Check if {{ outputs }} is formal",
        model="openai:/gpt-4",
    )

    instructions = judge.instructions
    assert instructions == "Check if {{ outputs }} is formal"


def test_kind_property():
    judge = make_judge(
        name="test_judge",
        instructions="Check if {{ outputs }} is valid",
        model="openai:/gpt-4",
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

    assert "expert judge" in captured_prompt
    assert "step-by-step record" in captured_prompt
    assert "provided to you" in captured_prompt
    assert "Evaluation Rating Fields" in captured_prompt
    assert "- result: The evaluation rating/result" in captured_prompt
    assert "- rationale: Detailed explanation for the evaluation" in captured_prompt
    assert "Instructions" in captured_prompt
    assert "Analyze this {{ trace }} for quality" in captured_prompt


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

    with pytest.raises(MlflowException, match="'trace' must be a Trace instance, got str"):
        judge(trace="not a trace")

    with pytest.raises(MlflowException, match="'trace' must be a Trace instance, got dict"):
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

    assert len(mock_invoke_judge_model.calls) == 1
    _, prompt, _ = mock_invoke_judge_model.calls[0]

    user_msg = prompt[1]
    user_content = user_msg.content

    expected_inputs_json = json.dumps(trace_inputs, default=str, indent=2)
    expected_outputs_json = json.dumps(trace_outputs, default=str, indent=2)

    assert f"inputs: {expected_inputs_json}" in user_content
    assert f"outputs: {expected_outputs_json}" in user_content


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

    expected_inputs_json = json.dumps({"value": trace_inputs}, default=str, indent=2)
    expected_outputs_json = json.dumps({"value": trace_outputs}, default=str, indent=2)

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
    expected_expectations_json = json.dumps(expected_answer, default=str, indent=2)

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

    assert isinstance(prompt, str)
    assert "analyze a trace" in prompt.lower()


def test_field_based_template_with_trace_and_explicit_inputs(mock_invoke_judge_model):
    # NB: Tests that when both trace and explicit inputs are provided for a field-based template,
    # the explicit inputs take precedence over trace extraction
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

    # Call with both trace and explicit values
    judge(trace=trace, inputs=explicit_inputs, outputs=explicit_outputs)

    assert len(mock_invoke_judge_model.calls) == 1
    model_uri, prompt, assessment_name = mock_invoke_judge_model.calls[0]
    messages = prompt

    assert isinstance(messages, list)
    assert len(messages) == 2

    user_message = messages[1].content
    # Verify explicit values were used, not trace values
    assert "explicitly provided" in user_message
    assert "Explicit answer" in user_message
    assert "in the trace" not in user_message
    assert "Trace answer" not in user_message


def test_field_based_template_extracts_missing_fields_from_trace(
    mock_invoke_judge_model,
):
    # NB: Tests that when trace is provided but some fields are missing,
    # they are extracted from the trace
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

    # Call with trace and only explicit inputs (outputs should be extracted from trace)
    judge(trace=trace, inputs=explicit_inputs)

    assert len(mock_invoke_judge_model.calls) == 1
    model_uri, prompt, assessment_name = mock_invoke_judge_model.calls[0]
    messages = prompt

    user_message = messages[1].content
    # Verify explicit inputs were used
    assert "Explicitly provided" in user_message
    # Verify outputs were extracted from trace
    assert "Trace output" in user_message


def test_trace_based_template_with_additional_inputs(mock_invoke_judge_model):
    # NB: Tests that trace-based templates properly interpolate field variables in instructions
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

    # Call with both trace and additional inputs
    judge(trace=trace, inputs=additional_inputs)

    assert len(mock_invoke_judge_model.calls) == 1
    model_uri, prompt, assessment_name = mock_invoke_judge_model.calls[0]

    # Should be a string prompt for trace-based evaluation
    assert isinstance(prompt, str)
    assert "analyze a trace" in prompt.lower()

    # The template variable {{ inputs }} should be replaced with the actual JSON
    expected_inputs_json = json.dumps(additional_inputs, default=str, indent=2)
    assert expected_inputs_json in prompt
    assert "reference" in prompt
    assert "This is the expected behavior" in prompt
    # Should NOT have the literal {{ inputs }} anymore
    assert "{{ inputs }}" not in prompt


def test_mixed_template_validation_allows_trace_with_fields():
    # NB: Tests that we can create a judge with both trace and field variables
    # This used to raise an error but should now be allowed
    judge = make_judge(
        name="test_judge",
        instructions="Evaluate {{ trace }} against {{ inputs }} and {{ outputs }}",
        model="openai:/gpt-4",
    )

    assert judge.template_variables == {"trace", "inputs", "outputs"}


def test_mixed_trace_and_fields_template_comprehensive(mock_invoke_judge_model):
    # NB: Comprehensive test for mixed template with all field types
    judge = make_judge(
        name="test_judge",
        instructions=(
            "Evaluate the {{ trace }} considering the reference {{ inputs }}, "
            "expected {{ outputs }}, and ground truth {{ expectations }}"
        ),
        model="openai:/gpt-4",
    )

    assert judge.template_variables == {"trace", "inputs", "outputs", "expectations"}

    # Create a trace
    trace_inputs = {"question": "What is MLflow?"}
    trace_outputs = {"answer": "MLflow is an open source platform"}

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(trace_inputs)
        span.set_outputs(trace_outputs)

    trace = mlflow.get_trace(span.trace_id)

    # Provide additional context
    additional_inputs = {"reference": "This is the expected input format"}
    additional_outputs = {"expected_format": "JSON with answer field"}
    additional_expectations = {"criteria": "Answer must mention ML lifecycle"}

    # Call with trace and all additional fields
    judge(
        trace=trace,
        inputs=additional_inputs,
        outputs=additional_outputs,
        expectations=additional_expectations,
    )

    assert len(mock_invoke_judge_model.calls) == 1
    model_uri, prompt, assessment_name = mock_invoke_judge_model.calls[0]

    # Should be a string prompt for trace-based evaluation
    assert isinstance(prompt, str)
    assert "analyze a trace" in prompt.lower()

    # All template variables should be interpolated with actual values
    expected_inputs_json = json.dumps(additional_inputs, default=str, indent=2)
    expected_outputs_json = json.dumps(additional_outputs, default=str, indent=2)
    expected_expectations_json = json.dumps(additional_expectations, default=str, indent=2)

    assert expected_inputs_json in prompt
    assert expected_outputs_json in prompt
    assert expected_expectations_json in prompt

    # Verify specific values are present
    assert "reference" in prompt
    assert "This is the expected input format" in prompt
    assert "expected_format" in prompt
    assert "JSON with answer field" in prompt
    assert "criteria" in prompt
    assert "Answer must mention ML lifecycle" in prompt

    # Should NOT have the literal template variables anymore
    assert "{{ inputs }}" not in prompt
    assert "{{ outputs }}" not in prompt
    assert "{{ expectations }}" not in prompt
    # But {{ trace }} should remain since it's handled by the agent
    assert "{{ trace }}" in prompt or "{{trace}}" in prompt
