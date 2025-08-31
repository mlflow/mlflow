import json

import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow
import mlflow.genai.judges.instructions_judge
from mlflow.entities import Span, SpanType, Trace, TraceData, TraceInfo
from mlflow.entities.assessment import Feedback
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.instructions_judge import InstructionsJudge
from mlflow.genai.scorers.base import ScorerKind
from mlflow.tracing.utils import build_otel_context


@pytest.fixture
def mock_invoke_judge_model(monkeypatch):
    def _mock(model_uri, prompt, assessment_name):
        return Feedback(name=assessment_name, value=True, rationale="The response is formal")

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", _mock)
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
        name="test_judge", instructions="Check if {{text}} is formal", model="openai:/gpt-4"
    )

    assert isinstance(judge, InstructionsJudge)
    assert judge.name == "test_judge"
    expected_instructions = (
        "Instructions-based judge: test_judge\n\nInstructions:\n-------------\n\n"
        "Check if {{text}} is formal"
    )
    assert judge.instructions == expected_instructions
    assert judge.model == "openai:/gpt-4"


def test_make_judge_with_default_model(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    judge = make_judge(name="test_judge", instructions="Check if {{response}} is accurate")

    assert judge.model == "openai:/gpt-4.1-mini"


def test_make_judge_with_databricks_default(monkeypatch):
    monkeypatch.setattr("mlflow.genai.judges.utils.is_databricks_uri", lambda x: True)

    judge = make_judge(name="test_judge", instructions="Check if {{text}} is valid")

    assert judge.model == "databricks"


@pytest.mark.parametrize(
    ("instructions", "expected_vars", "expected_custom"),
    [
        (
            "Check if {{query}} is answered by {{response}} with {{tone}}",
            {"query", "response", "tone"},
            {"query", "response", "tone"},
        ),
        (
            "Check {{answer}} against {{expected_answer}}",
            {"answer", "expected_answer"},
            {"answer", "expected_answer"},
        ),
        (
            "Validate {{source_text}} and {{translated_text}}",
            {"source_text", "translated_text"},
            {"source_text", "translated_text"},
        ),
    ],
)
def test_template_variable_extraction(instructions, expected_vars, expected_custom):
    judge = make_judge(name="test_judge", instructions=instructions, model="openai:/gpt-4")

    assert judge.template_variables == expected_vars
    assert judge._custom_template_variables == expected_custom


@pytest.mark.parametrize(
    ("name", "instructions", "model", "error_pattern"),
    [
        ("", "Check {{text}}", "openai:/gpt-4", "name must be a non-empty string"),
        ("test", "", "openai:/gpt-4", "instructions must be a non-empty string"),
        (
            "test",
            "Check response",
            "openai:/gpt-4",
            "Instructions template must contain at least one variable",
        ),
        (
            "test",
            "Check {{text}}",
            "invalid-model",
            "Malformed model uri 'invalid-model'",
        ),
        ("test", "Check {{text}}", "invalid:/", "Malformed model uri 'invalid:/'"),
        ("test", "Check {{text}}", "openai:", "Malformed model uri 'openai:'"),
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
    judge = make_judge(name="test_judge", instructions="Check if {{text}} is valid", model=model)
    assert judge.model == model


@pytest.mark.parametrize(
    ("instructions", "model", "error_pattern"),
    [
        (
            "Analyze {{trace}} and check {{custom_field}}",
            "openai:/gpt-4",
            "When submitting a 'trace' variable, no other variables are permitted",
        ),
        (
            "Analyze {{trace}} and {{inputs}}",
            "openai:/gpt-4",
            "Instructions template cannot contain both 'trace' and 'inputs'/'outputs'",
        ),
        (
            "Analyze {{trace}} and {{outputs}}",
            "openai:/gpt-4",
            "Instructions template cannot contain both 'trace' and 'inputs'/'outputs'",
        ),
        (
            "Analyze {{trace}} for errors",
            "databricks",
            "Model cannot be 'databricks' when using 'trace' variable",
        ),
    ],
)
def test_trace_variable_restrictions(instructions, model, error_pattern):
    with pytest.raises(MlflowException, match=error_pattern):
        make_judge(name="test_judge", instructions=instructions, model=model)


def test_call_with_trace_not_supported():
    judge = make_judge(
        name="test_judge", instructions="Check if {{text}} is valid", model="openai:/gpt-4"
    )

    # Trace parameter is not supported in PR #1
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'trace'"):
        judge(trace="some_trace")


def test_call_validates_missing_custom_variables():
    judge = make_judge(
        name="test_judge",
        instructions="Check if {{query}} matches {{expected_answer}}",
        model="openai:/gpt-4",
    )

    with pytest.raises(MlflowException, match="Required template variables .* are missing"):
        judge(inputs={"query": "What is 2+2?"})


def test_call_with_no_inputs_or_outputs():
    judge = make_judge(
        name="test_judge", instructions="Check if {{text}} is valid", model="openai:/gpt-4"
    )

    with pytest.raises(MlflowException, match="Must specify 'inputs' or 'outputs' for evaluation"):
        judge()


def test_call_with_valid_inputs_returns_feedback(mock_invoke_judge_model):
    judge = make_judge(
        name="formality_judge",
        instructions="Check if {{response}} is formal",
        model="openai:/gpt-4",
    )

    result = judge(outputs={"response": "Dear Sir/Madam, I am writing to inquire..."})

    assert isinstance(result, Feedback)
    assert result.name == "formality_judge"
    assert result.value is True
    assert result.rationale == "The response is formal"


def test_call_with_expectations_as_json(monkeypatch):
    captured_prompt = None

    def mock_invoke(model_uri, prompt, assessment_name):
        nonlocal captured_prompt
        captured_prompt = prompt
        return Feedback(name=assessment_name, value=True)

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", mock_invoke)

    judge = make_judge(
        name="test_judge",
        instructions="Check {{answer}} against {{expectations}}",
        model="openai:/gpt-4",
    )

    judge(inputs={"answer": "42"}, expectations={"correct": True, "score": 100})

    assert '"correct": true' in captured_prompt
    assert '"score": 100' in captured_prompt


def test_call_with_custom_variables_from_inputs(monkeypatch):
    captured_prompt = None

    def mock_invoke(model_uri, prompt, assessment_name):
        nonlocal captured_prompt
        captured_prompt = prompt
        return Feedback(name=assessment_name, value=True)

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", mock_invoke)

    judge = make_judge(
        name="test_judge",
        instructions="Check if {{question}} meets {{criteria}}",
        model="openai:/gpt-4",
    )

    result = judge(inputs={"question": "What is AI?", "criteria": "technical accuracy"})

    assert isinstance(result, Feedback)
    assert "Check if What is AI? meets technical accuracy" in captured_prompt


def test_instructions_property():
    judge = make_judge(
        name="test_judge", instructions="Check if {{text}} is formal", model="openai:/gpt-4"
    )

    instructions = judge.instructions
    assert "Instructions-based judge: test_judge" in instructions
    assert "Check if {{text}} is formal" in instructions


def test_kind_property():
    judge = make_judge(
        name="test_judge", instructions="Check if {{text}} is valid", model="openai:/gpt-4"
    )

    assert judge.kind == ScorerKind.CLASS


@pytest.mark.parametrize(
    ("inputs", "outputs", "expectations"),
    [
        ({"text": "hello", "result": "world"}, None, None),
        ({"text": "hello"}, {"result": "world"}, None),
        ({"text": "hello"}, {"result": "world"}, {"expected": "world"}),
        (None, {"text": "hello", "result": "world"}, None),
    ],
)
def test_call_with_various_input_combinations(
    mock_invoke_judge_model, inputs, outputs, expectations
):
    judge = make_judge(
        name="test_judge", instructions="Check {{text}} and {{result}}", model="openai:/gpt-4"
    )

    result = judge(inputs=inputs, outputs=outputs, expectations=expectations)
    assert isinstance(result, Feedback)


def test_prompt_formatting_with_all_variable_types(monkeypatch):
    captured_prompt = None

    def mock_invoke(model_uri, prompt, assessment_name):
        nonlocal captured_prompt
        captured_prompt = prompt
        return Feedback(name=assessment_name, value=True)

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", mock_invoke)

    judge = make_judge(
        name="test",
        instructions="Query: {{query}}, Response: {{response}}, Custom: {{my_var}}",
        model="openai:/gpt-4",
    )

    judge(inputs={"query": "test", "my_var": "custom_value"}, outputs={"response": "answer"})

    assert "Query: test" in captured_prompt
    assert "Response: answer" in captured_prompt
    assert "Custom: custom_value" in captured_prompt
