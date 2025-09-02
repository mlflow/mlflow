import json
from dataclasses import asdict

import pandas as pd
import pytest
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

import mlflow
import mlflow.genai
import mlflow.genai.judges.instructions_judge
from mlflow.entities import Span, SpanType, Trace, TraceData, TraceInfo
from mlflow.entities.assessment import Feedback
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges import make_judge
from mlflow.genai.judges.instructions_judge import InstructionsJudge
from mlflow.genai.scorers.base import Scorer, ScorerKind, SerializedScorer
from mlflow.genai.scorers.registry import _get_scorer_store
from mlflow.tracing.utils import build_otel_context


@pytest.fixture
def mock_invoke_judge_model(monkeypatch):
    calls = []

    def _mock(model_uri, prompt, assessment_name):
        calls.append((model_uri, prompt, assessment_name))
        return Feedback(name=assessment_name, value=True, rationale="The response is formal")

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", _mock)
    _mock.calls = calls
    _mock.reset_mock = lambda: calls.clear()
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
    captured_messages = None

    def mock_invoke(model_uri, prompt, assessment_name):
        nonlocal captured_messages
        captured_messages = prompt
        return Feedback(name=assessment_name, value=True)

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", mock_invoke)

    judge = make_judge(
        name="test_judge",
        instructions="Check {{answer}} against {{expectations}}",
        model="openai:/gpt-4",
    )

    judge(inputs={"answer": "42"}, expectations={"correct": True, "score": 100})

    # Check that we have a list of messages
    assert isinstance(captured_messages, list)
    assert len(captured_messages) == 2

    # Expectations should be in the user message as JSON
    user_msg = captured_messages[1]
    assert '"correct": true' in user_msg["content"]
    assert '"score": 100' in user_msg["content"]


def test_call_with_custom_variables_from_inputs(monkeypatch):
    captured_messages = None

    def mock_invoke(model_uri, prompt, assessment_name):
        nonlocal captured_messages
        captured_messages = prompt
        return Feedback(name=assessment_name, value=True)

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", mock_invoke)

    judge = make_judge(
        name="test_judge",
        instructions="Check if {{question}} meets {{criteria}}",
        model="openai:/gpt-4",
    )

    result = judge(inputs={"question": "What is AI?", "criteria": "technical accuracy"})

    assert isinstance(result, Feedback)
    # Check that we have a list of messages
    assert isinstance(captured_messages, list)
    assert len(captured_messages) == 2

    # Check system message has the template
    system_msg = captured_messages[0]
    assert "Check if {{question}} meets {{criteria}}" in system_msg["content"]

    # Check user message has the values
    user_msg = captured_messages[1]
    assert "criteria: technical accuracy" in user_msg["content"]
    assert "question: What is AI?" in user_msg["content"]


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
    captured_messages = None

    def mock_invoke(model_uri, prompt, assessment_name):
        nonlocal captured_messages
        captured_messages = prompt
        return Feedback(name=assessment_name, value=True)

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", mock_invoke)

    judge = make_judge(
        name="test",
        instructions="Query: {{query}}, Response: {{response}}, Custom: {{my_var}}",
        model="openai:/gpt-4",
    )

    judge(inputs={"query": "test", "my_var": "custom_value"}, outputs={"response": "answer"})

    # Check that we have a list of messages
    assert isinstance(captured_messages, list)
    assert len(captured_messages) == 2

    # Check system message has the template
    system_msg = captured_messages[0]
    assert "Query: {{query}}, Response: {{response}}, Custom: {{my_var}}" in system_msg["content"]

    # Check user message has all the values
    user_msg = captured_messages[1]
    assert "my_var: custom_value" in user_msg["content"]
    assert "query: test" in user_msg["content"]
    assert "response: answer" in user_msg["content"]


def test_output_format_instructions_added(monkeypatch):
    captured_messages = None

    def mock_invoke(model_uri, prompt, assessment_name):
        nonlocal captured_messages
        captured_messages = prompt
        return Feedback(name=assessment_name, value=True, rationale="Test rationale")

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", mock_invoke)

    judge = make_judge(
        name="test_judge",
        instructions="Check if {{text}} is formal",
        model="openai:/gpt-4",
    )

    result = judge(outputs={"text": "Hello there"})

    # Check that we have a list of messages
    assert isinstance(captured_messages, list)
    assert len(captured_messages) == 2

    # Check system message
    system_msg = captured_messages[0]
    assert system_msg["role"] == "system"
    assert "You are a helpful judge" in system_msg["content"]
    assert "Check if {{text}} is formal" in system_msg["content"]
    assert "JSON format" in system_msg["content"]

    # Check user message
    user_msg = captured_messages[1]
    assert user_msg["role"] == "user"
    assert "text: Hello there" in user_msg["content"]

    assert result.value is True
    assert result.rationale == "Test rationale"


def test_output_format_instructions_with_complex_template(monkeypatch):
    captured_messages = None

    def mock_invoke(model_uri, prompt, assessment_name):
        nonlocal captured_messages
        captured_messages = prompt
        return Feedback(name=assessment_name, value="yes", rationale="Test rationale")

    monkeypatch.setattr(mlflow.genai.judges.instructions_judge, "invoke_judge_model", mock_invoke)

    judge = make_judge(
        name="complex_judge",
        instructions="Evaluate {{response}} for {{criteria}} considering {{context}}",
        model="openai:/gpt-4",
    )

    result = judge(
        inputs={"context": "formal business setting"},
        outputs={"response": "Hey what's up"},
        expectations={"criteria": "professionalism"},
    )

    # Check that we have a list of messages
    assert isinstance(captured_messages, list)
    assert len(captured_messages) == 2

    # Check system message
    system_msg = captured_messages[0]
    assert system_msg["role"] == "system"
    assert "You are a helpful judge" in system_msg["content"]
    assert "Evaluate {{response}} for {{criteria}} considering {{context}}" in system_msg["content"]
    assert "JSON format" in system_msg["content"]

    # Check user message has all the variable values
    user_msg = captured_messages[1]
    assert user_msg["role"] == "user"
    assert "context: formal business setting" in user_msg["content"]
    assert "criteria: professionalism" in user_msg["content"]
    assert "response: Hey what's up" in user_msg["content"]

    assert result.value == "yes"
    assert result.rationale == "Test rationale"


def test_judge_registration_as_scorer(mock_invoke_judge_model):
    experiment = mlflow.create_experiment("test_judge_registration")

    original_instructions = "Evaluate if the response {{response}} is professional and formal."
    judge = make_judge(
        name="test_judge",
        instructions=original_instructions,
        model="openai:/gpt-4",
    )

    formatted_instructions = judge.instructions
    assert "Instructions-based judge: test_judge" in formatted_instructions
    assert original_instructions in formatted_instructions
    assert judge.model == "openai:/gpt-4"
    assert judge.template_variables == {"response"}

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
    assert retrieved_scorer.instructions == formatted_instructions
    assert retrieved_scorer.model == "openai:/gpt-4"
    assert retrieved_scorer.template_variables == {"response"}
    assert "Instructions-based judge: test_judge" in retrieved_scorer.instructions
    assert original_instructions in retrieved_scorer.instructions

    deserialized = Scorer.model_validate(serialized)
    assert isinstance(deserialized, InstructionsJudge)
    assert deserialized.name == judge.name
    assert deserialized.instructions == formatted_instructions
    assert deserialized.model == judge.model
    assert deserialized.template_variables == {"response"}

    test_output = {"response": "This output demonstrates professional communication."}
    result = retrieved_scorer(outputs=test_output)
    assert isinstance(result, Feedback)
    assert result.name == "test_judge"

    expected_prompt = (
        "Evaluate if the response This output demonstrates professional "
        "communication. is professional and formal."
    )
    assert len(mock_invoke_judge_model.calls) == 1
    model_uri, prompt, assessment_name = mock_invoke_judge_model.calls[0]
    assert model_uri == "openai:/gpt-4"
    assert prompt == expected_prompt
    assert assessment_name == "test_judge"

    mock_invoke_judge_model.reset_mock()
    result2 = deserialized(outputs=test_output)
    assert isinstance(result2, Feedback)
    assert len(mock_invoke_judge_model.calls) == 1
    model_uri, prompt, assessment_name = mock_invoke_judge_model.calls[0]
    assert model_uri == "openai:/gpt-4"
    assert prompt == expected_prompt
    assert assessment_name == "test_judge"

    v2_instructions = "Evaluate if the output {{outputs}} is professional, formal, and concise."
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
    assert "Instructions-based judge: test_judge" in v1_scorer.instructions
    assert original_instructions in v1_scorer.instructions
    assert v1_scorer.model == "openai:/gpt-4"

    v2_scorer, v2_num = versions[1]
    assert v2_num == 2
    assert isinstance(v2_scorer, InstructionsJudge)
    assert "Instructions-based judge: test_judge" in v2_scorer.instructions
    assert v2_instructions in v2_scorer.instructions
    assert v2_scorer.model == "openai:/gpt-4o"

    latest = store.get_scorer(experiment, "test_judge")
    assert isinstance(latest, InstructionsJudge)
    assert "Instructions-based judge: test_judge" in latest.instructions
    assert v2_instructions in latest.instructions
    assert latest.model == "openai:/gpt-4o"


def test_judge_registration_preserves_custom_variables(mock_invoke_judge_model):
    experiment = mlflow.create_experiment("test_custom_vars")

    instructions_with_custom = (
        "Check if {{query}} is answered correctly by {{response}} "
        "according to {{criteria}} with {{threshold}} accuracy"
    )
    judge = make_judge(
        name="custom_judge",
        instructions=instructions_with_custom,
        model="openai:/gpt-4",
    )

    assert judge.template_variables == {"query", "response", "criteria", "threshold"}

    store = _get_scorer_store()
    version = store.register_scorer(experiment, judge)
    assert version == 1

    retrieved_judge = store.get_scorer(experiment, "custom_judge", version)
    assert isinstance(retrieved_judge, InstructionsJudge)
    assert "Instructions-based judge: custom_judge" in retrieved_judge.instructions
    assert instructions_with_custom in retrieved_judge.instructions
    assert retrieved_judge.template_variables == {"query", "response", "criteria", "threshold"}

    result = retrieved_judge(
        inputs={"query": "What is 2+2?", "criteria": "mathematical accuracy"},
        outputs={"response": "The answer is 4", "threshold": "95%"},
    )
    assert isinstance(result, Feedback)
    assert result.name == "custom_judge"

    expected_prompt = (
        "Check if What is 2+2? is answered correctly by The answer is 4 "
        "according to mathematical accuracy with 95% accuracy"
    )
    assert len(mock_invoke_judge_model.calls) == 1
    model_uri, prompt, assessment_name = mock_invoke_judge_model.calls[0]
    assert model_uri == "openai:/gpt-4"
    assert prompt == expected_prompt
    assert assessment_name == "custom_judge"

    mock_invoke_judge_model.reset_mock()

    with pytest.raises(MlflowException, match="Required template variables .* are missing"):
        retrieved_judge(
            inputs={"query": "What is 2+2?"},
            outputs={"response": "The answer is 4"},
        )


def test_model_dump_comprehensive():
    basic_judge = make_judge(
        name="basic_judge",
        instructions="Check if {{inputs}} is correct",
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
        == "Check if {{inputs}} is correct"
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
        instructions="Check if {{inputs}} matches {{expectations}} for {{custom_field}}",
        model="anthropic:/claude-3",
    )

    complex_serialized = complex_judge.model_dump()

    assert complex_serialized["instructions_judge_pydantic_data"]["instructions"] == (
        "Check if {{inputs}} matches {{expectations}} for {{custom_field}}"
    )
    assert complex_serialized["instructions_judge_pydantic_data"]["model"] == "anthropic:/claude-3"

    default_model_judge = make_judge(
        name="default_judge",
        instructions="Evaluate {{outputs}}",
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
        assert raw_instructions in deserialized.instructions
        assert f"Instructions-based judge: {deserialized.name}" in deserialized.instructions
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
        "instructions_judge_pydantic_data": {"instructions": "Check {{inputs}}"},
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
        "instructions_judge_pydantic_data": {"instructions": 123, "model": "openai:/gpt-4"},
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
        instructions="Evaluate {{inputs}} and {{outputs}}",
        model="openai:/gpt-3.5-turbo",
    )

    serialized = judge.model_dump()

    expected_scorer = SerializedScorer(
        name="test_dataclass_judge",
        aggregations=[],
        mlflow_version=mlflow.__version__,
        serialization_version=1,
        instructions_judge_pydantic_data={
            "instructions": "Evaluate {{inputs}} and {{outputs}}",
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
        instructions="Evaluate if the {{response}} is helpful for answering the {{question}}",
        model="openai:/gpt-4",
        aggregations=["mean"],
    )

    assert judge.aggregations == ["mean"]

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

    assert "response_quality/mean" in result.metrics
    assert result.metrics["response_quality/mean"] == 1.0

    assert "response_quality/value" in result.result_df.columns
    assert len(result.result_df["response_quality/value"]) == 2
    assert all(score is True for score in result.result_df["response_quality/value"])


def test_instructions_judge_with_no_aggregations(mock_invoke_judge_model):
    judge = make_judge(
        name="response_quality",
        instructions="Evaluate if the {{response}} is helpful for answering the {{question}}",
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


def test_make_judge_with_aggregations_validation():
    with pytest.raises(MlflowException, match="Invalid aggregation 'invalid'"):
        make_judge(
            name="test_judge",
            instructions="Check if {{text}} is valid",
            model="openai:/gpt-4",
            aggregations=["mean", "invalid", "max"],
        )

    with pytest.raises(MlflowException, match="Valid aggregations are"):
        make_judge(
            name="test_judge",
            instructions="Check if {{text}} is valid",
            model="openai:/gpt-4",
            aggregations=["not_valid"],
        )

    with pytest.raises(MlflowException, match="Aggregation must be either a string"):
        make_judge(
            name="test_judge",
            instructions="Check if {{text}} is valid",
            model="openai:/gpt-4",
            aggregations=["mean", 123],
        )

    def custom_aggregation(values):
        return sum(values) / len(values) if values else 0

    judge_with_custom_func = make_judge(
        name="test_judge",
        instructions="Check if {{text}} is valid",
        model="openai:/gpt-4",
        aggregations=["mean", custom_aggregation],
    )
    assert "mean" in judge_with_custom_func.aggregations
    assert custom_aggregation in judge_with_custom_func.aggregations

    all_valid_aggregations = ["min", "max", "mean", "median", "variance", "p90"]
    judge_with_all_aggs = make_judge(
        name="test_judge",
        instructions="Check if {{text}} is valid",
        model="openai:/gpt-4",
        aggregations=all_valid_aggregations,
    )
    assert judge_with_all_aggs.aggregations == all_valid_aggregations


def test_make_judge_with_aggregations(mock_invoke_judge_model):
    judge_with_custom_aggs = make_judge(
        name="formal_judge",
        instructions="Check if {{text}} is formal",
        model="openai:/gpt-4",
        aggregations=["mean", "max", "min"],
    )

    assert judge_with_custom_aggs.name == "formal_judge"
    assert judge_with_custom_aggs.aggregations == ["mean", "max", "min"]

    judge_with_default_aggs = make_judge(
        name="simple_judge",
        instructions="Check if {{text}} is valid",
        model="openai:/gpt-4",
    )

    assert judge_with_default_aggs.name == "simple_judge"
    assert judge_with_default_aggs.aggregations == []

    judge_with_empty_aggs = make_judge(
        name="no_aggs_judge",
        instructions="Check if {{text}} exists",
        model="openai:/gpt-4",
        aggregations=[],
    )

    assert judge_with_empty_aggs.name == "no_aggs_judge"
    assert judge_with_empty_aggs.aggregations == []
