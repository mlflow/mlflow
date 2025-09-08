import mlflow
from mlflow.entities.assessment import Expectation
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.genai.utils.trace_utils import (
    extract_expectations_from_trace,
    extract_inputs_from_trace,
    extract_outputs_from_trace,
    extract_request_from_trace,
    extract_response_from_trace,
)


def test_extract_inputs_outputs_from_trace():
    trace_inputs = {"question": "What is MLflow?"}
    trace_outputs = {"answer": "MLflow is an ML platform"}

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(trace_inputs)
        span.set_outputs(trace_outputs)

    trace = mlflow.get_trace(span.trace_id)

    assert extract_inputs_from_trace(trace) == trace_inputs
    assert extract_outputs_from_trace(trace) == trace_outputs

    # Test string extraction (existing functions have different behavior)
    request_str = extract_request_from_trace(trace)
    response_str = extract_response_from_trace(trace)
    assert "What is MLflow?" in request_str
    assert "MLflow is an ML platform" in response_str


def test_extract_inputs_outputs_with_non_dict_values():
    with mlflow.start_span(name="test_span") as span:
        span.set_inputs("simple string input")
        span.set_outputs("simple string output")

    trace = mlflow.get_trace(span.trace_id)

    assert extract_inputs_from_trace(trace) == {"value": "simple string input"}
    assert extract_outputs_from_trace(trace) == {"value": "simple string output"}

    assert extract_request_from_trace(trace) == "simple string input"
    assert extract_response_from_trace(trace) == "simple string output"


def test_extract_expectations_human_only():
    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"question": "test"})
        span.set_outputs({"answer": "test"})

    trace = mlflow.get_trace(span.trace_id)

    human_expectation = {"expected_answer": "correct answer"}
    llm_expectation = {"score": 0.8}

    mlflow.log_assessment(
        trace_id=trace.info.trace_id,
        assessment=Expectation(
            name="ground_truth",
            value=human_expectation,
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user1"),
        ),
    )

    mlflow.log_assessment(
        trace_id=trace.info.trace_id,
        assessment=Expectation(
            name="llm_rating",
            value=llm_expectation,
            source=AssessmentSource(
                source_type=AssessmentSourceType.LLM_JUDGE, source_id="gpt-4"
            ),
        ),
    )

    trace = mlflow.get_trace(span.trace_id)

    human_only_expectations = extract_expectations_from_trace(trace, human_only=True)
    assert human_only_expectations == human_expectation

    all_expectations = extract_expectations_from_trace(trace, human_only=False)
    assert all_expectations == {"ground_truth": human_expectation, "llm_rating": llm_expectation}


def test_extract_expectations_multiple_human():
    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"question": "test"})
        span.set_outputs({"answer": "test"})

    trace = mlflow.get_trace(span.trace_id)

    mlflow.log_assessment(
        trace_id=trace.info.trace_id,
        assessment=Expectation(
            name="ground_truth_1",
            value={"expected": "answer1"},
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user1"),
        ),
    )

    mlflow.log_assessment(
        trace_id=trace.info.trace_id,
        assessment=Expectation(
            name="ground_truth_2",
            value={"expected": "answer2"},
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user2"),
        ),
    )

    trace = mlflow.get_trace(span.trace_id)

    human_only_expectations = extract_expectations_from_trace(trace, human_only=True)
    assert human_only_expectations == {
        "ground_truth_1": {"expected": "answer1"},
        "ground_truth_2": {"expected": "answer2"},
    }


def test_extract_expectations_none_when_no_expectations():
    with mlflow.start_span(name="test_span") as span:
        span.set_inputs({"question": "test"})
        span.set_outputs({"answer": "test"})

    trace = mlflow.get_trace(span.trace_id)

    assert extract_expectations_from_trace(trace, human_only=True) is None
    assert extract_expectations_from_trace(trace, human_only=False) is None


def test_extract_from_empty_trace():
    with mlflow.start_span(name="test_span") as span:
        pass

    # Get the trace by span ID
    trace = mlflow.get_trace(span.trace_id)

    assert extract_inputs_from_trace(trace) is None
    assert extract_outputs_from_trace(trace) is None
    assert extract_expectations_from_trace(trace) is None
    assert extract_request_from_trace(trace) == " "
    assert extract_response_from_trace(trace) == " "


def test_extract_request_response_nested_structure():
    nested_inputs = {
        "user": {"name": "John", "query": "nested input"},
        "metadata": {"timestamp": "2024-01-01"},
    }
    nested_outputs = {
        "result": {"answer": "nested output", "confidence": 0.95},
        "status": "success",
    }

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(nested_inputs)
        span.set_outputs(nested_outputs)

    trace = mlflow.get_trace(span.trace_id)

    # Test dict extraction
    assert extract_inputs_from_trace(trace) == nested_inputs
    assert extract_outputs_from_trace(trace) == nested_outputs

    # Test string extraction
    request_str = extract_request_from_trace(trace)
    response_str = extract_response_from_trace(trace)
    assert "nested input" in request_str
    assert "nested output" in response_str


def test_extract_request_response_list_structure():
    list_inputs = {"items": ["item1", "item2", "item3"]}
    list_outputs = {"results": ["result1", "result2", "result3"]}

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(list_inputs)
        span.set_outputs(list_outputs)

    trace = mlflow.get_trace(span.trace_id)

    assert extract_inputs_from_trace(trace) == list_inputs
    assert extract_outputs_from_trace(trace) == list_outputs

    request_str = extract_request_from_trace(trace)
    response_str = extract_response_from_trace(trace)
    assert "item" in request_str
    assert "result" in response_str


def test_extract_request_response_mixed_types():
    mixed_inputs = {
        "text": "test",
        "number": 123,
        "boolean": True,
        "list": [1, 2, 3],
        "nested": {"key": "value"},
    }
    mixed_outputs = {
        "response": "mixed response",
        "score": 0.85,
        "success": True,
        "items": ["a", "b", "c"],
    }

    with mlflow.start_span(name="test_span") as span:
        span.set_inputs(mixed_inputs)
        span.set_outputs(mixed_outputs)

    trace = mlflow.get_trace(span.trace_id)

    # Test dict extraction
    assert extract_inputs_from_trace(trace) == mixed_inputs
    assert extract_outputs_from_trace(trace) == mixed_outputs

    # Test string extraction
    request_str = extract_request_from_trace(trace)
    response_str = extract_response_from_trace(trace)
    assert "test" in request_str
    assert "response" in response_str


def test_extract_from_trace_with_no_root_span():
    from unittest.mock import Mock
    from mlflow.entities.trace import Trace

    mock_trace = Mock(spec=Trace)
    mock_trace.data = Mock()
    mock_trace.data._get_root_span = Mock(return_value=None)

    assert extract_request_from_trace(mock_trace) is None
    assert extract_response_from_trace(mock_trace) is None
    assert extract_inputs_from_trace(mock_trace) is None
    assert extract_outputs_from_trace(mock_trace) is None
