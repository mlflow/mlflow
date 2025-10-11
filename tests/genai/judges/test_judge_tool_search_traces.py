from unittest import mock

import pandas as pd
import pytest

from mlflow.entities.assessment import Expectation, Feedback
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation, TraceLocationType
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.tools.search_traces import (
    SearchTracesTool,
    _convert_assessments_to_tool_types,
    _get_experiment_id,
)
from mlflow.genai.judges.tools.types import Expectation as ToolExpectation
from mlflow.genai.judges.tools.types import Feedback as ToolFeedback
from mlflow.genai.judges.tools.types import TraceInfo as ToolTraceInfo
from mlflow.types.llm import ToolDefinition


def test_search_traces_tool_name() -> None:
    tool = SearchTracesTool()
    assert tool.name == "search_traces"


def test_search_traces_tool_get_definition() -> None:
    tool = SearchTracesTool()
    definition = tool.get_definition()

    assert isinstance(definition, ToolDefinition)
    assert definition.function.name == "search_traces"
    assert "Search for traces within the same MLflow experiment" in definition.function.description
    assert definition.function.parameters.type == "object"
    assert definition.function.parameters.required == []
    assert definition.type == "function"

    properties = definition.function.parameters.properties
    assert "filter_string" in properties
    assert "order_by" in properties
    assert "max_results" in properties


def test_convert_assessments_to_tool_types_with_expectations() -> None:
    source = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user123")
    expectation = Expectation(
        name="test_expectation",
        source=source,
        span_id="span-1",
        value=True,
    )
    expectation.rationale = "Expected to be true"
    expectation.assessment_id = "assess-1"
    expectations = [expectation]

    result = _convert_assessments_to_tool_types(expectations)

    assert len(result) == 1
    assert isinstance(result[0], ToolExpectation)
    assert result[0].name == "test_expectation"
    assert result[0].source == AssessmentSourceType.HUMAN
    assert result[0].rationale == "Expected to be true"
    assert result[0].span_id == "span-1"
    assert result[0].assessment_id == "assess-1"
    assert result[0].value is True


def test_convert_assessments_to_tool_types_with_feedback() -> None:
    source = AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="judge-1")
    error = AssessmentError(
        error_code="VALIDATION_ERROR",
        error_message="Invalid input",
        stack_trace="Stack trace here",
    )
    feedback = Feedback(
        name="test_feedback",
        source=source,
        span_id="span-2",
        value="positive",
        error=error,
    )
    feedback.rationale = "Feedback rationale"
    feedback.assessment_id = "assess-2"
    feedbacks = [feedback]

    result = _convert_assessments_to_tool_types(feedbacks)

    assert len(result) == 1
    assert isinstance(result[0], ToolFeedback)
    assert result[0].name == "test_feedback"
    assert result[0].source == AssessmentSourceType.LLM_JUDGE
    assert result[0].rationale == "Feedback rationale"
    assert result[0].span_id == "span-2"
    assert result[0].assessment_id == "assess-2"
    assert result[0].value == "positive"
    assert result[0].error_code == "VALIDATION_ERROR"
    assert result[0].error_message == "Invalid input"
    assert result[0].stack_trace == "Stack trace here"


def test_convert_assessments_to_tool_types_with_feedback_no_error() -> None:
    source = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="user456")
    feedback = Feedback(
        name="feedback_no_error",
        source=source,
        span_id=None,
        value="negative",
        error=None,
    )
    feedback.assessment_id = "assess-3"
    feedbacks = [feedback]

    result = _convert_assessments_to_tool_types(feedbacks)

    assert len(result) == 1
    assert isinstance(result[0], ToolFeedback)
    assert result[0].error_code is None
    assert result[0].error_message is None
    assert result[0].stack_trace is None


def test_convert_assessments_to_tool_types_mixed() -> None:
    source = AssessmentSource(source_type=AssessmentSourceType.HUMAN)
    assessments = [
        Expectation(name="exp1", source=source, value=True),
        Feedback(name="fb1", source=source, value="positive"),
    ]

    result = _convert_assessments_to_tool_types(assessments)

    assert len(result) == 2
    assert isinstance(result[0], ToolExpectation)
    assert isinstance(result[1], ToolFeedback)


def test_get_experiment_id_success() -> None:
    trace_location = TraceLocation.from_experiment_id("exp-123")
    trace_info = TraceInfo(
        trace_id="trace-1",
        trace_location=trace_location,
        request_time=1234567890,
        state=TraceState.OK,
    )
    trace = Trace(info=trace_info, data=None)

    experiment_id = _get_experiment_id(trace)

    assert experiment_id == "exp-123"


def test_get_experiment_id_no_trace_location() -> None:
    trace_info = TraceInfo(
        trace_id="trace-1",
        trace_location=None,
        request_time=1234567890,
        state=TraceState.OK,
    )
    trace = Trace(info=trace_info, data=None)

    with pytest.raises(MlflowException, match="Current trace has no trace location"):
        _get_experiment_id(trace)


def test_get_experiment_id_not_mlflow_experiment() -> None:
    trace_location = TraceLocation(type=TraceLocationType.INFERENCE_TABLE)
    trace_info = TraceInfo(
        trace_id="trace-1",
        trace_location=trace_location,
        request_time=1234567890,
        state=TraceState.OK,
    )
    trace = Trace(info=trace_info, data=None)

    with pytest.raises(MlflowException, match="Current trace is not from an MLflow experiment"):
        _get_experiment_id(trace)


def test_get_experiment_id_no_experiment_id() -> None:
    trace_location = TraceLocation(type=TraceLocationType.MLFLOW_EXPERIMENT)
    trace_info = TraceInfo(
        trace_id="trace-1",
        trace_location=trace_location,
        request_time=1234567890,
        state=TraceState.OK,
    )
    trace = Trace(info=trace_info, data=None)

    with pytest.raises(MlflowException, match="Current trace has no experiment_id"):
        _get_experiment_id(trace)


@pytest.fixture
def mock_trace() -> Trace:
    trace_location = TraceLocation.from_experiment_id("exp-456")
    trace_info = TraceInfo(
        trace_id="trace-current",
        trace_location=trace_location,
        request_time=1234567890,
        state=TraceState.OK,
    )
    return Trace(info=trace_info, data=None)


@pytest.fixture
def mock_search_traces_df() -> pd.DataFrame:
    trace_data = [
        {
            "trace_id": "trace-1",
            "trace": (
                '{"info": {"trace_id": "trace-1", "experiment_id": "exp-456", '
                '"request_time": 1000000000, "state": "OK", "execution_duration": 150}}'
            ),
            "request": "request1",
            "response": "response1",
        },
        {
            "trace_id": "trace-2",
            "trace": (
                '{"info": {"trace_id": "trace-2", "experiment_id": "exp-456", '
                '"request_time": 1000000100, "state": "ERROR", "execution_duration": 200}}'
            ),
            "request": "request2",
            "response": "response2",
        },
    ]
    return pd.DataFrame(trace_data)


def test_search_traces_tool_invoke_success(
    mock_trace: Trace, mock_search_traces_df: pd.DataFrame
) -> None:
    tool = SearchTracesTool()

    source = AssessmentSource(source_type=AssessmentSourceType.HUMAN)
    mock_trace1_info = TraceInfo(
        trace_id="trace-1",
        trace_location=TraceLocation.from_experiment_id("exp-456"),
        request_time=1000000000,
        state=TraceState.OK,
        execution_duration=150,
        assessments=[Expectation(name="quality", source=source, value=True)],
    )
    mock_trace1 = Trace(info=mock_trace1_info, data=None)

    mock_trace2_info = TraceInfo(
        trace_id="trace-2",
        trace_location=TraceLocation.from_experiment_id("exp-456"),
        request_time=1000000100,
        state=TraceState.ERROR,
        execution_duration=200,
        assessments=[],
    )
    mock_trace2 = Trace(info=mock_trace2_info, data=None)

    with mock.patch("mlflow.search_traces", return_value=mock_search_traces_df) as mock_search:
        with mock.patch.object(Trace, "from_json", side_effect=[mock_trace1, mock_trace2]):
            result = tool.invoke(
                mock_trace, filter_string='attributes.status = "OK"', max_results=10
            )

        mock_search.assert_called_once_with(
            locations=["exp-456"],
            filter_string='attributes.status = "OK"',
            order_by=["timestamp ASC"],
            max_results=10,
            extract_fields=["trace_id", "trace", "request", "response"],
        )

    assert len(result) == 2
    assert isinstance(result[0], ToolTraceInfo)
    assert result[0].trace_id == "trace-1"
    assert result[0].request == "request1"
    assert result[0].response == "response1"
    assert result[0].state == TraceState.OK
    assert result[0].execution_duration == 150
    assert len(result[0].assessments) == 1

    assert result[1].trace_id == "trace-2"
    assert result[1].state == TraceState.ERROR
    assert result[1].execution_duration == 200
    assert len(result[1].assessments) == 0


def test_search_traces_tool_invoke_with_order_by(
    mock_trace: Trace, mock_search_traces_df: pd.DataFrame
) -> None:
    tool = SearchTracesTool()

    mock_trace1_info = TraceInfo(
        trace_id="trace-1",
        trace_location=TraceLocation.from_experiment_id("exp-456"),
        request_time=1000000000,
        state=TraceState.OK,
        execution_duration=150,
        assessments=[],
    )
    mock_trace1 = Trace(info=mock_trace1_info, data=None)

    mock_trace2_info = TraceInfo(
        trace_id="trace-2",
        trace_location=TraceLocation.from_experiment_id("exp-456"),
        request_time=1000000100,
        state=TraceState.ERROR,
        execution_duration=200,
        assessments=[],
    )
    mock_trace2 = Trace(info=mock_trace2_info, data=None)

    with mock.patch("mlflow.search_traces", return_value=mock_search_traces_df) as mock_search:
        with mock.patch.object(Trace, "from_json", side_effect=[mock_trace1, mock_trace2]):
            result = tool.invoke(
                mock_trace, order_by=["timestamp DESC", "trace_id ASC"], max_results=5
            )

        mock_search.assert_called_once_with(
            locations=["exp-456"],
            filter_string=None,
            order_by=["timestamp DESC", "trace_id ASC"],
            max_results=5,
            extract_fields=["trace_id", "trace", "request", "response"],
        )

    assert len(result) == 2


def test_search_traces_tool_invoke_default_order_by(
    mock_trace: Trace, mock_search_traces_df: pd.DataFrame
) -> None:
    tool = SearchTracesTool()

    mock_trace1_info = TraceInfo(
        trace_id="trace-1",
        trace_location=TraceLocation.from_experiment_id("exp-456"),
        request_time=1000000000,
        state=TraceState.OK,
        execution_duration=150,
        assessments=[],
    )
    mock_trace1 = Trace(info=mock_trace1_info, data=None)

    mock_trace2_info = TraceInfo(
        trace_id="trace-2",
        trace_location=TraceLocation.from_experiment_id("exp-456"),
        request_time=1000000100,
        state=TraceState.ERROR,
        execution_duration=200,
        assessments=[],
    )
    mock_trace2 = Trace(info=mock_trace2_info, data=None)

    with mock.patch("mlflow.search_traces", return_value=mock_search_traces_df) as mock_search:
        with mock.patch.object(Trace, "from_json", side_effect=[mock_trace1, mock_trace2]):
            result = tool.invoke(mock_trace)

        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["order_by"] == ["timestamp ASC"]
        assert call_kwargs["max_results"] == 20

    assert len(result) == 2


def test_search_traces_tool_invoke_empty_results(mock_trace: Trace) -> None:
    tool = SearchTracesTool()
    empty_df = pd.DataFrame(columns=["trace_id", "trace", "request", "response"])

    with mock.patch("mlflow.search_traces", return_value=empty_df):
        result = tool.invoke(mock_trace)

    assert len(result) == 0
    assert result == []


def test_search_traces_tool_invoke_search_fails(mock_trace: Trace) -> None:
    tool = SearchTracesTool()

    with mock.patch("mlflow.search_traces", side_effect=Exception("Search failed")):
        with pytest.raises(MlflowException, match="Failed to search traces"):
            tool.invoke(mock_trace)


def test_search_traces_tool_invoke_invalid_trace_json(mock_trace: Trace) -> None:
    tool = SearchTracesTool()

    invalid_df = pd.DataFrame(
        {
            "trace_id": ["trace-1", "trace-2"],
            "trace": ["invalid-json", '{"info": {}}'],
            "request": ["req1", "req2"],
            "response": ["resp1", "resp2"],
        }
    )

    with mock.patch("mlflow.search_traces", return_value=invalid_df):
        result = tool.invoke(mock_trace)

    assert len(result) == 0


def test_search_traces_tool_invoke_partial_failure(
    mock_trace: Trace, mock_search_traces_df: pd.DataFrame
) -> None:
    tool = SearchTracesTool()

    corrupted_df = mock_search_traces_df.copy()
    corrupted_df.loc[0, "trace"] = "corrupted-json"

    mock_trace2_info = TraceInfo(
        trace_id="trace-2",
        trace_location=TraceLocation.from_experiment_id("exp-456"),
        request_time=1000000100,
        state=TraceState.ERROR,
        execution_duration=200,
        assessments=[],
    )
    mock_trace2 = Trace(info=mock_trace2_info, data=None)

    with mock.patch("mlflow.search_traces", return_value=corrupted_df):
        with mock.patch.object(
            Trace, "from_json", side_effect=[ValueError("Corrupted JSON"), mock_trace2]
        ):
            result = tool.invoke(mock_trace)

    assert len(result) == 1
    assert result[0].trace_id == "trace-2"


def test_search_traces_tool_invoke_no_filter(
    mock_trace: Trace, mock_search_traces_df: pd.DataFrame
) -> None:
    tool = SearchTracesTool()

    mock_trace1_info = TraceInfo(
        trace_id="trace-1",
        trace_location=TraceLocation.from_experiment_id("exp-456"),
        request_time=1000000000,
        state=TraceState.OK,
        execution_duration=150,
        assessments=[],
    )
    mock_trace1 = Trace(info=mock_trace1_info, data=None)

    mock_trace2_info = TraceInfo(
        trace_id="trace-2",
        trace_location=TraceLocation.from_experiment_id("exp-456"),
        request_time=1000000100,
        state=TraceState.ERROR,
        execution_duration=200,
        assessments=[],
    )
    mock_trace2 = Trace(info=mock_trace2_info, data=None)

    with mock.patch("mlflow.search_traces", return_value=mock_search_traces_df) as mock_search:
        with mock.patch.object(Trace, "from_json", side_effect=[mock_trace1, mock_trace2]):
            result = tool.invoke(mock_trace, filter_string=None)

        assert mock_search.call_args[1]["filter_string"] is None

    assert len(result) == 2
