from unittest.mock import Mock, patch

import pytest

import mlflow
from mlflow.entities import TraceData, TraceInfo, TraceLocation, TraceState
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai import scorer
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.session_utils import (
    classify_scorers,
    evaluate_session_level_scorers,
    get_first_trace_in_session,
    group_traces_by_session,
    validate_session_level_evaluation_inputs,
)
from mlflow.tracing.constant import TraceMetadataKey


class _MultiTurnTestScorer:
    """Helper class for testing multi-turn scorers."""

    def __init__(self, name="test_multi_turn_scorer"):
        self.name = name
        self.is_session_level_scorer = True
        self.aggregations = []

    def run(self, session=None, **kwargs):
        return True

    def __call__(self, traces=None, **kwargs):
        return 1.0


# ==================== Tests for classify_scorers ====================


def test_classify_scorers_all_single_turn():
    @scorer
    def custom_scorer1(outputs):
        return 1.0

    @scorer
    def custom_scorer2(outputs):
        return 2.0

    scorers_list = [custom_scorer1, custom_scorer2]
    single_turn, multi_turn = classify_scorers(scorers_list)

    assert len(single_turn) == 2
    assert len(multi_turn) == 0
    assert single_turn == scorers_list


def test_classify_scorers_all_multi_turn():
    multi_turn_scorer1 = _MultiTurnTestScorer(name="multi_turn_scorer1")
    multi_turn_scorer2 = _MultiTurnTestScorer(name="multi_turn_scorer2")

    scorers_list = [multi_turn_scorer1, multi_turn_scorer2]
    single_turn, multi_turn = classify_scorers(scorers_list)

    assert len(single_turn) == 0
    assert len(multi_turn) == 2
    assert multi_turn == scorers_list
    # Verify they are actually multi-turn
    assert multi_turn_scorer1.is_session_level_scorer is True
    assert multi_turn_scorer2.is_session_level_scorer is True


def test_classify_scorers_mixed():
    @scorer
    def single_turn_scorer(outputs):
        return 1.0

    multi_turn_scorer = _MultiTurnTestScorer(name="multi_turn_scorer")

    scorers_list = [single_turn_scorer, multi_turn_scorer]
    single_turn, multi_turn = classify_scorers(scorers_list)

    assert len(single_turn) == 1
    assert len(multi_turn) == 1
    assert single_turn[0] == single_turn_scorer
    assert multi_turn[0] == multi_turn_scorer
    # Verify properties
    assert single_turn_scorer.is_session_level_scorer is False
    assert multi_turn_scorer.is_session_level_scorer is True


def test_classify_scorers_empty_list():
    single_turn, multi_turn = classify_scorers([])

    assert len(single_turn) == 0
    assert len(multi_turn) == 0


# ==================== Tests for group_traces_by_session ====================


def _create_mock_trace(trace_id: str, session_id: str | None, request_time: int):
    """Helper to create a mock trace with session_id and request_time."""
    trace_metadata = {}
    if session_id is not None:
        trace_metadata[TraceMetadataKey.TRACE_SESSION] = session_id

    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=request_time,
        execution_duration=1000,
        state=TraceState.OK,
        trace_metadata=trace_metadata,
        tags={},
    )

    trace = Mock(spec=Trace)
    trace.info = trace_info
    trace.data = TraceData(spans=[])
    return trace


def _create_mock_eval_item(trace):
    """Helper to create a mock EvalItem with a trace."""
    eval_item = Mock(spec=EvalItem)
    eval_item.trace = trace
    eval_item.source = None  # Explicitly set to None so it doesn't return a Mock
    return eval_item


def test_group_traces_by_session_single_session():
    trace1 = _create_mock_trace("trace-1", "session-1", 1000)
    trace2 = _create_mock_trace("trace-2", "session-1", 2000)
    trace3 = _create_mock_trace("trace-3", "session-1", 3000)

    eval_item1 = _create_mock_eval_item(trace1)
    eval_item2 = _create_mock_eval_item(trace2)
    eval_item3 = _create_mock_eval_item(trace3)

    eval_items = [eval_item1, eval_item2, eval_item3]
    session_groups = group_traces_by_session(eval_items)

    assert len(session_groups) == 1
    assert "session-1" in session_groups
    assert len(session_groups["session-1"]) == 3

    # Check that all traces are included
    session_traces = [item.trace for item in session_groups["session-1"]]
    assert trace1 in session_traces
    assert trace2 in session_traces
    assert trace3 in session_traces


def test_group_traces_by_session_multiple_sessions():
    trace1 = _create_mock_trace("trace-1", "session-1", 1000)
    trace2 = _create_mock_trace("trace-2", "session-1", 2000)
    trace3 = _create_mock_trace("trace-3", "session-2", 1500)
    trace4 = _create_mock_trace("trace-4", "session-2", 2500)

    eval_items = [
        _create_mock_eval_item(trace1),
        _create_mock_eval_item(trace2),
        _create_mock_eval_item(trace3),
        _create_mock_eval_item(trace4),
    ]

    session_groups = group_traces_by_session(eval_items)

    assert len(session_groups) == 2
    assert "session-1" in session_groups
    assert "session-2" in session_groups
    assert len(session_groups["session-1"]) == 2
    assert len(session_groups["session-2"]) == 2


def test_group_traces_by_session_excludes_no_session_id():
    trace1 = _create_mock_trace("trace-1", "session-1", 1000)
    trace2 = _create_mock_trace("trace-2", None, 2000)  # No session_id
    trace3 = _create_mock_trace("trace-3", "session-1", 3000)

    eval_items = [
        _create_mock_eval_item(trace1),
        _create_mock_eval_item(trace2),
        _create_mock_eval_item(trace3),
    ]

    session_groups = group_traces_by_session(eval_items)

    assert len(session_groups) == 1
    assert "session-1" in session_groups
    assert len(session_groups["session-1"]) == 2
    # trace2 should not be included
    session_traces = [item.trace for item in session_groups["session-1"]]
    assert trace1 in session_traces
    assert trace2 not in session_traces
    assert trace3 in session_traces


def test_group_traces_by_session_excludes_none_traces():
    trace1 = _create_mock_trace("trace-1", "session-1", 1000)

    eval_item1 = _create_mock_eval_item(trace1)
    eval_item2 = Mock()
    eval_item2.trace = None  # No trace
    eval_item2.source = None  # No source

    eval_items = [eval_item1, eval_item2]
    session_groups = group_traces_by_session(eval_items)

    assert len(session_groups) == 1
    assert "session-1" in session_groups
    assert len(session_groups["session-1"]) == 1


def test_group_traces_by_session_empty_list():
    session_groups = group_traces_by_session([])

    assert len(session_groups) == 0
    assert session_groups == {}


# ==================== Tests for get_first_trace_in_session ====================


def test_get_first_trace_in_session_chronological_order():
    trace1 = _create_mock_trace("trace-1", "session-1", 3000)
    trace2 = _create_mock_trace("trace-2", "session-1", 1000)  # Earliest
    trace3 = _create_mock_trace("trace-3", "session-1", 2000)

    eval_item1 = _create_mock_eval_item(trace1)
    eval_item2 = _create_mock_eval_item(trace2)
    eval_item3 = _create_mock_eval_item(trace3)

    session_items = [eval_item1, eval_item2, eval_item3]

    first_item = get_first_trace_in_session(session_items)

    assert first_item.trace == trace2
    assert first_item == eval_item2


def test_get_first_trace_in_session_single_trace():
    trace1 = _create_mock_trace("trace-1", "session-1", 1000)
    eval_item1 = _create_mock_eval_item(trace1)

    session_items = [eval_item1]

    first_item = get_first_trace_in_session(session_items)

    assert first_item.trace == trace1
    assert first_item == eval_item1


def test_get_first_trace_in_session_same_timestamp():
    # When timestamps are equal, min() will return the first one in the list
    trace1 = _create_mock_trace("trace-1", "session-1", 1000)
    trace2 = _create_mock_trace("trace-2", "session-1", 1000)
    trace3 = _create_mock_trace("trace-3", "session-1", 1000)

    eval_item1 = _create_mock_eval_item(trace1)
    eval_item2 = _create_mock_eval_item(trace2)
    eval_item3 = _create_mock_eval_item(trace3)

    session_items = [eval_item1, eval_item2, eval_item3]

    first_item = get_first_trace_in_session(session_items)

    # Should return one of the traces with timestamp 1000 (likely the first one)
    assert first_item.trace.info.request_time == 1000


# ==================== Tests for validate_session_level_evaluation_inputs ====================


def test_validate_session_level_evaluation_inputs_no_session_level_scorers():
    @scorer
    def single_turn_scorer(outputs):
        return 1.0

    scorers_list = [single_turn_scorer]

    # Should not raise any exceptions
    validate_session_level_evaluation_inputs(
        scorers=scorers_list,
        predict_fn=None,
    )


def test_validate_session_level_evaluation_inputs_with_predict_fn():
    multi_turn_scorer = _MultiTurnTestScorer()
    scorers_list = [multi_turn_scorer]

    def dummy_predict_fn():
        return "output"

    with pytest.raises(
        MlflowException,
        match=r"Multi-turn scorers are not yet supported with predict_fn.*"
        r"Please pass existing traces containing session IDs to `data`",
    ):
        validate_session_level_evaluation_inputs(
            scorers=scorers_list,
            predict_fn=dummy_predict_fn,
        )


def test_validate_session_level_evaluation_inputs_mixed_scorers():
    @scorer
    def single_turn_scorer(outputs):
        return 1.0

    multi_turn_scorer = _MultiTurnTestScorer()
    scorers_list = [single_turn_scorer, multi_turn_scorer]

    # Should not raise any exceptions
    validate_session_level_evaluation_inputs(
        scorers=scorers_list,
        predict_fn=None,
    )


# ==================== Tests for evaluate_session_level_scorers ====================


def _create_test_trace(trace_id: str, request_time: int = 0) -> Trace:
    """Helper to create a minimal test trace"""
    return Trace(
        info=TraceInfo(
            trace_id=trace_id,
            trace_location=TraceLocation.from_experiment_id("0"),
            request_time=request_time,
            execution_duration=100,
            state=TraceState.OK,
            trace_metadata={},
            tags={},
        ),
        data=TraceData(spans=[]),
    )


def _create_eval_item(trace_id: str, request_time: int = 0) -> EvalItem:
    """Helper to create a minimal EvalItem with a trace"""
    trace = _create_test_trace(trace_id, request_time)
    return EvalItem(
        request_id=trace_id,
        trace=trace,
        inputs={},
        outputs={},
        expectations={},
    )


def test_evaluate_session_level_scorers_success():
    mock_scorer = Mock(spec=mlflow.genai.Scorer)
    mock_scorer.name = "test_scorer"
    mock_scorer.run.return_value = 0.8

    # Test with a single session containing multiple traces
    session_items = [
        _create_eval_item("trace1", request_time=100),
        _create_eval_item("trace2", request_time=200),
    ]

    with patch(
        "mlflow.genai.evaluation.session_utils.standardize_scorer_value"
    ) as mock_standardize:
        # Return a new Feedback object each time to avoid metadata overwriting
        def create_feedback(*args, **kwargs):
            return [
                Feedback(
                    name="test_scorer",
                    source=AssessmentSource(
                        source_type=AssessmentSourceType.CODE, source_id="test"
                    ),
                    value=0.8,
                )
            ]

        mock_standardize.side_effect = create_feedback

        result = evaluate_session_level_scorers("session1", session_items, [mock_scorer])

        # Verify scorer was called once (for the single session)
        assert mock_scorer.run.call_count == 1

        # Verify scorer received session traces
        call_args = mock_scorer.run.call_args
        assert "session" in call_args.kwargs
        assert len(call_args.kwargs["session"]) == 2  # session has 2 traces

        # Verify result contains assessments for first trace
        assert "trace1" in result  # First trace (earliest timestamp)
        assert len(result["trace1"]) == 1
        assert result["trace1"][0].name == "test_scorer"
        assert result["trace1"][0].value == 0.8

        # Verify session_id was added to metadata
        assert result["trace1"][0].metadata is not None
        assert result["trace1"][0].metadata[TraceMetadataKey.TRACE_SESSION] == "session1"


def test_evaluate_session_level_scorers_handles_scorer_error():
    mock_scorer = Mock(spec=mlflow.genai.Scorer)
    mock_scorer.name = "failing_scorer"
    mock_scorer.run.side_effect = ValueError("Scorer failed!")

    session_items = [_create_eval_item("trace1", 100)]

    result = evaluate_session_level_scorers("session1", session_items, [mock_scorer])

    # Verify error feedback was created
    assert "trace1" in result
    assert len(result["trace1"]) == 1
    feedback = result["trace1"][0]
    assert feedback.name == "failing_scorer"
    assert feedback.error is not None
    assert feedback.error.error_code == "SCORER_ERROR"
    assert feedback.error.stack_trace is not None

    assert feedback.error.to_proto().error_message == "Scorer failed!"
    assert isinstance(feedback.error.error_message, str)
    assert feedback.error.error_message == "Scorer failed!"


def test_evaluate_session_level_scorers_multiple_feedbacks_per_scorer():
    mock_scorer = Mock(spec=mlflow.genai.Scorer)
    mock_scorer.name = "multi_feedback_scorer"
    mock_scorer.run.return_value = {"metric1": 0.7, "metric2": 0.9}

    session_items = [_create_eval_item("trace1", 100)]

    with patch(
        "mlflow.genai.evaluation.session_utils.standardize_scorer_value"
    ) as mock_standardize:
        feedbacks = [
            Feedback(
                name="multi_feedback_scorer/metric1",
                source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="test"),
                value=0.7,
            ),
            Feedback(
                name="multi_feedback_scorer/metric2",
                source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="test"),
                value=0.9,
            ),
        ]
        mock_standardize.return_value = feedbacks

        result = evaluate_session_level_scorers("session1", session_items, [mock_scorer])

        # Verify both feedbacks are stored
        assert "trace1" in result
        assert len(result["trace1"]) == 2
        # Find feedbacks by name
        feedback_by_name = {f.name: f for f in result["trace1"]}
        assert "multi_feedback_scorer/metric1" in feedback_by_name
        assert "multi_feedback_scorer/metric2" in feedback_by_name
        assert feedback_by_name["multi_feedback_scorer/metric1"].value == 0.7
        assert feedback_by_name["multi_feedback_scorer/metric2"].value == 0.9


def test_evaluate_session_level_scorers_first_trace_selection():
    mock_scorer = Mock(spec=mlflow.genai.Scorer)
    mock_scorer.name = "first_trace_scorer"
    mock_scorer.run.return_value = 1.0

    # Create session with traces in non-chronological order
    session_items = [
        _create_eval_item("trace2", request_time=200),  # Second chronologically
        _create_eval_item("trace1", request_time=100),  # First chronologically
        _create_eval_item("trace3", request_time=300),  # Third chronologically
    ]

    with patch(
        "mlflow.genai.evaluation.session_utils.standardize_scorer_value"
    ) as mock_standardize:
        feedback = Feedback(
            name="first_trace_scorer",
            source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="test"),
            value=1.0,
        )
        mock_standardize.return_value = [feedback]

        result = evaluate_session_level_scorers("session1", session_items, [mock_scorer])

        # Verify assessment is stored on trace1 (earliest request_time)
        assert "trace1" in result
        assert "trace2" not in result
        assert "trace3" not in result
        assert len(result["trace1"]) == 1
        assert result["trace1"][0].name == "first_trace_scorer"
        assert result["trace1"][0].value == 1.0


def test_evaluate_session_level_scorers_multiple_scorers():
    mock_scorer1 = Mock(spec=mlflow.genai.Scorer)
    mock_scorer1.name = "scorer1"
    mock_scorer1.run.return_value = 0.6

    mock_scorer2 = Mock(spec=mlflow.genai.Scorer)
    mock_scorer2.name = "scorer2"
    mock_scorer2.run.return_value = 0.8

    session_items = [_create_eval_item("trace1", 100)]

    with patch(
        "mlflow.genai.evaluation.session_utils.standardize_scorer_value"
    ) as mock_standardize:

        def create_feedback(name, value):
            return [
                Feedback(
                    name=name,
                    source=AssessmentSource(
                        source_type=AssessmentSourceType.CODE, source_id="test"
                    ),
                    value=value,
                )
            ]

        mock_standardize.side_effect = [
            create_feedback("scorer1", 0.6),
            create_feedback("scorer2", 0.8),
        ]

        result = evaluate_session_level_scorers(
            "session1", session_items, [mock_scorer1, mock_scorer2]
        )

        # Verify both scorers were evaluated (runs in parallel)
        assert mock_scorer1.run.call_count == 1
        assert mock_scorer2.run.call_count == 1

        # Verify result contains assessments from both scorers
        assert "trace1" in result
        assert len(result["trace1"]) == 2
        # Find feedbacks by name
        feedback_by_name = {f.name: f for f in result["trace1"]}
        assert "scorer1" in feedback_by_name
        assert "scorer2" in feedback_by_name
        assert feedback_by_name["scorer1"].value == 0.6
        assert feedback_by_name["scorer2"].value == 0.8
