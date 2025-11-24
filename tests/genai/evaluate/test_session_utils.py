import os
from unittest.mock import Mock, patch

import pytest

import mlflow
from mlflow.entities import TraceData, TraceInfo, TraceLocation, TraceState
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai import scorer
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.session_utils import (
    classify_scorers,
    evaluate_multi_turn_scorers,
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
    """Test that all scorers are classified as single-turn when none are multi-turn."""

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
    """Test that all scorers are classified as multi-turn.

    When all scorers have is_session_level_scorer=True.
    """
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
    """Test classification of mixed single-turn and multi-turn scorers."""

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
    """Test classification of an empty list of scorers."""
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
    return eval_item


def test_group_traces_by_session_single_session():
    """Test grouping traces that all belong to a single session."""
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
    """Test grouping traces that belong to different sessions."""
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
    """Test that traces without session_id are excluded from grouping."""
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
    """Test that eval items without traces are excluded from grouping."""
    trace1 = _create_mock_trace("trace-1", "session-1", 1000)

    eval_item1 = _create_mock_eval_item(trace1)
    eval_item2 = Mock()
    eval_item2.trace = None  # No trace

    eval_items = [eval_item1, eval_item2]
    session_groups = group_traces_by_session(eval_items)

    assert len(session_groups) == 1
    assert "session-1" in session_groups
    assert len(session_groups["session-1"]) == 1


def test_group_traces_by_session_empty_list():
    """Test grouping an empty list of eval items."""
    session_groups = group_traces_by_session([])

    assert len(session_groups) == 0
    assert session_groups == {}


# ==================== Tests for get_first_trace_in_session ====================


def test_get_first_trace_in_session_chronological_order():
    """Test that the first trace is correctly identified by request_time."""
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
    """Test getting the first trace when there's only one trace."""
    trace1 = _create_mock_trace("trace-1", "session-1", 1000)
    eval_item1 = _create_mock_eval_item(trace1)

    session_items = [eval_item1]

    first_item = get_first_trace_in_session(session_items)

    assert first_item.trace == trace1
    assert first_item == eval_item1


def test_get_first_trace_in_session_same_timestamp():
    """Test behavior when multiple traces have the same timestamp."""
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
    """Test that validation passes when there are no session-level scorers."""

    @scorer
    def single_turn_scorer(outputs):
        return 1.0

    scorers_list = [single_turn_scorer]

    # Should not raise any exceptions
    validate_session_level_evaluation_inputs(
        scorers=scorers_list,
        predict_fn=None,
    )


def test_validate_session_level_evaluation_inputs_feature_flag_disabled():
    """Test that validation raises error when feature flag is disabled."""

    # Make sure feature flag is disabled
    os.environ.pop("MLFLOW_ENABLE_MULTI_TURN_EVALUATION", None)

    multi_turn_scorer = _MultiTurnTestScorer()
    scorers_list = [multi_turn_scorer]

    with pytest.raises(
        MlflowException,
        match="Multi-turn evaluation is not enabled",
    ):
        validate_session_level_evaluation_inputs(
            scorers=scorers_list,
            predict_fn=None,
        )


def test_validate_session_level_evaluation_inputs_with_predict_fn():
    """Test that validation raises error when predict_fn is provided with session-level scorers."""

    # Enable feature flag
    os.environ["MLFLOW_ENABLE_MULTI_TURN_EVALUATION"] = "true"

    try:
        multi_turn_scorer = _MultiTurnTestScorer()
        scorers_list = [multi_turn_scorer]

        def dummy_predict_fn():
            return "output"

        with pytest.raises(
            MlflowException,
            match="Multi-turn scorers are not yet supported with predict_fn",
        ):
            validate_session_level_evaluation_inputs(
                scorers=scorers_list,
                predict_fn=dummy_predict_fn,
            )
    finally:
        os.environ.pop("MLFLOW_ENABLE_MULTI_TURN_EVALUATION", None)


def test_validate_session_level_evaluation_inputs_valid():
    """Test that validation passes with valid session-level input."""

    # Enable feature flag
    os.environ["MLFLOW_ENABLE_MULTI_TURN_EVALUATION"] = "true"

    try:
        multi_turn_scorer = _MultiTurnTestScorer()
        scorers_list = [multi_turn_scorer]

        # Should not raise any exceptions
        validate_session_level_evaluation_inputs(
            scorers=scorers_list,
            predict_fn=None,
        )
    finally:
        os.environ.pop("MLFLOW_ENABLE_MULTI_TURN_EVALUATION", None)


def test_validate_session_level_evaluation_inputs_mixed_scorers():
    """Test validation with mixed single-turn and session-level scorers."""

    # Enable feature flag
    os.environ["MLFLOW_ENABLE_MULTI_TURN_EVALUATION"] = "true"

    try:
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
    finally:
        os.environ.pop("MLFLOW_ENABLE_MULTI_TURN_EVALUATION", None)


# ==================== Tests for evaluate_multi_turn_scorers ====================


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


def test_evaluate_multi_turn_scorers_basic():
    """Test basic multi-turn scorer evaluation"""
    # Create mock multi-turn scorer
    mock_scorer = Mock(spec=mlflow.genai.Scorer)
    mock_scorer.name = "test_scorer"
    mock_scorer.run.return_value = 0.8

    # Create session with traces
    session_groups = {
        "session1": [
            _create_eval_item("trace1", request_time=100),
            _create_eval_item("trace2", request_time=200),
        ]
    }

    with patch("mlflow.genai.evaluation.utils.standardize_scorer_value") as mock_standardize:
        from mlflow.entities.assessment import Feedback

        feedback = Feedback(
            name="test_scorer",
            source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="test"),
            value=0.8,
        )
        mock_standardize.return_value = [feedback]

        result = evaluate_multi_turn_scorers([mock_scorer], session_groups)

        # Verify scorer was called with session traces
        mock_scorer.run.assert_called_once()
        call_args = mock_scorer.run.call_args
        assert "session" in call_args.kwargs
        traces = call_args.kwargs["session"]
        assert len(traces) == 2

        # Verify result structure
        assert "trace1" in result
        assert "test_scorer" in result["trace1"]
        assert result["trace1"]["test_scorer"].value == 0.8


def test_evaluate_multi_turn_scorers_multiple_sessions():
    """Test evaluation across multiple sessions"""
    mock_scorer = Mock(spec=mlflow.genai.Scorer)
    mock_scorer.name = "session_scorer"
    mock_scorer.run.return_value = 1.0

    # Create multiple sessions
    session_groups = {
        "session1": [_create_eval_item("trace1", 100), _create_eval_item("trace2", 200)],
        "session2": [_create_eval_item("trace3", 150), _create_eval_item("trace4", 250)],
    }

    with patch("mlflow.genai.evaluation.utils.standardize_scorer_value") as mock_standardize:
        from mlflow.entities.assessment import Feedback

        feedback = Feedback(
            name="session_scorer",
            source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="test"),
            value=1.0,
        )
        mock_standardize.return_value = [feedback]

        result = evaluate_multi_turn_scorers([mock_scorer], session_groups)

        # Verify scorer was called for each session
        assert mock_scorer.run.call_count == 2

        # Verify results for both sessions
        assert "trace1" in result  # First trace of session1
        assert "trace3" in result  # First trace of session2
        assert result["trace1"]["session_scorer"].value == 1.0
        assert result["trace3"]["session_scorer"].value == 1.0


def test_evaluate_multi_turn_scorers_adds_session_metadata():
    """Test that session_id is added to feedback metadata"""
    mock_scorer = Mock(spec=mlflow.genai.Scorer)
    mock_scorer.name = "metadata_scorer"
    mock_scorer.run.return_value = 0.5

    session_groups = {
        "test_session_123": [_create_eval_item("trace1", 100)],
    }

    with patch("mlflow.genai.evaluation.utils.standardize_scorer_value") as mock_standardize:
        from mlflow.entities.assessment import Feedback

        # Feedback without metadata
        feedback = Feedback(
            name="metadata_scorer",
            source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="test"),
            value=0.5,
        )
        mock_standardize.return_value = [feedback]

        result = evaluate_multi_turn_scorers([mock_scorer], session_groups)

        # Verify session_id was added to metadata
        assert result["trace1"]["metadata_scorer"].metadata is not None
        assert (
            result["trace1"]["metadata_scorer"].metadata[TraceMetadataKey.TRACE_SESSION]
            == "test_session_123"
        )


def test_evaluate_multi_turn_scorers_handles_scorer_error():
    """Test error handling when scorer raises exception"""
    mock_scorer = Mock(spec=mlflow.genai.Scorer)
    mock_scorer.name = "failing_scorer"
    mock_scorer.run.side_effect = ValueError("Scorer failed!")

    session_groups = {
        "session1": [_create_eval_item("trace1", 100)],
    }

    result = evaluate_multi_turn_scorers([mock_scorer], session_groups)

    # Verify error feedback was created
    assert "trace1" in result
    assert "failing_scorer" in result["trace1"]
    feedback = result["trace1"]["failing_scorer"]
    assert feedback.error is not None
    assert feedback.error.error_code == "SCORER_ERROR"
    assert "Scorer failed!" in feedback.error.error_message
    assert feedback.error.stack_trace is not None


def test_evaluate_multi_turn_scorers_multiple_feedbacks_per_scorer():
    """Test scorer returning multiple feedbacks"""
    mock_scorer = Mock(spec=mlflow.genai.Scorer)
    mock_scorer.name = "multi_feedback_scorer"
    mock_scorer.run.return_value = {"metric1": 0.7, "metric2": 0.9}

    session_groups = {
        "session1": [_create_eval_item("trace1", 100)],
    }

    with patch("mlflow.genai.evaluation.utils.standardize_scorer_value") as mock_standardize:
        from mlflow.entities.assessment import Feedback

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

        result = evaluate_multi_turn_scorers([mock_scorer], session_groups)

        # Verify both feedbacks are stored
        assert "trace1" in result
        assert "multi_feedback_scorer/metric1" in result["trace1"]
        assert "multi_feedback_scorer/metric2" in result["trace1"]
        assert result["trace1"]["multi_feedback_scorer/metric1"].value == 0.7
        assert result["trace1"]["multi_feedback_scorer/metric2"].value == 0.9


def test_evaluate_multi_turn_scorers_first_trace_selection():
    """Test that assessments are stored on the chronologically first trace"""
    mock_scorer = Mock(spec=mlflow.genai.Scorer)
    mock_scorer.name = "first_trace_scorer"
    mock_scorer.run.return_value = 1.0

    # Create session with traces in non-chronological order
    session_groups = {
        "session1": [
            _create_eval_item("trace2", request_time=200),  # Second chronologically
            _create_eval_item("trace1", request_time=100),  # First chronologically
            _create_eval_item("trace3", request_time=300),  # Third chronologically
        ]
    }

    with patch("mlflow.genai.evaluation.utils.standardize_scorer_value") as mock_standardize:
        from mlflow.entities.assessment import Feedback

        feedback = Feedback(
            name="first_trace_scorer",
            source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="test"),
            value=1.0,
        )
        mock_standardize.return_value = [feedback]

        result = evaluate_multi_turn_scorers([mock_scorer], session_groups)

        # Verify assessment is stored on trace1 (earliest request_time)
        assert "trace1" in result
        assert "trace2" not in result
        assert "trace3" not in result
        assert result["trace1"]["first_trace_scorer"].value == 1.0


def test_evaluate_multi_turn_scorers_multiple_scorers():
    """Test evaluation with multiple multi-turn scorers"""
    mock_scorer1 = Mock(spec=mlflow.genai.Scorer)
    mock_scorer1.name = "scorer1"
    mock_scorer1.run.return_value = 0.6

    mock_scorer2 = Mock(spec=mlflow.genai.Scorer)
    mock_scorer2.name = "scorer2"
    mock_scorer2.run.return_value = 0.8

    session_groups = {
        "session1": [_create_eval_item("trace1", 100)],
    }

    with patch("mlflow.genai.evaluation.utils.standardize_scorer_value") as mock_standardize:
        from mlflow.entities.assessment import Feedback

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

        mock_standardize.side_effect = [create_feedback("scorer1", 0.6), create_feedback("scorer2", 0.8)]

        result = evaluate_multi_turn_scorers([mock_scorer1, mock_scorer2], session_groups)

        # Verify both scorers were evaluated
        assert "trace1" in result
        assert "scorer1" in result["trace1"]
        assert "scorer2" in result["trace1"]
        assert result["trace1"]["scorer1"].value == 0.6
        assert result["trace1"]["scorer2"].value == 0.8


def test_evaluate_multi_turn_scorers_empty_session_groups():
    """Test evaluation with empty session groups"""
    mock_scorer = Mock(spec=mlflow.genai.Scorer)
    mock_scorer.name = "test_scorer"

    result = evaluate_multi_turn_scorers([mock_scorer], {})

    # Should return empty result
    assert result == {}
    mock_scorer.run.assert_not_called()
