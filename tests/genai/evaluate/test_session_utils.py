import json
from unittest import mock
from unittest.mock import Mock, patch

import pytest

import mlflow
from mlflow.entities import TraceData, TraceInfo, TraceLocation, TraceState
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_location import TraceLocationType, UCSchemaLocation
from mlflow.exceptions import MlflowException
from mlflow.genai import Scorer, scorer
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.session_utils import (
    classify_scorers,
    evaluate_session_level_scorers,
    get_first_trace_in_session,
    group_sessions_by_scorer_level,
    group_traces_by_session,
    resolve_session_level_index,
    validate_session_level_evaluation_inputs,
)
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracking.client import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_EXPERIMENT_SESSION_HIERARCHY


class _MultiTurnTestScorer(Scorer):
    """Helper class for testing multi-turn scorers."""

    def __init__(self, name="test_multi_turn_scorer"):
        super().__init__(name=name, aggregations=[])

    @property
    def is_session_level_scorer(self) -> bool:
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


# ==================== Tests for hierarchy-level grouping ====================


def _path(*levels):
    return json.dumps(list(levels), separators=(",", ":"))


def test_group_traces_by_session_at_level_groups_by_prefix():
    trace1 = _create_mock_trace("trace-1", _path("trip-2", "ep-2", "dom-2"), 1000)
    trace2 = _create_mock_trace("trace-2", _path("trip-2", "ep-9", "dom-1"), 2000)
    trace3 = _create_mock_trace("trace-3", _path("trip-3", "ep-1", "dom-1"), 3000)

    eval_items = [
        _create_mock_eval_item(trace1),
        _create_mock_eval_item(trace2),
        _create_mock_eval_item(trace3),
    ]

    # level_index=0 ("trip"): group by the first element only.
    trip_groups = group_traces_by_session(eval_items, level_index=0)
    assert set(trip_groups) == {_path("trip-2"), _path("trip-3")}
    assert [item.trace for item in trip_groups[_path("trip-2")]] == [trace1, trace2]

    # level_index=1 ("episode"): group by the first two elements.
    episode_groups = group_traces_by_session(eval_items, level_index=1)
    assert set(episode_groups) == {
        _path("trip-2", "ep-2"),
        _path("trip-2", "ep-9"),
        _path("trip-3", "ep-1"),
    }


def test_group_traces_by_session_at_level_plain_and_short_paths_form_own_groups():
    plain = _create_mock_trace("trace-1", "plain-session", 1000)
    short = _create_mock_trace("trace-2", _path("trip-2"), 2000)
    deeper = _create_mock_trace("trace-3", _path("trip-2", "ep-2"), 3000)

    eval_items = [
        _create_mock_eval_item(plain),
        _create_mock_eval_item(short),
        _create_mock_eval_item(deeper),
    ]

    # At the episode level (index 1), a plain session and a path shorter than the
    # level are their own groups, keyed by the original value.
    groups = group_traces_by_session(eval_items, level_index=1)
    assert set(groups) == {"plain-session", _path("trip-2"), _path("trip-2", "ep-2")}


def test_group_traces_by_session_without_level_is_full_value():
    trace1 = _create_mock_trace("trace-1", _path("trip-2", "ep-2"), 1000)
    trace2 = _create_mock_trace("trace-2", _path("trip-2", "ep-9"), 2000)

    eval_items = [_create_mock_eval_item(trace1), _create_mock_eval_item(trace2)]
    assert set(group_traces_by_session(eval_items)) == {
        _path("trip-2", "ep-2"),
        _path("trip-2", "ep-9"),
    }


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
        match=r"Session-level scorers require traces with session IDs.*"
        r"Either pass a ConversationSimulator to `data` with `predict_fn`",
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

        # Verify result is for first item
        assert result.eval_item.trace.info.trace_id == "trace1"
        assert len(result.assessments) == 1
        assert result.assessments[0].name == "test_scorer"
        assert result.assessments[0].value == 0.8

        # Verify session_id was added to metadata
        assert result.assessments[0].metadata is not None
        assert result.assessments[0].metadata[TraceMetadataKey.TRACE_SESSION] == "session1"


def test_evaluate_session_level_scorers_handles_scorer_error():
    mock_scorer = Mock(spec=mlflow.genai.Scorer)
    mock_scorer.name = "failing_scorer"
    mock_scorer.run.side_effect = ValueError("Scorer failed!")

    session_items = [_create_eval_item("trace1", 100)]

    result = evaluate_session_level_scorers("session1", session_items, [mock_scorer])

    # Verify error feedback was created
    assert result.eval_item.trace.info.trace_id == "trace1"
    assert len(result.assessments) == 1
    feedback = result.assessments[0]
    assert feedback.name == "failing_scorer"
    assert feedback.error is not None
    assert feedback.error.error_code == "SCORER_ERROR"
    assert feedback.error.stack_trace is not None

    assert feedback.error.to_proto().error_message == "Scorer failed!"
    assert isinstance(feedback.error.error_message, str)
    assert feedback.error.error_message == "Scorer failed!"

    # Verify session_id metadata is present even on error feedbacks
    assert feedback.metadata is not None
    assert feedback.metadata[TraceMetadataKey.TRACE_SESSION] == "session1"


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
        assert result.eval_item.trace.info.trace_id == "trace1"
        assert len(result.assessments) == 2
        # Find feedbacks by name
        feedback_by_name = {f.name: f for f in result.assessments}
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

        # Verify assessment is for trace1 (earliest request_time)
        assert result.eval_item.trace.info.trace_id == "trace1"
        assert len(result.assessments) == 1
        assert result.assessments[0].name == "first_trace_scorer"
        assert result.assessments[0].value == 1.0


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
        assert result.eval_item.trace.info.trace_id == "trace1"
        assert len(result.assessments) == 2
        # Find feedbacks by name
        feedback_by_name = {f.name: f for f in result.assessments}
        assert "scorer1" in feedback_by_name
        assert "scorer2" in feedback_by_name
        assert feedback_by_name["scorer1"].value == 0.6
        assert feedback_by_name["scorer2"].value == 0.8


def test_evaluate_session_level_scorers_error_multiple_traces():
    mock_scorer = Mock(spec=mlflow.genai.Scorer)
    mock_scorer.name = "failing_scorer"
    mock_scorer.run.side_effect = RuntimeError("boom")

    session_items = [
        _create_eval_item("trace1", request_time=100),
        _create_eval_item("trace2", request_time=200),
    ]

    result = evaluate_session_level_scorers("session-abc", session_items, [mock_scorer])

    assert result.eval_item.trace.info.trace_id == "trace1"
    feedback = result.assessments[0]
    assert feedback.error is not None
    assert feedback.metadata[TraceMetadataKey.TRACE_SESSION] == "session-abc"


# ==================== Tests for resolve_session_level_index ====================


def _set_hierarchy(experiment_id, levels):
    tag_value = json.dumps(
        {"version": 1, "levels": [{"name": level} for level in levels]},
        separators=(",", ":"),
    )
    MlflowClient().set_experiment_tag(experiment_id, MLFLOW_EXPERIMENT_SESSION_HIERARCHY, tag_value)


def _create_trace_in_experiment(trace_id, experiment_id, request_time=1000):
    # A session trace: level resolution only considers items that carry a session.
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=TraceLocation.from_experiment_id(experiment_id),
        request_time=request_time,
        execution_duration=100,
        state=TraceState.OK,
        trace_metadata={TraceMetadataKey.TRACE_SESSION: "session-1"},
        tags={},
    )
    trace = Mock(spec=Trace)
    trace.info = trace_info
    trace.data = TraceData(spans=[])
    return _create_mock_eval_item(trace)


def test_resolve_session_level_index_resolves_position():
    experiment_id = mlflow.set_experiment("hierarchy-resolve").experiment_id
    _set_hierarchy(experiment_id, ["trip", "episode", "domain"])
    items = [_create_trace_in_experiment("t1", experiment_id)]

    index = resolve_session_level_index("episode", items, experiment_id, ["my_scorer"])
    assert index == 1


def test_resolve_session_level_index_falls_back_to_run_experiment():
    # Traces without an experiment (e.g. UC-located) fall back to the run's experiment.
    experiment_id = mlflow.set_experiment("hierarchy-fallback").experiment_id
    _set_hierarchy(experiment_id, ["trip", "episode"])
    trace_info = TraceInfo(
        trace_id="uc-trace",
        trace_location=TraceLocation(
            type=TraceLocationType.UC_SCHEMA,
            uc_schema=UCSchemaLocation(catalog_name="cat", schema_name="schema"),
        ),
        request_time=1000,
        execution_duration=100,
        state=TraceState.OK,
        trace_metadata={},
        tags={},
    )
    item = Mock(spec=EvalItem)
    item.trace = Mock(spec=Trace)
    item.trace.info = trace_info
    item.trace.data = TraceData(spans=[])

    assert trace_info.experiment_id is None
    assert resolve_session_level_index("trip", [item], experiment_id, ["my_scorer"]) == 0


def test_resolve_session_level_index_missing_tag():
    experiment_id = mlflow.set_experiment("hierarchy-missing-tag").experiment_id
    items = [_create_trace_in_experiment("t1", experiment_id)]

    with pytest.raises(MlflowException, match="does not have a valid"):
        resolve_session_level_index("episode", items, experiment_id, ["my_scorer"])


def test_resolve_session_level_index_invalid_tag():
    experiment_id = mlflow.set_experiment("hierarchy-invalid-tag").experiment_id
    MlflowClient().set_experiment_tag(
        experiment_id, MLFLOW_EXPERIMENT_SESSION_HIERARCHY, "not-json"
    )
    items = [_create_trace_in_experiment("t1", experiment_id)]

    with pytest.raises(MlflowException, match="does not have a valid"):
        resolve_session_level_index("episode", items, experiment_id, ["my_scorer"])


def test_resolve_session_level_index_unknown_level():
    experiment_id = mlflow.set_experiment("hierarchy-unknown-level").experiment_id
    _set_hierarchy(experiment_id, ["trip"])
    items = [_create_trace_in_experiment("t1", experiment_id)]

    with pytest.raises(MlflowException, match="not declared by experiment"):
        resolve_session_level_index("episode", items, experiment_id, ["my_scorer"])


def test_resolve_session_level_index_conflicting_positions_across_experiments():
    exp_a = mlflow.set_experiment("hierarchy-conflict-a").experiment_id
    exp_b = mlflow.set_experiment("hierarchy-conflict-b").experiment_id
    _set_hierarchy(exp_a, ["trip", "episode"])
    _set_hierarchy(exp_b, ["episode", "trip"])
    items = [
        _create_trace_in_experiment("t1", exp_a),
        _create_trace_in_experiment("t2", exp_b),
    ]

    with pytest.raises(MlflowException, match="different positions"):
        resolve_session_level_index("episode", items, exp_a, ["my_scorer"])


# ==================== Tests for group_sessions_by_scorer_level ====================


def test_group_sessions_by_scorer_level_buckets_by_level():
    experiment_id = mlflow.set_experiment("hierarchy-buckets").experiment_id
    _set_hierarchy(experiment_id, ["trip", "episode"])
    trace1 = _create_mock_trace("t1", _path("trip-2", "ep-2", "dom-2"), 1000)
    trace2 = _create_mock_trace("t2", _path("trip-2", "ep-2", "dom-9"), 2000)
    trace3 = _create_mock_trace("t3", _path("trip-2", "ep-9", "dom-1"), 3000)
    trace4 = _create_mock_trace("t4", _path("trip-3", "ep-1", "dom-1"), 4000)
    for trace in [trace1, trace2, trace3, trace4]:
        trace.info.trace_location = TraceLocation.from_experiment_id(experiment_id)
    eval_items = [_create_mock_eval_item(trace) for trace in [trace1, trace2, trace3, trace4]]

    @scorer
    def full_session_scorer(session):
        return len(session)

    @scorer(session_level="trip")
    def trip_scorer(session):
        return len(session)

    @scorer(session_level="episode")
    def episode_scorer(session):
        return len(session)

    buckets = group_sessions_by_scorer_level(
        eval_items, [full_session_scorer, trip_scorer, episode_scorer], experiment_id
    )

    buckets_by_level = {
        (scorers[0].session_level if scorers else None): (groups, scorers)
        for groups, scorers in buckets
    }
    assert set(buckets_by_level) == {None, "trip", "episode"}

    # None bucket: full-value grouping.
    assert set(buckets_by_level[None][0]) == {
        _path("trip-2", "ep-2", "dom-2"),
        _path("trip-2", "ep-2", "dom-9"),
        _path("trip-2", "ep-9", "dom-1"),
        _path("trip-3", "ep-1", "dom-1"),
    }
    # trip bucket: first element only.
    assert set(buckets_by_level["trip"][0]) == {_path("trip-2"), _path("trip-3")}
    # episode bucket: first two elements.
    assert set(buckets_by_level["episode"][0]) == {
        _path("trip-2", "ep-2"),
        _path("trip-2", "ep-9"),
        _path("trip-3", "ep-1"),
    }


def test_group_sessions_by_scorer_level_without_levels_keeps_full_grouping():
    experiment_id = mlflow.set_experiment("hierarchy-no-levels").experiment_id
    trace1 = _create_mock_trace("t1", "session-1", 1000)
    trace2 = _create_mock_trace("t2", "session-1", 2000)
    eval_items = [_create_mock_eval_item(trace1), _create_mock_eval_item(trace2)]

    @scorer
    def full_session_scorer(session):
        return len(session)

    buckets = group_sessions_by_scorer_level(eval_items, [full_session_scorer], experiment_id)

    assert len(buckets) == 1
    groups, scorers = buckets[0]
    assert set(groups) == {"session-1"}
    assert scorers == [full_session_scorer]


# ==================== Tests for session_level validation ====================


def test_validate_inputs_rejects_session_level_on_single_turn_scorer():
    @scorer
    def single_turn_scorer(outputs):
        return True

    single_turn_scorer.session_level = "episode"

    with pytest.raises(MlflowException, match="not a session-level scorer"):
        validate_session_level_evaluation_inputs(
            scorers=[single_turn_scorer],
            predict_fn=None,
        )


def test_validate_inputs_accepts_session_level_on_session_scorer():
    @scorer(session_level="episode")
    def session_scorer(session):
        return len(session)

    # Should not raise.
    validate_session_level_evaluation_inputs(scorers=[session_scorer], predict_fn=None)


def test_resolve_session_level_index_ignores_untagged_run_experiment():
    # Traces live in a tagged experiment; the run's own experiment is untagged and
    # must not be resolved.
    tagged_exp = mlflow.set_experiment("hierarchy-tagged-exp").experiment_id
    _set_hierarchy(tagged_exp, ["trip", "episode"])
    untagged_exp = mlflow.set_experiment("hierarchy-untagged-run-exp").experiment_id
    items = [_create_trace_in_experiment("t1", tagged_exp)]

    index = resolve_session_level_index(
        "episode", items, run_experiment_id=untagged_exp, scorer_names=["my_scorer"]
    )
    assert index == 1


def test_resolve_session_level_index_item_without_trace_falls_back_to_run_experiment():
    # A dataset-record item (no trace, session from the record) resolves against
    # the run's experiment.
    run_exp = mlflow.set_experiment("hierarchy-no-trace-fallback").experiment_id
    _set_hierarchy(run_exp, ["trip", "episode"])
    item = Mock(spec=EvalItem)
    item.trace = None
    item.source = Mock(source_data={"session_id": "session-1"})

    index = resolve_session_level_index(
        "episode", [item], run_experiment_id=run_exp, scorer_names=["my_scorer"]
    )
    assert index == 1


def test_resolve_session_level_index_untagged_trace_experiment_still_errors():
    # A trace in an untagged experiment fails on that experiment even when the run's
    # experiment is tagged: the trace's own hierarchy is what applies.
    untagged_exp = mlflow.set_experiment("hierarchy-untagged-trace-exp").experiment_id
    tagged_run_exp = mlflow.set_experiment("hierarchy-tagged-run-exp").experiment_id
    _set_hierarchy(tagged_run_exp, ["trip", "episode"])
    items = [_create_trace_in_experiment("t1", untagged_exp)]

    with pytest.raises(MlflowException, match="does not have a valid"):
        resolve_session_level_index(
            "episode", items, run_experiment_id=tagged_run_exp, scorer_names=["my_scorer"]
        )


def test_group_sessions_by_scorer_level_caches_hierarchy_per_experiment():
    # Resolving several levels in one run fetches each experiment's tag once.
    experiment_id = mlflow.set_experiment("hierarchy-cache-memoization").experiment_id
    _set_hierarchy(experiment_id, ["trip", "episode", "domain"])
    items = [
        _create_trace_in_experiment("t1", experiment_id),
        _create_trace_in_experiment("t2", experiment_id),
    ]

    @scorer(session_level="trip")
    def trip_scorer(session):
        return len(session)

    @scorer(session_level="domain")
    def domain_scorer(session):
        return len(session)

    real_get_experiment = MlflowClient.get_experiment
    fetched = []

    def spy(self, experiment_id):
        fetched.append(experiment_id)
        return real_get_experiment(self, experiment_id)

    with mock.patch.object(MlflowClient, "get_experiment", spy):
        buckets = group_sessions_by_scorer_level(items, [trip_scorer, domain_scorer], experiment_id)

    assert len(buckets) == 2
    assert fetched == [experiment_id]


def test_resolve_session_level_index_ignores_sessionless_items_for_fallback():
    # Traces in a tagged experiment, run in an untagged one, plus a sessionless
    # row: the extra row must not drag the untagged run experiment into resolution.
    tagged_exp = mlflow.set_experiment("hierarchy-sessionless-a").experiment_id
    _set_hierarchy(tagged_exp, ["trip", "episode"])
    untagged_exp = mlflow.set_experiment("hierarchy-sessionless-b").experiment_id
    sessionless_item = Mock(spec=EvalItem)
    sessionless_item.trace = None
    sessionless_item.source = None
    items = [
        _create_trace_in_experiment("t1", tagged_exp),
        sessionless_item,
    ]

    index = resolve_session_level_index("episode", items, untagged_exp, ["my_scorer"])
    assert index == 1


def test_resolve_session_level_index_no_experiment_determined():
    # A session to group but no experiment to read the hierarchy from (no trace
    # experiment and no run experiment) is an error.
    item = Mock(spec=EvalItem)
    item.trace = None
    item.source = Mock(source_data={"session_id": "session-1"})

    with pytest.raises(MlflowException, match="no experiment could be determined"):
        resolve_session_level_index("episode", [item], None, ["my_scorer"])


def test_resolve_session_level_index_empty_items_returns_zero():
    # Nothing to group; any index is equivalent and no experiment is fetched.
    assert resolve_session_level_index("episode", [], None, ["my_scorer"]) == 0
