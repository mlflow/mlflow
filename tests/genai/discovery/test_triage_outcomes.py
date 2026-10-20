import mlflow
from mlflow.entities.assessment import AssessmentError
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.genai.discovery.extraction import collect_session_rationales, extract_failing_traces

_SOURCE = AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="test")


def _add_feedback(trace, name, value, rationale="", error=None, metadata=None):
    mlflow.log_feedback(
        trace_id=trace.info.trace_id,
        name=name,
        value=value,
        rationale=rationale,
        source=_SOURCE,
        error=error,
        metadata=metadata,
    )


def _refetch(traces):
    return [mlflow.get_trace(trace.info.trace_id) for trace in traces]


def test_a_scorer_error_is_a_scorer_problem_not_a_passing_trace(make_trace):
    trace = make_trace()
    _add_feedback(
        trace,
        "satisfaction",
        None,
        error=AssessmentError(error_code="RATE_LIMIT", error_message="429"),
    )

    result = extract_failing_traces(_refetch([trace]), "satisfaction")

    assert result.failing_traces == []
    assert result.scorer_error_trace_ids == [trace.info.trace_id]
    assert result.unclassified_trace_ids == []
    assert "429" in result.scorer_errors[trace.info.trace_id]


def test_a_null_verdict_leaves_the_trace_unclassified(make_trace):
    trace = make_trace()
    _add_feedback(trace, "satisfaction", None, rationale="judge returned nothing")

    result = extract_failing_traces(_refetch([trace]), "satisfaction")

    assert result.failing_traces == []
    assert result.unclassified_trace_ids == [trace.info.trace_id]
    assert result.scorer_error_trace_ids == []


def test_a_trace_judged_by_one_scorer_is_classified_even_if_another_errored(make_trace):
    trace = make_trace()
    _add_feedback(
        trace, "satisfaction", None, error=AssessmentError(error_code="BOOM", error_message="down")
    )
    _add_feedback(trace, "quality", False, rationale="wrong answer")

    result = extract_failing_traces(_refetch([trace]), ["satisfaction", "quality"])

    assert [t.info.trace_id for t in result.failing_traces] == [trace.info.trace_id]
    assert result.scorer_error_trace_ids == []
    assert result.rationale_map[trace.info.trace_id] == "wrong answer"


def test_a_passing_trace_is_neither_errored_nor_unclassified(make_trace):
    trace = make_trace()
    _add_feedback(trace, "satisfaction", True, rationale="goals met")

    result = extract_failing_traces(_refetch([trace]), "satisfaction")

    assert result.failing_traces == []
    assert result.scorer_error_trace_ids == []
    assert result.unclassified_trace_ids == []


def test_an_all_judge_error_run_is_classified_as_errored_not_passing(make_trace):
    traces = [make_trace() for _ in range(3)]
    for trace in traces:
        _add_feedback(
            trace,
            "satisfaction",
            None,
            error=AssessmentError(error_code="RATE_LIMIT", error_message="429"),
        )

    result = extract_failing_traces(_refetch(traces), "satisfaction")

    # Not one of these traces passed; every one of them went unjudged.
    assert result.failing_traces == []
    assert len(result.scorer_error_trace_ids) == 3
    assert result.unclassified_trace_ids == []


def test_session_scoped_feedback_is_read_from_the_session_not_the_failing_turn(make_trace):
    first = make_trace(session_id="sess-1")
    second = make_trace(session_id="sess-1")
    # Session-level feedback lands on the first turn, which itself passed triage.
    _add_feedback(
        first,
        "satisfaction",
        False,
        rationale="the assistant never resolved the booking",
        metadata={"mlflow.trace.session": "sess-1"},
    )
    session = _refetch([first, second])

    combined = collect_session_rationales(
        [session[1]],
        {second.info.trace_id: "turn two returned nothing"},
        "satisfaction",
        session_traces=session,
    )

    assert "[session feedback] the assistant never resolved the booking" in combined
    assert "turn two returned nothing" in combined


def test_per_turn_feedback_is_not_relabelled_as_session_feedback(make_trace):
    trace = make_trace(session_id="sess-2")
    _add_feedback(trace, "satisfaction", False, rationale="this turn was wrong")
    session = _refetch([trace])

    combined = collect_session_rationales(session, {}, "satisfaction", session_traces=session)

    assert "[human feedback] this turn was wrong" in combined
    assert "[session feedback]" not in combined


def test_an_errored_assessment_rationale_is_not_used_as_evidence(make_trace):
    trace = make_trace()
    _add_feedback(
        trace,
        "satisfaction",
        None,
        rationale="judge crashed while scoring",
        error=AssessmentError(error_code="BOOM", error_message="down"),
    )
    session = _refetch([trace])

    combined = collect_session_rationales(session, {}, "satisfaction", session_traces=session)

    assert "judge crashed" not in combined
