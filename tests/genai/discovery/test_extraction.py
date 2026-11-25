from unittest import mock

import pytest

import mlflow
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.genai.discovery.entities import _ConversationAnalysis
from mlflow.genai.discovery.extraction import (
    collect_session_rationales,
    extract_assessment_rationale,
    extract_execution_path,
    extract_execution_paths_for_session,
    extract_failing_traces,
    extract_failure_labels,
    extract_span_errors,
)
from mlflow.genai.discovery.pipeline import _build_analyses
from mlflow.tracing.constant import AssessmentMetadataKey, TraceMetadataKey

# ---- extract_span_errors ----


def test_extract_span_errors_with_error_span(make_trace):
    trace = make_trace(error_span=True)
    result = extract_span_errors(trace)
    assert result
    assert "Connection failed" in result


def test_extract_span_errors_no_errors(make_trace):
    trace = make_trace()
    result = extract_span_errors(trace)
    assert result == ""


def test_extract_span_errors_truncation(make_trace):
    trace = make_trace(error_span=True)
    result = extract_span_errors(trace, max_length=10)
    assert len(result) <= 10


# ---- extract_execution_path ----


@pytest.mark.parametrize(
    ("error_span", "expected_substring"),
    [
        (False, "(no routing)"),
        (True, "tool_call"),
    ],
)
def test_extract_execution_path(make_trace, error_span, expected_substring):
    trace = make_trace(error_span=error_span)
    result = extract_execution_path(trace)
    assert expected_substring in result


# ---- extract_execution_paths_for_session ----


def test_extract_execution_paths_for_session_deduplicates(make_trace):
    traces = [make_trace(), make_trace()]
    result = extract_execution_paths_for_session(traces)
    assert ";" not in result


def test_extract_execution_paths_for_session_combines_paths(make_trace):
    traces = [make_trace(), make_trace(error_span=True)]
    result = extract_execution_paths_for_session(traces)
    assert ";" in result


# ---- extract_failure_labels ----


def _make_llm_response(content: str):
    response = mock.MagicMock()
    response.choices = [mock.MagicMock()]
    response.choices[0].message.content = content
    response.usage = None
    return response


def test_extract_failure_labels_empty_analyses():
    labels, label_to_analysis = extract_failure_labels([], "openai:/gpt-5-mini")
    assert labels == []
    assert label_to_analysis == []


def test_extract_failure_labels_single_analysis():
    analyses = [
        _ConversationAnalysis(
            full_rationale="The assistant failed to provide weather data",
            affected_trace_ids=["t1"],
            execution_path="weather_tool > api_call",
        ),
    ]

    with mock.patch(
        "mlflow.genai.discovery.extraction._call_llm",
        return_value=_make_llm_response("didn't provide weather data despite explicit request"),
    ) as mock_llm:
        labels, label_to_analysis = extract_failure_labels(analyses, "openai:/gpt-5-mini")

    mock_llm.assert_called_once()
    assert len(labels) == 1
    assert "[weather_tool > api_call]" in labels[0]
    assert "weather data" in labels[0]
    assert label_to_analysis == [0]


def test_extract_failure_labels_multi_label():
    analyses = [
        _ConversationAnalysis(
            full_rationale="Two problems: auth failed and response was empty",
            affected_trace_ids=["t1"],
            execution_path="api_tool",
        ),
    ]

    with mock.patch(
        "mlflow.genai.discovery.extraction._call_llm",
        return_value=_make_llm_response("auth token expired\nempty response body"),
    ) as mock_llm:
        labels, label_to_analysis = extract_failure_labels(analyses, "openai:/gpt-5-mini")

    mock_llm.assert_called_once()
    assert len(labels) == 2
    assert label_to_analysis == [0, 0]
    assert "[api_tool] auth token expired" in labels
    assert "[api_tool] empty response body" in labels


def test_extract_failure_labels_keeps_leading_hyphen():
    analyses = [
        _ConversationAnalysis(
            full_rationale="Returned a sentinel instead of an error",
            affected_trace_ids=["t1"],
            execution_path="api_tool",
        ),
    ]

    with mock.patch(
        "mlflow.genai.discovery.extraction._call_llm",
        return_value=_make_llm_response("- -1 returned instead of error\n* -inf score emitted"),
    ):
        labels, _ = extract_failure_labels(analyses, "openai:/gpt-5-mini")

    assert "[api_tool] -1 returned instead of error" in labels
    assert "[api_tool] -inf score emitted" in labels


# ---- extract_failing_traces ----

_SOURCE = AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="test")


def _add_feedback(
    trace, name, value=None, rationale="", run_id=None, source=_SOURCE, error=None, session=None
):
    metadata = {}
    if run_id:
        metadata[AssessmentMetadataKey.SOURCE_RUN_ID] = run_id
    if session:
        metadata[TraceMetadataKey.TRACE_SESSION] = session
    mlflow.log_feedback(
        trace_id=trace.info.trace_id,
        name=name,
        value=value,
        rationale=rationale,
        source=source,
        error=error,
        metadata=metadata or None,
    )


def _refetch(traces):
    return [mlflow.get_trace(t.info.trace_id) for t in traces]


def test_extract_failing_traces(make_trace):
    traces = [make_trace() for _ in range(3)]
    _add_feedback(traces[0], "satisfaction", True, "good")
    _add_feedback(traces[1], "satisfaction", False, "bad response")
    _add_feedback(traces[2], "satisfaction", False, "incomplete")

    result = extract_failing_traces(_refetch(traces), "satisfaction")

    assert len(result.failing_traces) == 2
    assert result.failing_traces[0].info.trace_id == traces[1].info.trace_id
    assert result.failing_traces[1].info.trace_id == traces[2].info.trace_id
    assert result.rationale_map[traces[1].info.trace_id] == "bad response"
    assert result.rationale_map[traces[2].info.trace_id] == "incomplete"


def test_extract_failing_traces_with_list_of_scorer_names(make_trace):
    traces = [make_trace() for _ in range(3)]
    _add_feedback(traces[0], "satisfaction", True, "good")
    _add_feedback(traces[0], "quality", True, "ok")
    _add_feedback(traces[1], "satisfaction", False, "bad response")
    _add_feedback(traces[1], "quality", True, "ok")
    _add_feedback(traces[2], "satisfaction", True, "good")
    _add_feedback(traces[2], "quality", False, "poor quality")

    result = extract_failing_traces(_refetch(traces), ["satisfaction", "quality"])

    assert len(result.failing_traces) == 2
    assert result.failing_traces[0].info.trace_id == traces[1].info.trace_id
    assert result.failing_traces[1].info.trace_id == traces[2].info.trace_id
    assert result.rationale_map[traces[1].info.trace_id] == "bad response"
    assert result.rationale_map[traces[2].info.trace_id] == "poor quality"


def test_extract_failing_traces_multiple_scorers_fail_same_row(make_trace):
    traces = [make_trace() for _ in range(2)]
    _add_feedback(traces[0], "scorer_a", False, "reason a")
    _add_feedback(traces[0], "scorer_b", False, "reason b")
    _add_feedback(traces[1], "scorer_a", True, "ok")
    _add_feedback(traces[1], "scorer_b", True, "ok")

    result = extract_failing_traces(_refetch(traces), ["scorer_a", "scorer_b"])

    assert len(result.failing_traces) == 1
    assert result.failing_traces[0].info.trace_id == traces[0].info.trace_id
    assert "reason a" in result.rationale_map[traces[0].info.trace_id]
    assert "reason b" in result.rationale_map[traces[0].info.trace_id]


def test_extract_failing_traces_reads_only_the_runs_own_verdicts(make_trace):
    # Every earlier run's verdicts stay on a trace. Scoped to a run, a trace that run didn't judge
    # is unclassified rather than read on a stale verdict, and a trace it did judge is read on
    # its own verdict even when an earlier run's says otherwise.
    unjudged = make_trace()
    judged = make_trace()
    _add_feedback(unjudged, "satisfaction", False, "stale failure", run_id="run-old")
    _add_feedback(judged, "satisfaction", False, "stale failure", run_id="run-old")
    _add_feedback(judged, "satisfaction", True, "fine now", run_id="run-new")

    result = extract_failing_traces(_refetch([unjudged, judged]), "satisfaction", run_id="run-new")

    assert result.failing_traces == []
    assert result.unclassified_trace_ids == [unjudged.info.trace_id]


def test_extract_failing_traces_reads_the_runs_failure_over_a_later_one(make_trace):
    trace = make_trace()
    _add_feedback(trace, "satisfaction", False, "this run's failure", run_id="run-new")
    _add_feedback(trace, "satisfaction", True, "another run's pass", run_id="run-other")

    result = extract_failing_traces(_refetch([trace]), "satisfaction", run_id="run-new")

    assert [t.info.trace_id for t in result.failing_traces] == [trace.info.trace_id]
    assert result.rationale_map[trace.info.trace_id] == "this run's failure"


def test_extract_failing_traces_reads_the_runs_judge_error_over_an_older_verdict(make_trace):
    trace = make_trace()
    _add_feedback(trace, "satisfaction", True, "earlier pass", run_id="run-old")
    _add_feedback(
        trace,
        "satisfaction",
        run_id="run-new",
        error=AssessmentError(error_code="RATE_LIMIT_EXCEEDED", error_message="rate limited"),
    )

    result = extract_failing_traces(_refetch([trace]), "satisfaction", run_id="run-new")

    assert result.failing_traces == []
    assert result.scorer_error_trace_ids == [trace.info.trace_id]
    assert "rate limited" in result.scorer_errors[trace.info.trace_id]


_HUMAN = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="reviewer")


def test_extract_failing_traces_keeps_a_humans_verdict(make_trace):
    # A human's verdict carries no run: it still decides a trace this run didn't judge.
    trace = make_trace()
    _add_feedback(trace, "satisfaction", False, "a reviewer says it's wrong", source=_HUMAN)

    result = extract_failing_traces(_refetch([trace]), "satisfaction", run_id="run-new")

    assert [t.info.trace_id for t in result.failing_traces] == [trace.info.trace_id]
    assert result.rationale_map[trace.info.trace_id] == "a reviewer says it's wrong"


def test_extract_failing_traces_reads_the_runs_newer_verdict_over_a_humans(make_trace):
    trace = make_trace()
    _add_feedback(trace, "satisfaction", False, "an earlier reviewer's call", source=_HUMAN)
    _add_feedback(trace, "satisfaction", True, "judged fine now", run_id="run-new")

    result = extract_failing_traces(_refetch([trace]), "satisfaction", run_id="run-new")

    assert result.failing_traces == []
    assert result.unclassified_trace_ids == []


def test_extract_assessment_rationale_reads_only_the_runs_own_and_human_feedback(make_trace):
    # This run's verdict has no rationale, so an earlier run's must not stand in for it, while a
    # human's still does.
    trace = make_trace()
    _add_feedback(trace, "satisfaction", False, "an earlier run's reason", run_id="run-old")
    _add_feedback(trace, "satisfaction", False, "", run_id="run-new")

    assert (
        extract_assessment_rationale(
            trace=_refetch([trace])[0], scorer_name="satisfaction", run_id="run-new"
        )
        == ""
    )

    _add_feedback(trace, "satisfaction", False, "a reviewer's reason", source=_HUMAN)
    assert (
        extract_assessment_rationale(_refetch([trace])[0], "satisfaction", run_id="run-new")
        == "a reviewer's reason"
    )


def test_collect_session_rationales_reads_only_the_runs_own_feedback(make_trace):
    trace = make_trace()
    _add_feedback(trace, "satisfaction", False, "an earlier run's reason", run_id="run-old")
    (refetched,) = _refetch([trace])

    combined = collect_session_rationales(
        [refetched], {trace.info.trace_id: "this run's reason"}, "satisfaction", run_id="run-new"
    )

    assert "this run's reason" in combined
    assert "an earlier run's reason" not in combined


def test_collect_session_rationales_reads_only_the_runs_own_session_feedback(make_trace):
    first, second = (make_trace(session_id="s-1") for _ in range(2))
    _add_feedback(
        first,
        "satisfaction",
        False,
        "an earlier run's session reason",
        run_id="run-old",
        session="s-1",
    )
    session = _refetch([first, second])

    combined = collect_session_rationales(
        session[1:],
        {second.info.trace_id: "this run's reason"},
        "satisfaction",
        session_traces=session,
        run_id="run-new",
    )
    assert "[session feedback]" not in combined
    assert "an earlier run's session reason" not in combined

    _add_feedback(
        first, "satisfaction", False, "this run's session reason", run_id="run-new", session="s-1"
    )
    session = _refetch([first, second])
    combined = collect_session_rationales(
        session[1:], {}, "satisfaction", session_traces=session, run_id="run-new"
    )
    assert combined == "[session feedback] this run's session reason"


def test_build_analyses_reads_only_the_runs_own_feedback(make_trace):
    trace = make_trace()
    _add_feedback(trace, "satisfaction", False, "an earlier run's reason", run_id="run-old")
    (refetched,) = _refetch([trace])

    analyses, _ = _build_analyses(
        [refetched],
        {trace.info.trace_id: "this run's reason"},
        {},
        "satisfaction",
        run_id="run-new",
    )

    (analysis,) = analyses
    assert "this run's reason" in analysis.full_rationale
    assert "an earlier run's reason" not in analysis.full_rationale


def test_extract_failing_traces_without_a_run_reads_every_runs_verdicts(make_trace):
    trace = make_trace()
    _add_feedback(trace, "satisfaction", False, "earlier failure", run_id="run-old")

    result = extract_failing_traces(_refetch([trace]), "satisfaction")

    assert [t.info.trace_id for t in result.failing_traces] == [trace.info.trace_id]


def test_extract_failing_traces_empty_list():
    result = extract_failing_traces([], "satisfaction")
    assert result.failing_traces == []
    assert result.rationale_map == {}


def test_extract_failing_traces_no_matching_scorer(make_trace):
    traces = [make_trace()]
    _add_feedback(traces[0], "other_scorer", False, "bad")

    result = extract_failing_traces(_refetch(traces), "satisfaction")
    assert result.failing_traces == []


def test_extract_failing_traces_no_failures(make_trace):
    traces = [make_trace()]
    _add_feedback(traces[0], "satisfaction", True, "good")

    result = extract_failing_traces(_refetch(traces), "satisfaction")
    assert result.failing_traces == []
    assert result.rationale_map == {}


# ---- extract_assessment_rationale ----


@pytest.mark.parametrize(
    ("feedback_name", "query_name", "expected"),
    [
        ("scorer_a", "scorer_a", "test rationale"),
        ("scorer_a", "scorer_b", ""),
    ],
)
def test_extract_assessment_rationale(make_trace, feedback_name, query_name, expected):
    trace = make_trace()
    _add_feedback(trace, feedback_name, False, "test rationale")
    result = extract_assessment_rationale(_refetch([trace])[0], query_name)
    assert result == expected


# ---- collect_session_rationales ----


def test_collect_session_rationales_combines_sources(make_trace):
    trace = make_trace(error_span=True)
    _add_feedback(trace, "scorer_a", False, "human says bad")
    trace = _refetch([trace])[0]

    rationale_map = {trace.info.trace_id: "triage rationale"}
    result = collect_session_rationales([trace], rationale_map, "scorer_a")

    assert "triage rationale" in result
    assert "[human feedback] human says bad" in result
    assert "[span errors]" in result


def test_collect_session_rationales_deduplicates(make_trace):
    trace = make_trace()
    _add_feedback(trace, "scorer_a", False, "same text")
    trace = _refetch([trace])[0]

    rationale_map = {trace.info.trace_id: "same text"}
    result = collect_session_rationales([trace], rationale_map, "scorer_a")

    assert result.count("same text") == 1


def test_extract_failure_labels_attributes_llm_calls_to_issue_detection():
    from mlflow.gateway.constants import GatewayCaller

    analyses = [
        _ConversationAnalysis(
            full_rationale="The assistant failed to provide weather data",
            affected_trace_ids=["t1"],
            execution_path="weather_tool > api_call",
        ),
    ]

    with mock.patch(
        "mlflow.genai.discovery.extraction._call_llm",
        return_value=_make_llm_response("didn't provide weather data"),
    ) as mock_llm:
        extract_failure_labels(analyses, "gateway:/my-endpoint")

    assert mock_llm.call_args.kwargs["gateway_caller"] == GatewayCaller.ISSUE_DETECTION
