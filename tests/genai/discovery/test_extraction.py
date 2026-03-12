from unittest import mock

import pytest

import mlflow
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


# ---- extract_failing_traces ----

_SOURCE = AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="test")


def _add_feedback(trace, name, value, rationale=""):
    mlflow.log_feedback(
        trace_id=trace.info.trace_id,
        name=name,
        value=value,
        rationale=rationale,
        source=_SOURCE,
    )


def _refetch(traces):
    return [mlflow.get_trace(t.info.trace_id) for t in traces]


def test_extract_failing_traces(make_trace):
    traces = [make_trace() for _ in range(3)]
    _add_feedback(traces[0], "satisfaction", True, "good")
    _add_feedback(traces[1], "satisfaction", False, "bad response")
    _add_feedback(traces[2], "satisfaction", False, "incomplete")

    failing, rationales = extract_failing_traces(_refetch(traces), "satisfaction")

    assert len(failing) == 2
    assert failing[0].info.trace_id == traces[1].info.trace_id
    assert failing[1].info.trace_id == traces[2].info.trace_id
    assert rationales[traces[1].info.trace_id] == "bad response"
    assert rationales[traces[2].info.trace_id] == "incomplete"


def test_extract_failing_traces_with_list_of_scorer_names(make_trace):
    traces = [make_trace() for _ in range(3)]
    _add_feedback(traces[0], "satisfaction", True, "good")
    _add_feedback(traces[0], "quality", True, "ok")
    _add_feedback(traces[1], "satisfaction", False, "bad response")
    _add_feedback(traces[1], "quality", True, "ok")
    _add_feedback(traces[2], "satisfaction", True, "good")
    _add_feedback(traces[2], "quality", False, "poor quality")

    failing, rationales = extract_failing_traces(_refetch(traces), ["satisfaction", "quality"])

    assert len(failing) == 2
    assert failing[0].info.trace_id == traces[1].info.trace_id
    assert failing[1].info.trace_id == traces[2].info.trace_id
    assert rationales[traces[1].info.trace_id] == "bad response"
    assert rationales[traces[2].info.trace_id] == "poor quality"


def test_extract_failing_traces_multiple_scorers_fail_same_row(make_trace):
    traces = [make_trace() for _ in range(2)]
    _add_feedback(traces[0], "scorer_a", False, "reason a")
    _add_feedback(traces[0], "scorer_b", False, "reason b")
    _add_feedback(traces[1], "scorer_a", True, "ok")
    _add_feedback(traces[1], "scorer_b", True, "ok")

    failing, rationales = extract_failing_traces(_refetch(traces), ["scorer_a", "scorer_b"])

    assert len(failing) == 1
    assert failing[0].info.trace_id == traces[0].info.trace_id
    assert "reason a" in rationales[traces[0].info.trace_id]
    assert "reason b" in rationales[traces[0].info.trace_id]


def test_extract_failing_traces_empty_list():
    failing, rationales = extract_failing_traces([], "satisfaction")
    assert failing == []
    assert rationales == {}


def test_extract_failing_traces_no_matching_scorer(make_trace):
    traces = [make_trace()]
    _add_feedback(traces[0], "other_scorer", False, "bad")

    failing, rationales = extract_failing_traces(_refetch(traces), "satisfaction")
    assert failing == []


def test_extract_failing_traces_no_failures(make_trace):
    traces = [make_trace()]
    _add_feedback(traces[0], "satisfaction", True, "good")

    failing, rationales = extract_failing_traces(_refetch(traces), "satisfaction")
    assert failing == []
    assert rationales == {}


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
