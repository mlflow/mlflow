import time
from unittest import mock

import pytest

from mlflow.entities.issue import Issue, IssueStatus
from mlflow.genai.discovery import utils
from mlflow.genai.discovery.entities import _ConversationAnalysis, _IdentifiedIssue
from mlflow.genai.discovery.utils import (
    _TokenCounter,
    build_summary,
    collect_example_trace_ids,
    format_annotation_prompt,
    format_trace_content,
    get_session_id,
    group_traces_by_session,
    log_discovery_artifacts,
)


def test_format_trace_content_includes_errors(make_trace):
    trace = make_trace(error_span=True)
    content = format_trace_content(trace)
    assert "Errors:" in content
    assert "Connection failed" in content


def test_format_trace_content_no_errors(make_trace):
    trace = make_trace()
    content = format_trace_content(trace)
    assert "Errors:" not in content


def test_collect_example_trace_ids_gathers_from_analyses():
    analyses = [
        _ConversationAnalysis(
            full_rationale="r1", affected_trace_ids=["t1", "t2"], execution_path="p1"
        ),
        _ConversationAnalysis(full_rationale="r2", affected_trace_ids=["t3"], execution_path="p2"),
    ]
    issue = _IdentifiedIssue(
        name="test",
        description="d",
        root_cause="rc",
        severity="high",
        example_indices=[0, 1],
        categories=[],
    )
    result = collect_example_trace_ids(issue, analyses)
    assert result == ["t1", "t2", "t3"]


def test_collect_example_trace_ids_skips_out_of_bounds():
    analyses = [
        _ConversationAnalysis(full_rationale="r1", affected_trace_ids=["t1"], execution_path="p1"),
    ]
    issue = _IdentifiedIssue(
        name="test",
        description="d",
        root_cause="rc",
        severity="high",
        example_indices=[0, 5],
        categories=[],
    )
    result = collect_example_trace_ids(issue, analyses)
    assert result == ["t1"]


def test_collect_example_trace_ids_caps_at_max(monkeypatch):
    monkeypatch.setattr(utils, "MAX_EXAMPLE_TRACE_IDS", 2)
    analyses = [
        _ConversationAnalysis(
            full_rationale="r1", affected_trace_ids=["t1", "t2", "t3"], execution_path="p1"
        ),
    ]
    issue = _IdentifiedIssue(
        name="test",
        description="d",
        root_cause="rc",
        severity="high",
        example_indices=[0],
        categories=[],
    )
    result = collect_example_trace_ids(issue, analyses)
    assert len(result) == 2


# ---- get_session_id ----


@pytest.mark.parametrize(
    ("session_id", "expected"),
    [
        ("sess-123", "sess-123"),
        (None, None),
    ],
)
def test_get_session_id(make_trace, session_id, expected):
    trace = make_trace(session_id=session_id)
    assert get_session_id(trace) == expected


# ---- log_discovery_artifacts ----


def test_log_discovery_artifacts_logs_text():
    mock_client = mock.MagicMock()
    with mock.patch("mlflow.MlflowClient", return_value=mock_client):
        log_discovery_artifacts("run-1", {"file.txt": "content"})
    mock_client.log_text.assert_called_once_with("run-1", "content", "file.txt")


def test_log_discovery_artifacts_skips_empty_run_id():
    with mock.patch("mlflow.MlflowClient") as mock_client_cls:
        log_discovery_artifacts("", {"file.txt": "content"})
    mock_client_cls.assert_not_called()


def test_log_discovery_artifacts_handles_exception():
    mock_client = mock.MagicMock()
    mock_client.log_text.side_effect = Exception("fail")
    with mock.patch("mlflow.MlflowClient", return_value=mock_client):
        log_discovery_artifacts("run-1", {"file.txt": "content"})
    mock_client.log_text.assert_called_once()


# ---- format_annotation_prompt ----


def test_format_annotation_prompt():
    issue = Issue(
        issue_id="i1",
        experiment_id="exp1",
        name="Test Issue",
        description="Test description",
        status=IssueStatus.PENDING,
        created_timestamp=0,
        last_updated_timestamp=0,
        root_causes=["cause1", "cause2"],
    )
    result = format_annotation_prompt(issue, "trace content here", "triage rationale here")
    assert "=== ISSUE ===" in result
    assert "Test Issue" in result
    assert "Test description" in result
    assert "cause1; cause2" in result
    assert "=== TRACE ===" in result
    assert "trace content here" in result
    assert "=== TRIAGE JUDGE RATIONALE ===" in result
    assert "triage rationale here" in result


# ---- build_summary ----


def test_build_summary_no_issues():
    result = build_summary([], 100)
    assert result == "Analyzed 100 traces. No issues found."
    assert not result.startswith("##")


def test_build_summary_with_issues():
    issues = [
        Issue(
            issue_id="i1",
            experiment_id="exp1",
            name="Slow Response",
            description="API responses are slow",
            status=IssueStatus.PENDING,
            created_timestamp=0,
            last_updated_timestamp=0,
            severity="high",
            root_causes=["Database query optimization needed"],
        ),
        Issue(
            issue_id="i2",
            experiment_id="exp1",
            name="Connection Errors",
            description="Frequent connection timeouts",
            status=IssueStatus.PENDING,
            created_timestamp=0,
            last_updated_timestamp=0,
            severity="medium",
            root_causes=["Network instability", "Upstream service issues"],
        ),
    ]
    result = build_summary(issues, 50)
    assert "Analyzed **50** traces" in result
    assert "Found **2** issues" in result
    assert not result.startswith("##")
    assert "### 1. Slow Response" in result
    assert "### 2. Connection Errors" in result
    assert "Database query optimization needed" in result
    assert "Network instability; Upstream service issues" in result


def test_token_counter_tracks_usage():
    counter = _TokenCounter()
    assert counter.input_tokens == 0
    assert counter.output_tokens == 0
    assert counter.cost_usd == 0.0

    mock_response = mock.MagicMock()
    mock_response.usage = mock.MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response._hidden_params = {"response_cost": 0.005}

    counter.track(mock_response)

    assert counter.input_tokens == 100
    assert counter.output_tokens == 50
    assert counter.cost_usd == 0.005


def test_group_traces_by_session_groups_by_session_id(make_trace):
    traces = [
        make_trace(session_id="session-1"),
        make_trace(session_id="session-1"),
        make_trace(session_id="session-2"),
    ]

    result = group_traces_by_session(traces)

    assert len(result) == 2
    assert "session-1" in result
    assert "session-2" in result
    assert len(result["session-1"]) == 2
    assert len(result["session-2"]) == 1


def test_group_traces_by_session_creates_standalone_groups_for_no_session(make_trace):
    trace1 = make_trace(session_id=None)
    trace2 = make_trace(session_id=None)
    trace3 = make_trace(session_id="session-A")

    traces = [trace1, trace2, trace3]
    result = group_traces_by_session(traces)

    assert len(result) == 3
    assert trace1.info.trace_id in result
    assert trace2.info.trace_id in result
    assert "session-A" in result
    assert len(result[trace1.info.trace_id]) == 1
    assert len(result[trace2.info.trace_id]) == 1
    assert len(result["session-A"]) == 1


def test_group_traces_by_session_sorts_by_timestamp(make_trace):
    trace1 = make_trace(session_id="session-1")
    time.sleep(0.01)
    trace2 = make_trace(session_id="session-1")
    time.sleep(0.01)
    trace3 = make_trace(session_id="session-1")

    traces = [trace3, trace1, trace2]

    result = group_traces_by_session(traces)

    session_traces = result["session-1"]
    assert session_traces[0].info.trace_id == trace1.info.trace_id
    assert session_traces[1].info.trace_id == trace2.info.trace_id
    assert session_traces[2].info.trace_id == trace3.info.trace_id
