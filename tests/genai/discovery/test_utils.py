from unittest import mock

import pytest

from mlflow.entities.issue import Issue, IssueStatus
from mlflow.genai.discovery import utils
from mlflow.genai.discovery.entities import _ConversationAnalysis, _IdentifiedIssue
from mlflow.genai.discovery.utils import (
    collect_example_trace_ids,
    format_annotation_prompt,
    format_trace_content,
    get_session_id,
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
        name="test", description="d", root_cause="rc", severity="high", example_indices=[0, 1]
    )
    result = collect_example_trace_ids(issue, analyses)
    assert result == ["t1", "t2", "t3"]


def test_collect_example_trace_ids_skips_out_of_bounds():
    analyses = [
        _ConversationAnalysis(full_rationale="r1", affected_trace_ids=["t1"], execution_path="p1"),
    ]
    issue = _IdentifiedIssue(
        name="test", description="d", root_cause="rc", severity="high", example_indices=[0, 5]
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
        name="test", description="d", root_cause="rc", severity="high", example_indices=[0]
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
