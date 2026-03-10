from mlflow.genai.discovery.entities import _ConversationAnalysis, _IdentifiedIssue
from mlflow.genai.discovery.utils import (
    collect_example_trace_ids,
    format_trace_content,
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
    from mlflow.genai.discovery import utils

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
