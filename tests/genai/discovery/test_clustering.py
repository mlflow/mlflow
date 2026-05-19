import json
from unittest.mock import MagicMock, patch

from mlflow.entities.issue import IssueSeverity, IssueStatus
from mlflow.genai.discovery.clustering import (
    cluster_by_llm,
    summarize_cluster,
)
from mlflow.genai.discovery.constants import build_cluster_summary_prompt
from mlflow.genai.discovery.entities import (
    Issue,
    _ConversationAnalysis,
)
from mlflow.genai.discovery.utils import build_summary

# ---- cluster_by_llm ----


def test_cluster_by_llm_groups_similar():
    labels = [
        "[llm_pipeline] hallucinated facts",
        "[llm_pipeline] hallucinated different facts",
        "[database] query timeout",
    ]

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "groups": [
                        {"name": "Issue: Hallucination", "indices": [0, 1]},
                        {"name": "Issue: Query timeout", "indices": [2]},
                    ]
                })
            )
        )
    ]

    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        groups = cluster_by_llm(labels, max_issues=5, model="openai:/gpt-5")

    mock_completion.assert_called_once()
    assert len(groups) == 2
    flat = [idx for g in groups for idx in g]
    assert sorted(flat) == [0, 1, 2]


def test_cluster_by_llm_respects_max_issues():
    labels = [f"[domain_{i}] unique issue {i}" for i in range(5)]

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "groups": [
                        {"name": "Issue: Group A", "indices": [0, 1, 2]},
                        {"name": "Issue: Group B", "indices": [3, 4]},
                    ]
                })
            )
        )
    ]

    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        groups = cluster_by_llm(labels, max_issues=2, model="openai:/gpt-5")

    mock_completion.assert_called_once()
    assert len(groups) <= 2


# ---- summarize_cluster ----


def test_summarize_cluster():
    analyses = [
        _ConversationAnalysis(
            full_rationale="response generation via LLM",
            affected_trace_ids=["t-1"],
        ),
        _ConversationAnalysis(
            full_rationale="response generation via LLM",
            affected_trace_ids=["t-2"],
        ),
    ]

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "name": "hallucination",
                    "description": "LLM generates incorrect facts",
                    "root_cause": "Model confabulation",
                    "example_indices": [],
                    "severity": "high",
                    "categories": [],
                })
            )
        )
    ]

    with patch(
        "mlflow.genai.discovery.clustering._call_llm", return_value=mock_response
    ) as mock_call:
        result = summarize_cluster([0, 1], analyses, "openai:/gpt-5", categories=[])

    mock_call.assert_called_once()
    assert result.name == "hallucination"
    assert result.example_indices == [0, 1]


# ---- build_summary ----


def test_build_summary_no_issues():
    summary = build_summary([], 50)
    assert "50 traces" in summary
    assert "No issues found" in summary


def test_build_summary_with_issues():
    issues = [
        Issue(
            issue_id="test-id",
            experiment_id="0",
            name="tool_failure",
            description="Tool calls fail intermittently",
            status=IssueStatus.PENDING,
            created_timestamp=0,
            last_updated_timestamp=0,
            severity=IssueSeverity.HIGH,
            root_causes=["API timeout"],
        ),
    ]
    summary = build_summary(issues, 100)
    assert "tool_failure" in summary
    assert "API timeout" in summary


def test_summarize_cluster_filters_invalid_categories():
    analyses = [
        _ConversationAnalysis(
            full_rationale="[hallucination] agent made up facts",
            affected_trace_ids=["t-1"],
        ),
        _ConversationAnalysis(
            full_rationale="[tool_error] tool call failed",
            affected_trace_ids=["t-2"],
        ),
    ]

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps({
                    "name": "Issue: Multiple problems",
                    "description": "Various issues detected",
                    "root_cause": "Multiple root causes",
                    "example_indices": [],
                    "severity": "high",
                    "categories": ["hallucination", "invalid_cat", "tool_error", "another_invalid"],
                })
            )
        )
    ]

    valid_categories = ["hallucination", "tool_error", "latency"]

    with patch(
        "mlflow.genai.discovery.clustering._call_llm", return_value=mock_response
    ) as mock_call:
        result = summarize_cluster([0, 1], analyses, "openai:/gpt-5", categories=valid_categories)

    mock_call.assert_called_once()
    assert set(result.categories) == {"hallucination", "tool_error"}
    assert "invalid_cat" not in result.categories
    assert "another_invalid" not in result.categories


def test_build_cluster_summary_prompt_with_categories():
    categories = ["hallucination", "tool_error", "latency"]
    prompt = build_cluster_summary_prompt(categories=categories)

    assert "hallucination" in prompt
    assert "tool_error" in prompt
    assert "latency" in prompt
    assert "Assign one or more categories from" in prompt
