import json
from unittest.mock import MagicMock, patch

from mlflow.genai.discovery.clustering import (
    build_summary,
    cluster_by_llm,
    summarize_cluster,
)
from mlflow.genai.discovery.entities import (
    Issue,
    _ConversationAnalysis,
)

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
                content=json.dumps(
                    {
                        "groups": [
                            {"name": "Issue: Hallucination", "indices": [0, 1]},
                            {"name": "Issue: Query timeout", "indices": [2]},
                        ]
                    }
                )
            )
        )
    ]

    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        groups = cluster_by_llm(labels, max_issues=5)

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
                content=json.dumps(
                    {
                        "groups": [
                            {"name": "Issue: Group A", "indices": [0, 1, 2]},
                            {"name": "Issue: Group B", "indices": [3, 4]},
                        ]
                    }
                )
            )
        )
    ]

    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        groups = cluster_by_llm(labels, max_issues=2)

    mock_completion.assert_called_once()
    assert len(groups) <= 2


# ---- summarize_cluster ----


def test_summarize_cluster():
    analyses = [
        _ConversationAnalysis(
            surface="response generation via LLM",
            root_cause="Model hallucinated.",
            affected_trace_ids=["t-1"],
        ),
        _ConversationAnalysis(
            surface="response generation via LLM",
            root_cause="Model made up facts.",
            affected_trace_ids=["t-2"],
        ),
    ]

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "name": "hallucination",
                        "description": "LLM generates incorrect facts",
                        "root_cause": "Model confabulation",
                        "example_indices": [],
                        "confidence": "definitely_yes",
                    }
                )
            )
        )
    ]

    with patch("litellm.completion", return_value=mock_response) as mock_completion:
        result = summarize_cluster([0, 1], analyses, "openai:/gpt-5")

    mock_completion.assert_called_once()
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
            run_id="run-1",
            name="tool_failure",
            description="Tool calls fail intermittently",
            root_cause="API timeout",
            example_trace_ids=["t-0"],
            frequency=0.3,
            confidence="definitely_yes",
        ),
    ]
    summary = build_summary(issues, 100)
    assert "tool_failure" in summary
    assert "30%" in summary
    assert "API timeout" in summary
