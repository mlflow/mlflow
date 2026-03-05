import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.genai.discovery.entities import (
    Issue,
    _ConversationAnalysis,
)
from mlflow.genai.discovery.utils import (
    build_summary,
    cluster_analyses,
    extract_failing_traces,
    extract_span_errors,
    group_traces_by_session,
    sample_traces,
    summarize_cluster,
    verify_scorer,
)
from mlflow.genai.evaluation.entities import EvaluationResult

# ---- cluster_analyses ----


def test_cluster_analyses_single_analysis():
    analyses = [
        _ConversationAnalysis(
            surface="response generation via LLM pipeline",
            root_cause="Model produced incorrect output.",
            affected_trace_ids=["t-1"],
        )
    ]
    result = cluster_analyses(analyses, max_issues=5, labels=["[routing] hallucination"])

    assert result == [[0]]


def test_cluster_analyses_groups_similar():
    analyses = [
        _ConversationAnalysis(
            surface="response generation via LLM pipeline",
            root_cause="Model hallucinated facts.",
            affected_trace_ids=["t-1"],
        ),
        _ConversationAnalysis(
            surface="response generation via LLM pipeline",
            root_cause="Model hallucinated different facts.",
            affected_trace_ids=["t-2"],
        ),
        _ConversationAnalysis(
            surface="database query execution timeout",
            root_cause="Query took too long.",
            affected_trace_ids=["t-3"],
        ),
    ]

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
        groups = cluster_analyses(analyses, max_issues=5, labels=labels)

    mock_completion.assert_called_once()
    assert len(groups) == 2
    flat = [idx for g in groups for idx in g]
    assert sorted(flat) == [0, 1, 2]


def test_cluster_analyses_respects_max_issues():
    labels = [f"[domain_{i}] unique issue {i}" for i in range(5)]
    analyses = [
        _ConversationAnalysis(
            surface=f"unique issue number {i}",
            root_cause=f"Unique root cause {i}.",
            affected_trace_ids=[f"t-{i}"],
        )
        for i in range(5)
    ]

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
        groups = cluster_analyses(analyses, max_issues=2, labels=labels)

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


# ---- extract_failing_traces ----


def test_extract_failing_traces(make_trace):
    traces = [make_trace() for _ in range(3)]
    df = pd.DataFrame(
        {
            "satisfaction/value": [True, False, False],
            "satisfaction/rationale": ["good", "bad response", "incomplete"],
            "trace": traces,
        }
    )
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=df)

    failing, rationales = extract_failing_traces(eval_result, "satisfaction")

    assert len(failing) == 2
    assert failing[0].info.trace_id == traces[1].info.trace_id
    assert failing[1].info.trace_id == traces[2].info.trace_id
    assert rationales[traces[1].info.trace_id] == "bad response"
    assert rationales[traces[2].info.trace_id] == "incomplete"


def test_extract_failing_traces_with_list_of_scorer_names(make_trace):
    traces = [make_trace() for _ in range(3)]
    df = pd.DataFrame(
        {
            "satisfaction/value": [True, False, True],
            "satisfaction/rationale": ["good", "bad response", "good"],
            "quality/value": [True, True, False],
            "quality/rationale": ["ok", "ok", "poor quality"],
            "trace": traces,
        }
    )
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=df)

    failing, rationales = extract_failing_traces(eval_result, ["satisfaction", "quality"])

    assert len(failing) == 2
    assert failing[0].info.trace_id == traces[1].info.trace_id
    assert failing[1].info.trace_id == traces[2].info.trace_id
    assert rationales[traces[1].info.trace_id] == "bad response"
    assert rationales[traces[2].info.trace_id] == "poor quality"


def test_extract_failing_traces_multiple_scorers_fail_same_row(make_trace):
    traces = [make_trace() for _ in range(2)]
    df = pd.DataFrame(
        {
            "scorer_a/value": [False, True],
            "scorer_a/rationale": ["reason a", "ok"],
            "scorer_b/value": [False, True],
            "scorer_b/rationale": ["reason b", "ok"],
            "trace": traces,
        }
    )
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=df)

    failing, rationales = extract_failing_traces(eval_result, ["scorer_a", "scorer_b"])

    assert len(failing) == 1
    assert failing[0].info.trace_id == traces[0].info.trace_id
    assert "reason a" in rationales[traces[0].info.trace_id]
    assert "reason b" in rationales[traces[0].info.trace_id]


def test_extract_failing_traces_none_result_df():
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=None)
    failing, rationales = extract_failing_traces(eval_result, "satisfaction")
    assert failing == []
    assert rationales == {}


def test_extract_failing_traces_missing_column():
    df = pd.DataFrame({"other/value": [True]})
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=df)
    failing, rationales = extract_failing_traces(eval_result, "satisfaction")
    assert failing == []


def test_extract_failing_traces_no_failures(make_trace):
    traces = [make_trace()]
    df = pd.DataFrame(
        {
            "satisfaction/value": [True],
            "satisfaction/rationale": ["good"],
            "trace": traces,
        }
    )
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=df)
    failing, rationales = extract_failing_traces(eval_result, "satisfaction")
    assert failing == []
    assert rationales == {}


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


# ---- sample_traces ----


def test_sample_traces_no_sessions(make_trace):
    traces = [make_trace() for _ in range(20)]
    search_kwargs = {"filter_string": None, "return_type": "list", "locations": ["exp-1"]}

    with patch(
        "mlflow.genai.discovery.utils.mlflow.search_traces", return_value=traces
    ) as mock_search:
        result = sample_traces(5, search_kwargs)

    mock_search.assert_called_once()
    assert mock_search.call_args[1]["max_results"] == 25
    assert len(result) == 5
    assert all(t in traces for t in result)


def test_sample_traces_with_sessions(make_trace):
    s1_traces = [make_trace(session_id="s1") for _ in range(3)]
    s2_traces = [make_trace(session_id="s2") for _ in range(2)]
    s3_traces = [make_trace(session_id="s3") for _ in range(4)]
    all_traces = s1_traces + s2_traces + s3_traces
    search_kwargs = {"filter_string": None, "return_type": "list", "locations": ["exp-1"]}

    with patch(
        "mlflow.genai.discovery.utils.mlflow.search_traces", return_value=all_traces
    ) as mock_search:
        result = sample_traces(2, search_kwargs)

    mock_search.assert_called_once()
    session_ids = {(t.info.trace_metadata or {}).get("mlflow.trace.session") for t in result}
    assert len(session_ids) == 2


def test_sample_traces_empty_pool():
    search_kwargs = {"filter_string": None, "return_type": "list", "locations": ["exp-1"]}

    with patch("mlflow.genai.discovery.utils.mlflow.search_traces", return_value=[]) as mock_search:
        result = sample_traces(10, search_kwargs)

    mock_search.assert_called_once()
    assert result == []


def test_sample_traces_fewer_than_requested(make_trace):
    traces = [make_trace() for _ in range(3)]
    search_kwargs = {"filter_string": None, "return_type": "list", "locations": ["exp-1"]}

    with patch(
        "mlflow.genai.discovery.utils.mlflow.search_traces", return_value=traces
    ) as mock_search:
        result = sample_traces(10, search_kwargs)

    mock_search.assert_called_once()
    assert len(result) == 3


# ---- group_traces_by_session ----


def test_group_traces_by_session_with_sessions(make_trace):
    t1 = make_trace(session_id="s1")
    t2 = make_trace(session_id="s1")
    t3 = make_trace(session_id="s2")

    groups = group_traces_by_session([t1, t2, t3])

    assert len(groups) == 2
    assert len(groups["s1"]) == 2
    assert len(groups["s2"]) == 1


def test_group_traces_by_session_no_sessions(make_trace):
    t1 = make_trace()
    t2 = make_trace()

    groups = group_traces_by_session([t1, t2])

    assert len(groups) == 2
    assert t1.info.trace_id in groups
    assert t2.info.trace_id in groups


def test_group_traces_by_session_mixed(make_trace):
    t1 = make_trace(session_id="s1")
    t2 = make_trace()

    groups = group_traces_by_session([t1, t2])

    assert len(groups) == 2
    assert len(groups["s1"]) == 1
    assert t2.info.trace_id in groups


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


# ---- verify_scorer ----


def test_test_scorer_happy_path(make_trace):
    trace = make_trace()
    scorer = MagicMock()
    scorer.name = "test_scorer"

    result_trace = MagicMock()
    result_trace.info.assessments = [
        Feedback(
            name="test_scorer",
            value=True,
            source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="test"),
        )
    ]

    with patch("mlflow.genai.discovery.utils.mlflow.get_trace", return_value=result_trace):
        verify_scorer(scorer, trace)

    scorer.assert_called_once_with(trace=trace)


def test_test_scorer_no_feedback_raises(make_trace):
    trace = make_trace()
    scorer = MagicMock()
    scorer.name = "test_scorer"

    result_trace = MagicMock()
    result_trace.info.assessments = []

    with (
        patch("mlflow.genai.discovery.utils.mlflow.get_trace", return_value=result_trace),
        pytest.raises(Exception, match="produced no feedback"),
    ):
        verify_scorer(scorer, trace)


def test_test_scorer_null_value_raises(make_trace):
    trace = make_trace()
    scorer = MagicMock()
    scorer.name = "test_scorer"

    feedback = MagicMock(spec=Feedback)
    feedback.name = "test_scorer"
    feedback.value = None
    feedback.error_message = "model API error"

    result_trace = MagicMock()
    result_trace.info.assessments = [feedback]

    with (
        patch("mlflow.genai.discovery.utils.mlflow.get_trace", return_value=result_trace),
        pytest.raises(Exception, match="model API error"),
    ):
        verify_scorer(scorer, trace)
