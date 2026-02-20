from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mlflow.genai.discovery.entities import (
    _BatchTraceAnalysisResult,
    _IdentifiedIssue,
    _IssueClusteringResult,
    _ScorerInstructionsResult,
    _ScorerSpec,
    _TraceAnalysis,
)
from mlflow.genai.discovery.pipeline import discover_issues
from mlflow.genai.evaluation.entities import EvaluationResult


def _mock_start_run(**kwargs):
    mock_run = MagicMock()
    mock_run.info.run_id = "run-id"
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_run)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def test_discover_issues_no_experiment():
    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value=None),
        pytest.raises(Exception, match="No experiment specified"),
    ):
        discover_issues()


def test_discover_issues_empty_experiment():
    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.mlflow.search_traces", return_value=[]),
    ):
        result = discover_issues()

    assert result.issues == []
    assert result.total_traces_analyzed == 0


def test_discover_issues_all_traces_pass(make_trace):
    traces = [make_trace() for _ in range(5)]
    test_df = pd.DataFrame(
        {
            "_issue_discovery_judge/value": [True],
            "_issue_discovery_judge/rationale": ["ok"],
            "trace": [traces[0]],
        }
    )
    test_eval = EvaluationResult(run_id="run-test", metrics={}, result_df=test_df)
    result_df = pd.DataFrame(
        {
            "_issue_discovery_judge/value": [True] * 5,
            "_issue_discovery_judge/rationale": ["good"] * 5,
            "trace": traces,
        }
    )
    triage_eval = EvaluationResult(run_id="run-1", metrics={}, result_df=result_df)

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.mlflow.search_traces", return_value=traces),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            side_effect=[test_eval, triage_eval],
        ),
        patch("mlflow.genai.discovery.pipeline.mlflow.MlflowClient"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.start_run",
            side_effect=_mock_start_run,
        ),
    ):
        result = discover_issues()

    assert result.issues == []
    assert "no issues found" in result.summary.lower()


def test_discover_issues_full_pipeline(make_trace):
    traces = [make_trace() for _ in range(10)]

    test_df = pd.DataFrame(
        {
            "_issue_discovery_judge/value": [True],
            "_issue_discovery_judge/rationale": ["ok"],
            "trace": [traces[0]],
        }
    )
    test_eval = EvaluationResult(run_id="run-test", metrics={}, result_df=test_df)

    triage_df = pd.DataFrame(
        {
            "_issue_discovery_judge/value": [False] * 3 + [True] * 7,
            "_issue_discovery_judge/rationale": ["bad"] * 3 + ["good"] * 7,
            "trace": traces,
        }
    )
    triage_eval = EvaluationResult(run_id="run-triage", metrics={}, result_df=triage_df)

    deep_analysis_result = _BatchTraceAnalysisResult(
        analyses=[
            _TraceAnalysis(
                trace_index=i,
                failure_category="latency",
                failure_summary="Slow response",
                root_cause_hypothesis="Complex queries",
                affected_spans=["llm_call"],
                severity=3,
            )
            for i in range(3)
        ]
    )

    clustering_result = _IssueClusteringResult(
        issues=[
            _IdentifiedIssue(
                name="slow_response",
                description="Responses take too long",
                root_cause="Complex queries",
                example_indices=[0, 1],
                confidence=90,
            ),
        ]
    )

    scorer_instructions_result = _ScorerInstructionsResult(
        scorers=[
            _ScorerSpec(
                name="slow_response",
                detection_instructions="Check the {{ trace }} execution duration",
            )
        ]
    )

    validation_df = pd.DataFrame(
        {
            "slow_response/value": [False] * 3 + [True] * 7,
            "slow_response/rationale": ["slow"] * 3 + ["fast"] * 7,
        }
    )
    validation_eval = EvaluationResult(run_id="run-validate", metrics={}, result_df=validation_df)

    mock_llm = MagicMock(
        side_effect=[deep_analysis_result, clustering_result, scorer_instructions_result]
    )
    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.mlflow.search_traces", return_value=traces),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            side_effect=[test_eval, triage_eval, validation_eval],
        ),
        patch(
            "mlflow.genai.discovery.utils.get_chat_completions_with_structured_output",
            mock_llm,
        ),
        patch(
            "mlflow.genai.discovery.pipeline.get_chat_completions_with_structured_output",
            mock_llm,
        ),
        patch("mlflow.genai.discovery.pipeline.mlflow.MlflowClient"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.start_run",
            side_effect=_mock_start_run,
        ),
    ):
        result = discover_issues(triage_sample_size=10)

    assert len(result.issues) == 1
    assert result.issues[0].name == "slow_response"
    assert result.issues[0].frequency == pytest.approx(0.3)
    assert result.issues[0].example_trace_ids == [
        traces[0].info.trace_id,
        traces[1].info.trace_id,
    ]
    assert result.triage_run_id == "run-triage"
    assert result.validation_run_id == "run-validate"


def test_discover_issues_low_confidence_issues_filtered(make_trace):
    traces = [make_trace() for _ in range(5)]

    test_df = pd.DataFrame(
        {
            "_issue_discovery_judge/value": [True],
            "_issue_discovery_judge/rationale": ["ok"],
            "trace": [traces[0]],
        }
    )
    test_eval = EvaluationResult(run_id="run-test", metrics={}, result_df=test_df)

    triage_df = pd.DataFrame(
        {
            "_issue_discovery_judge/value": [False] * 2 + [True] * 3,
            "_issue_discovery_judge/rationale": ["bad"] * 2 + ["good"] * 3,
            "trace": traces,
        }
    )
    triage_eval = EvaluationResult(run_id="run-1", metrics={}, result_df=triage_df)

    deep_analysis_result = _BatchTraceAnalysisResult(
        analyses=[
            _TraceAnalysis(
                trace_index=i,
                failure_category="other",
                failure_summary="Rare issue",
                root_cause_hypothesis="Unknown",
                affected_spans=["span"],
                severity=2,
            )
            for i in range(2)
        ]
    )

    clustering_result = _IssueClusteringResult(
        issues=[
            _IdentifiedIssue(
                name="rare_issue",
                description="Happens very rarely",
                root_cause="Unknown",
                example_indices=[0],
                confidence=80,
            ),
        ]
    )

    mock_llm = MagicMock(side_effect=[deep_analysis_result, clustering_result])
    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.mlflow.search_traces", return_value=traces),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            side_effect=[test_eval, triage_eval],
        ),
        patch(
            "mlflow.genai.discovery.utils.get_chat_completions_with_structured_output",
            mock_llm,
        ),
        patch(
            "mlflow.genai.discovery.pipeline.get_chat_completions_with_structured_output",
            mock_llm,
        ),
        patch("mlflow.genai.discovery.pipeline.mlflow.MlflowClient"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.start_run",
            side_effect=_mock_start_run,
        ),
    ):
        result = discover_issues(triage_sample_size=5)

    assert len(result.issues) == 0


def test_discover_issues_explicit_experiment_id():
    with patch(
        "mlflow.genai.discovery.pipeline.mlflow.search_traces",
        return_value=[],
    ) as mock_search:
        discover_issues(experiment_id="exp-42")

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["locations"] == ["exp-42"]


def test_discover_issues_passes_filter_string():
    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.search_traces", return_value=[]
        ) as mock_search,
    ):
        discover_issues(filter_string="tag.env = 'prod'")

    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["filter_string"] == "tag.env = 'prod'"


def test_discover_issues_custom_satisfaction_scorer(make_trace):
    custom_scorer = MagicMock()
    custom_scorer.name = "custom"
    traces = [make_trace()]

    test_df = pd.DataFrame(
        {"custom/value": [True], "custom/rationale": ["ok"], "trace": [traces[0]]}
    )
    test_eval = EvaluationResult(run_id="run-test", metrics={}, result_df=test_df)

    result_df = pd.DataFrame(
        {
            "custom/value": [True],
            "custom/rationale": ["good"],
            "trace": traces,
        }
    )
    triage_eval = EvaluationResult(run_id="run-1", metrics={}, result_df=result_df)

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.mlflow.search_traces", return_value=traces),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            side_effect=[test_eval, triage_eval],
        ) as mock_eval,
        patch("mlflow.genai.discovery.pipeline.mlflow.MlflowClient"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.start_run",
            side_effect=_mock_start_run,
        ),
    ):
        discover_issues(satisfaction_scorer=custom_scorer)

    assert mock_eval.call_count == 2
    triage_call_kwargs = mock_eval.call_args_list[1][1]
    assert triage_call_kwargs["scorers"] == [custom_scorer]
