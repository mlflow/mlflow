from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mlflow.genai.discovery.entities import _IdentifiedIssue
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
        pytest.raises(Exception, match="No experiment specified|Pass traces"),
    ):
        discover_issues()


def test_discover_issues_empty_experiment():
    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline._sample_traces", return_value=[]),
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
        patch("mlflow.genai.discovery.pipeline._sample_traces", return_value=traces),
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

    cluster_summary_issue = _IdentifiedIssue(
        name="slow_response",
        description="Responses take too long",
        root_cause="Complex queries",
        example_indices=[0, 1],
        confidence=90,
    )

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline._sample_traces", return_value=traces),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            side_effect=[test_eval, triage_eval],
        ),
        patch(
            "mlflow.genai.discovery.pipeline._extract_failure_labels",
            return_value=["label1", "label2", "label3"],
        ),
        patch(
            "mlflow.genai.discovery.pipeline._cluster_analyses",
            return_value=[[0, 1, 2]],
        ) as mock_cluster,
        patch(
            "mlflow.genai.discovery.pipeline._summarize_cluster",
            return_value=cluster_summary_issue,
        ) as mock_summarize,
        patch("mlflow.genai.discovery.pipeline.mlflow.MlflowClient"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.start_run",
            side_effect=_mock_start_run,
        ),
        patch("mlflow.genai.discovery.pipeline._annotate_issue_traces") as mock_annotate,
    ):
        result = discover_issues(triage_sample_size=10)

    mock_cluster.assert_called_once()
    mock_summarize.assert_called_once()
    mock_annotate.assert_called_once()
    assert len(result.issues) == 1
    assert result.issues[0].name == "slow_response"
    assert result.issues[0].frequency == pytest.approx(0.2)
    assert result.triage_run_id == "run-triage"
    assert result.validation_run_id is None


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

    # _summarize_cluster returns issue with low confidence (below _MIN_CONFIDENCE=75)
    low_confidence_issue = _IdentifiedIssue(
        name="rare_issue",
        description="Happens very rarely",
        root_cause="Unknown",
        example_indices=[0],
        confidence=50,
    )

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline._sample_traces", return_value=traces),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            side_effect=[test_eval, triage_eval],
        ),
        patch(
            "mlflow.genai.discovery.pipeline._extract_failure_labels",
            return_value=["label1", "label2"],
        ),
        patch(
            "mlflow.genai.discovery.pipeline._cluster_analyses",
            return_value=[[0, 1]],
        ) as mock_cluster,
        patch(
            "mlflow.genai.discovery.pipeline._summarize_cluster",
            return_value=low_confidence_issue,
        ) as mock_summarize,
        patch("mlflow.genai.discovery.pipeline.mlflow.MlflowClient"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.start_run",
            side_effect=_mock_start_run,
        ),
    ):
        result = discover_issues(triage_sample_size=5)

    mock_cluster.assert_called_once()
    # Called once for the original cluster, then twice more for singleton re-splits
    # (confidence=50 < 75 triggers re-splitting of the 2-member cluster)
    assert mock_summarize.call_count == 3
    assert len(result.issues) == 0


def test_discover_issues_explicit_experiment_id():
    with patch(
        "mlflow.genai.discovery.pipeline._sample_traces",
        return_value=[],
    ) as mock_sample:
        discover_issues(experiment_id="exp-42")

    mock_sample.assert_called_once()
    search_kwargs = mock_sample.call_args[0][1]
    assert search_kwargs["locations"] == ["exp-42"]


def test_discover_issues_passes_filter_string():
    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline._sample_traces", return_value=[]) as mock_sample,
    ):
        discover_issues(filter_string="tag.env = 'prod'")

    search_kwargs = mock_sample.call_args[0][1]
    assert search_kwargs["filter_string"] == "tag.env = 'prod'"


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
        patch("mlflow.genai.discovery.pipeline._sample_traces", return_value=traces),
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


def test_discover_issues_additional_scorers(make_trace):
    custom_scorer = MagicMock()
    custom_scorer.name = "custom"
    extra_scorer = MagicMock()
    extra_scorer.name = "extra"
    traces = [make_trace()]

    test_df = pd.DataFrame(
        {"custom/value": [True], "custom/rationale": ["ok"], "trace": [traces[0]]}
    )
    test_eval = EvaluationResult(run_id="run-test", metrics={}, result_df=test_df)

    result_df = pd.DataFrame(
        {
            "custom/value": [True],
            "custom/rationale": ["good"],
            "extra/value": [True],
            "extra/rationale": ["fine"],
            "trace": traces,
        }
    )
    triage_eval = EvaluationResult(run_id="run-1", metrics={}, result_df=result_df)

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline._sample_traces", return_value=traces),
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
        discover_issues(
            satisfaction_scorer=custom_scorer,
            additional_scorers=[extra_scorer],
        )

    assert mock_eval.call_count == 2
    triage_call_kwargs = mock_eval.call_args_list[1][1]
    assert triage_call_kwargs["scorers"] == [custom_scorer, extra_scorer]


@pytest.mark.parametrize(
    ("name", "description", "expected"),
    [
        ("No issues detected [general]", "Everything looks fine", True),
        ("No problems found", "All traces passed", True),
        ("Slow response times [api]", "Responses are slow", False),
        ("Data errors [db]", "Schema validation passes but data is wrong", False),
        ("Missing data [api]", "No errors found in logs", True),
    ],
)
def test_is_non_issue(name, description, expected):
    from mlflow.genai.discovery.pipeline import _is_non_issue

    issue = _IdentifiedIssue(
        name=name,
        description=description,
        root_cause="test",
        example_indices=[0],
        confidence=90,
    )
    assert _is_non_issue(issue) == expected


def test_discover_issues_filters_no_issue_results(make_trace):
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

    no_issue_result = _IdentifiedIssue(
        name="No issues detected [general]",
        description="This trace analysis found no identifiable issues.",
        root_cause="N/A",
        example_indices=[0, 1, 2],
        confidence=100,
    )

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline._sample_traces", return_value=traces),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            side_effect=[test_eval, triage_eval],
        ),
        patch(
            "mlflow.genai.discovery.pipeline._extract_failure_labels",
            return_value=["label1", "label2", "label3"],
        ),
        patch(
            "mlflow.genai.discovery.pipeline._cluster_analyses",
            return_value=[[0, 1, 2]],
        ),
        patch(
            "mlflow.genai.discovery.pipeline._summarize_cluster",
            return_value=no_issue_result,
        ),
        patch("mlflow.genai.discovery.pipeline.mlflow.MlflowClient"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.start_run",
            side_effect=_mock_start_run,
        ),
    ):
        result = discover_issues(triage_sample_size=10)

    assert len(result.issues) == 0


def _make_litellm_response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


class TestAnnotateIssueTraces:
    def test_annotates_each_trace_with_feedback(self):
        from mlflow.genai.discovery.entities import Issue
        from mlflow.genai.discovery.pipeline import _annotate_issue_traces

        issues = [
            Issue(
                name="Slow responses [api]",
                description="Responses take too long",
                root_cause="Complex queries",
                example_trace_ids=["trace-1", "trace-2"],
                scorer=None,
                frequency=0.5,
                confidence=90,
                rationale_examples=[],
            ),
        ]
        rationale_map = {
            "trace-1": "Response was slow and incomplete",
            "trace-2": "Timed out on user request",
        }

        with (
            patch(
                "litellm.completion",
                return_value=_make_litellm_response("This trace shows slow response behavior."),
            ) as mock_completion,
            patch("mlflow.genai.discovery.pipeline.mlflow.log_feedback") as mock_feedback,
        ):
            _annotate_issue_traces(issues, rationale_map, {}, "openai:/gpt-5-mini")

        assert mock_completion.call_count == 2
        assert mock_feedback.call_count == 2
        for c in mock_feedback.call_args_list:
            assert c.kwargs["name"] == "Slow responses [api]"
            assert c.kwargs["value"] is False
            assert "slow response" in c.kwargs["rationale"]

    def test_populates_rationale_examples(self):
        from mlflow.genai.discovery.entities import Issue
        from mlflow.genai.discovery.pipeline import _annotate_issue_traces

        issues = [
            Issue(
                name="Bad data [db]",
                description="Wrong data returned",
                root_cause="Schema mismatch",
                example_trace_ids=["t1", "t2", "t3", "t4"],
                scorer=None,
                frequency=0.4,
                confidence=85,
                rationale_examples=[],
            ),
        ]
        rationale_map = {"t1": "r1", "t2": "r2", "t3": "r3", "t4": "r4"}

        with (
            patch(
                "litellm.completion",
                return_value=_make_litellm_response("Annotation text."),
            ),
            patch("mlflow.genai.discovery.pipeline.mlflow.log_feedback"),
        ):
            _annotate_issue_traces(issues, rationale_map, {}, "openai:/gpt-5-mini")

        assert len(issues[0].rationale_examples) == 3
        assert all(ex == "Annotation text." for ex in issues[0].rationale_examples)

    def test_no_work_items_returns_early(self):
        from mlflow.genai.discovery.entities import Issue
        from mlflow.genai.discovery.pipeline import _annotate_issue_traces

        issues = [
            Issue(
                name="Empty issue",
                description="No traces",
                root_cause="N/A",
                example_trace_ids=[],
                scorer=None,
                frequency=0.0,
                confidence=90,
                rationale_examples=[],
            ),
        ]

        with patch("litellm.completion") as mock_completion:
            _annotate_issue_traces(issues, {}, {}, "openai:/gpt-5-mini")

        mock_completion.assert_not_called()

    def test_llm_failure_falls_back_to_triage_rationale(self):
        from mlflow.genai.discovery.entities import Issue
        from mlflow.genai.discovery.pipeline import _annotate_issue_traces

        issues = [
            Issue(
                name="Timeout [api]",
                description="Request timed out",
                root_cause="Upstream latency",
                example_trace_ids=["t1"],
                scorer=None,
                frequency=0.1,
                confidence=80,
                rationale_examples=[],
            ),
        ]
        rationale_map = {"t1": "Original triage rationale"}

        with (
            patch(
                "litellm.completion",
                side_effect=Exception("LLM unavailable"),
            ),
            patch("mlflow.genai.discovery.pipeline.mlflow.log_feedback") as mock_feedback,
        ):
            _annotate_issue_traces(issues, rationale_map, {}, "openai:/gpt-5-mini")

        mock_feedback.assert_called_once()
        rationale = mock_feedback.call_args.kwargs["rationale"]
        assert "Original triage rationale" in rationale
        assert "Timeout [api]" in rationale

    def test_log_feedback_failure_handled_gracefully(self):
        from mlflow.genai.discovery.entities import Issue
        from mlflow.genai.discovery.pipeline import _annotate_issue_traces

        issues = [
            Issue(
                name="Error [db]",
                description="DB error",
                root_cause="Connection pool",
                example_trace_ids=["t1"],
                scorer=None,
                frequency=0.1,
                confidence=80,
                rationale_examples=[],
            ),
        ]

        with (
            patch(
                "litellm.completion",
                return_value=_make_litellm_response("Annotation."),
            ),
            patch(
                "mlflow.genai.discovery.pipeline.mlflow.log_feedback",
                side_effect=Exception("Tracking server down"),
            ),
        ):
            _annotate_issue_traces(issues, {"t1": "rationale"}, {}, "openai:/gpt-5-mini")

        assert issues[0].rationale_examples == []

    def test_multiple_issues_annotated_independently(self):
        from mlflow.genai.discovery.entities import Issue
        from mlflow.genai.discovery.pipeline import _annotate_issue_traces

        issues = [
            Issue(
                name="Issue A",
                description="Desc A",
                root_cause="Cause A",
                example_trace_ids=["t1"],
                scorer=None,
                frequency=0.2,
                confidence=85,
                rationale_examples=[],
            ),
            Issue(
                name="Issue B",
                description="Desc B",
                root_cause="Cause B",
                example_trace_ids=["t2", "t3"],
                scorer=None,
                frequency=0.3,
                confidence=90,
                rationale_examples=[],
            ),
        ]
        rationale_map = {"t1": "r1", "t2": "r2", "t3": "r3"}

        with (
            patch(
                "litellm.completion",
                return_value=_make_litellm_response("Annotated."),
            ),
            patch("mlflow.genai.discovery.pipeline.mlflow.log_feedback") as mock_feedback,
        ):
            _annotate_issue_traces(issues, rationale_map, {}, "openai:/gpt-5-mini")

        assert mock_feedback.call_count == 3
        feedback_names = [c.kwargs["name"] for c in mock_feedback.call_args_list]
        assert feedback_names.count("Issue A") == 1
        assert feedback_names.count("Issue B") == 2
        assert len(issues[0].rationale_examples) == 1
        assert len(issues[1].rationale_examples) == 2
