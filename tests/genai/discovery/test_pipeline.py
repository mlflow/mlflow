from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.genai.discovery.entities import Issue, _ConversationAnalysis, _IdentifiedIssue
from mlflow.genai.discovery.pipeline import (
    _annotate_issue_traces,
    _format_trace_content,
    _is_non_issue,
    _recluster_singletons,
    discover_issues,
    severity_gte,
    severity_max,
    verify_scorer,
)
from mlflow.genai.evaluation.entities import EvaluationResult

from tests.genai.discovery.conftest import _TestScorer


@pytest.fixture(autouse=True)
def _mock_set_experiment():
    with patch("mlflow.genai.discovery.pipeline.mlflow.set_experiment"):
        yield


def _mock_start_run(**kwargs):
    mock_run = MagicMock()
    mock_run.info.run_id = "run-id"
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_run)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _triage_eval(run_id="run-1"):
    return EvaluationResult(run_id=run_id, metrics={}, result_df=None)


def test_discover_issues_no_experiment():
    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value=None),
        pytest.raises(Exception, match="No experiment specified|Pass traces"),
    ):
        discover_issues()


def test_discover_issues_empty_experiment():
    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.sample_traces", return_value=[]),
    ):
        result = discover_issues()

    assert result.issues == []
    assert result.total_traces_analyzed == 0


def test_discover_issues_all_traces_pass(make_trace):
    traces = [make_trace() for _ in range(5)]

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.sample_traces", return_value=traces),
        patch("mlflow.genai.discovery.pipeline.verify_scorer"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            return_value=_triage_eval(),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=([], {}),
        ) as mock_extract,
        patch("mlflow.genai.discovery.pipeline.mlflow.MlflowClient"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.start_run",
            side_effect=_mock_start_run,
        ),
    ):
        result = discover_issues()

    mock_extract.assert_called_once()
    assert result.issues == []
    assert "no issues found" in result.summary.lower()


def test_discover_issues_full_pipeline(make_trace):
    traces = [make_trace() for _ in range(10)]
    failing = traces[:3]
    rationale_map = {t.info.trace_id: "bad" for t in failing}

    cluster_summary_issue = _IdentifiedIssue(
        name="slow_response",
        description="Responses take too long",
        root_cause="Complex queries",
        example_indices=[0, 1],
        severity="high",
    )

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.sample_traces", return_value=traces),
        patch("mlflow.genai.discovery.pipeline.verify_scorer"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            return_value=_triage_eval("run-triage"),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=(failing, rationale_map),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failure_labels",
            return_value=["label1", "label2", "label3"],
        ),
        patch(
            "mlflow.genai.discovery.pipeline.cluster_by_llm",
            return_value=[[0, 1, 2]],
        ) as mock_cluster,
        patch(
            "mlflow.genai.discovery.pipeline.summarize_cluster",
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


def test_discover_issues_low_severity_issues_filtered(make_trace):
    traces = [make_trace() for _ in range(5)]
    failing = traces[:2]
    rationale_map = {t.info.trace_id: "bad" for t in failing}

    # summarize_cluster returns issue with low severity (below MIN_SEVERITY="low")
    low_severity_issue = _IdentifiedIssue(
        name="rare_issue",
        description="Happens very rarely",
        root_cause="Unknown",
        example_indices=[0],
        severity="not_an_issue",
    )

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.sample_traces", return_value=traces),
        patch("mlflow.genai.discovery.pipeline.verify_scorer"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            return_value=_triage_eval(),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=(failing, rationale_map),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failure_labels",
            return_value=["label1", "label2"],
        ),
        patch(
            "mlflow.genai.discovery.pipeline.cluster_by_llm",
            return_value=[[0, 1]],
        ) as mock_cluster,
        patch(
            "mlflow.genai.discovery.pipeline.summarize_cluster",
            return_value=low_severity_issue,
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
    # (severity=not_an_issue < low triggers re-splitting of the 2-member cluster)
    assert mock_summarize.call_count == 3
    assert len(result.issues) == 0


def test_discover_issues_explicit_experiment_id():
    with patch(
        "mlflow.genai.discovery.pipeline.sample_traces",
        return_value=[],
    ) as mock_sample:
        discover_issues(experiment_id="exp-42")

    mock_sample.assert_called_once()
    search_kwargs = mock_sample.call_args[0][1]
    assert search_kwargs["locations"] == ["exp-42"]


def test_discover_issues_passes_filter_string():
    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.sample_traces", return_value=[]) as mock_sample,
    ):
        discover_issues(filter_string="tag.env = 'prod'")

    search_kwargs = mock_sample.call_args[0][1]
    assert search_kwargs["filter_string"] == "tag.env = 'prod'"


def test_discover_issues_custom_satisfaction_scorer(make_trace):
    custom_scorer = _TestScorer(name="custom")
    traces = [make_trace()]

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.sample_traces", return_value=traces),
        patch("mlflow.genai.discovery.pipeline.verify_scorer"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            return_value=_triage_eval(),
        ) as mock_eval,
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=([], {}),
        ),
        patch("mlflow.genai.discovery.pipeline.mlflow.MlflowClient"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.start_run",
            side_effect=_mock_start_run,
        ),
    ):
        discover_issues(scorers=[custom_scorer])

    mock_eval.assert_called_once()
    triage_call_kwargs = mock_eval.call_args[1]
    assert triage_call_kwargs["scorers"] == [custom_scorer]


def test_discover_issues_additional_scorers(make_trace):
    custom_scorer = _TestScorer(name="custom")
    extra_scorer = _TestScorer(name="extra")
    traces = [make_trace()]

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.sample_traces", return_value=traces),
        patch("mlflow.genai.discovery.pipeline.verify_scorer"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            return_value=_triage_eval(),
        ) as mock_eval,
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=([], {}),
        ),
        patch("mlflow.genai.discovery.pipeline.mlflow.MlflowClient"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.start_run",
            side_effect=_mock_start_run,
        ),
    ):
        discover_issues(scorers=[custom_scorer, extra_scorer])

    mock_eval.assert_called_once()
    triage_call_kwargs = mock_eval.call_args[1]
    assert triage_call_kwargs["scorers"] == [custom_scorer, extra_scorer]


@pytest.mark.parametrize(
    ("name", "description", "root_cause", "expected"),
    [
        # Canonical keyword — primary mechanism
        ("NO_ISSUE_DETECTED", "Goals were met", "N/A", True),
        ("no_issue_detected", "All good", "N/A", True),
        # Fallback patterns in name/description
        ("No issues detected [general]", "Everything looks fine", "test", True),
        ("No problems found", "All traces passed", "test", True),
        ("Missing data [api]", "No errors found in logs", "test", True),
        # Fallback pattern in root_cause
        ("General analysis", "System output summary", "no issues found in analysis", True),
        # New expanded patterns
        ("System review", "The system is functioning correctly overall", "N/A", True),
        ("Goals achieved", "The user's goals were achieved in this session", "N/A", True),
        ("Evaluation", "System is working as intended for this use case", "N/A", True),
        ("Summary", "Nothing wrong with the response quality", "N/A", True),
        ("Assessment", "No significant issue identified here", "N/A", True),
        # Real issues — not filtered
        ("Slow response times [api]", "Responses are slow", "Complex queries", False),
        ("Data errors [db]", "Schema validation passes but data is wrong", "Misconfig", False),
    ],
)
def test_is_non_issue(name, description, root_cause, expected):
    issue = _IdentifiedIssue(
        name=name,
        description=description,
        root_cause=root_cause,
        example_indices=[0],
        severity="high",
    )
    assert _is_non_issue(issue) == expected


def test_discover_issues_filters_no_issue_results(make_trace):
    traces = [make_trace() for _ in range(10)]
    failing = traces[:3]
    rationale_map = {t.info.trace_id: "bad" for t in failing}

    no_issue_result = _IdentifiedIssue(
        name="No issues detected [general]",
        description="This trace analysis found no identifiable issues.",
        root_cause="N/A",
        example_indices=[0, 1, 2],
        severity="high",
    )

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.sample_traces", return_value=traces),
        patch("mlflow.genai.discovery.pipeline.verify_scorer"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            return_value=_triage_eval("run-triage"),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=(failing, rationale_map),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failure_labels",
            return_value=["label1", "label2", "label3"],
        ),
        patch(
            "mlflow.genai.discovery.pipeline.cluster_by_llm",
            return_value=[[0, 1, 2]],
        ),
        patch(
            "mlflow.genai.discovery.pipeline.summarize_cluster",
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


def test_discover_issues_filters_canonical_no_issue_keyword(make_trace):
    traces = [make_trace() for _ in range(10)]
    failing = traces[:3]
    rationale_map = {t.info.trace_id: "bad" for t in failing}

    no_issue_result = _IdentifiedIssue(
        name="NO_ISSUE_DETECTED",
        description="The analyses do not represent a real failure.",
        root_cause="N/A",
        example_indices=[0, 1, 2],
        severity="not_an_issue",
    )

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.sample_traces", return_value=traces),
        patch("mlflow.genai.discovery.pipeline.verify_scorer"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            return_value=_triage_eval("run-triage"),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=(failing, rationale_map),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failure_labels",
            return_value=["label1", "label2", "label3"],
        ),
        patch(
            "mlflow.genai.discovery.pipeline.cluster_by_llm",
            return_value=[[0, 1, 2]],
        ),
        patch(
            "mlflow.genai.discovery.pipeline.summarize_cluster",
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


def _make_issue(**kwargs):
    defaults = {
        "issue_id": "test-id",
        "run_id": "test-run",
        "name": "Test issue",
        "description": "Test description",
        "root_cause": "Test cause",
        "example_trace_ids": [],
        "frequency": 0.0,
        "severity": "high",
    }
    defaults.update(kwargs)
    return Issue(**defaults)


def test_annotate_traces_annotates_each_trace_with_feedback():
    issues = [
        _make_issue(
            name="Slow responses [api]",
            description="Responses take too long",
            root_cause="Complex queries",
            example_trace_ids=["trace-1", "trace-2"],
            frequency=0.5,
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
        assert c.kwargs["name"] == "issue: Slow responses [api]"
        assert c.kwargs["value"] is False
        assert "slow response" in c.kwargs["rationale"]


def test_annotate_traces_no_work_items_returns_early():
    issues = [
        _make_issue(
            name="Empty issue",
            description="No traces",
            root_cause="N/A",
        ),
    ]

    with patch("litellm.completion") as mock_completion:
        _annotate_issue_traces(issues, {}, {}, "openai:/gpt-5-mini")

    mock_completion.assert_not_called()


def test_annotate_traces_llm_failure_falls_back_to_triage_rationale():
    issues = [
        _make_issue(
            name="Timeout [api]",
            description="Request timed out",
            root_cause="Upstream latency",
            example_trace_ids=["t1"],
            frequency=0.1,
            severity="low",
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


def test_annotate_traces_log_feedback_failure_handled_gracefully():
    issues = [
        _make_issue(
            name="Error [db]",
            description="DB error",
            root_cause="Connection pool",
            example_trace_ids=["t1"],
            frequency=0.1,
            severity="low",
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

    # No error raised — failure handled gracefully


def test_annotate_traces_multiple_issues_annotated_independently():
    issues = [
        _make_issue(
            name="Issue A",
            description="Desc A",
            root_cause="Cause A",
            example_trace_ids=["t1"],
            frequency=0.2,
        ),
        _make_issue(
            name="Issue B",
            description="Desc B",
            root_cause="Cause B",
            example_trace_ids=["t2", "t3"],
            frequency=0.3,
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
    assert feedback_names.count("issue: Issue A") == 1
    assert feedback_names.count("issue: Issue B") == 2


def test_annotate_traces_session_level_logs_on_first_trace():
    issues = [
        _make_issue(
            name="Slow responses [api]",
            description="Responses take too long",
            root_cause="Complex queries",
            example_trace_ids=["trace-1", "trace-2", "trace-3"],
            frequency=0.5,
        ),
    ]
    rationale_map = {
        "trace-1": "Response was slow",
        "trace-2": "Timed out",
        "trace-3": "Also slow",
    }
    trace_to_session = {
        "trace-1": "session-A",
        "trace-2": "session-A",
        "trace-3": "session-B",
    }
    session_first_trace = {
        "session-A": "trace-0",
        "session-B": "trace-3",
    }

    with (
        patch(
            "litellm.completion",
            return_value=_make_litellm_response("Session-level annotation."),
        ),
        patch("mlflow.genai.discovery.pipeline.mlflow.log_feedback") as mock_feedback,
    ):
        _annotate_issue_traces(
            issues,
            rationale_map,
            {},
            "openai:/gpt-5-mini",
            trace_to_session=trace_to_session,
            session_first_trace=session_first_trace,
        )

    # 2 sessions -> 2 feedback calls (not 3 per-trace)
    assert mock_feedback.call_count == 2
    logged_trace_ids = {c.kwargs["trace_id"] for c in mock_feedback.call_args_list}
    assert logged_trace_ids == {"trace-0", "trace-3"}
    for c in mock_feedback.call_args_list:
        assert c.kwargs["name"] == "issue: Slow responses [api]"
        metadata = c.kwargs["metadata"]
        assert "mlflow.trace.session" in metadata
    session_ids = {
        c.kwargs["metadata"]["mlflow.trace.session"] for c in mock_feedback.call_args_list
    }
    assert session_ids == {"session-A", "session-B"}


def test_format_trace_content_includes_errors(make_trace):
    trace = make_trace(error_span=True)
    content = _format_trace_content(trace)
    assert "Errors:" in content
    assert "Connection failed" in content


def test_format_trace_content_no_errors(make_trace):
    trace = make_trace()
    content = _format_trace_content(trace)
    assert "Errors:" not in content


def test_recluster_merges_similar_singletons():
    analyses = [
        _ConversationAnalysis(
            surface="tool error A",
            root_cause="API failure",
            affected_trace_ids=["t1"],
        ),
        _ConversationAnalysis(
            surface="tool error B",
            root_cause="API failure",
            affected_trace_ids=["t2"],
        ),
    ]
    singletons = [
        _IdentifiedIssue(
            name="Issue A",
            description="Error A",
            root_cause="API failure",
            example_indices=[0],
            severity="high",
        ),
        _IdentifiedIssue(
            name="Issue B",
            description="Error B",
            root_cause="API failure",
            example_indices=[1],
            severity="high",
        ),
    ]
    labels = ["[tool_call] API timeout", "[tool_call] API timeout"]

    merged_issue = _IdentifiedIssue(
        name="Issue: API timeouts",
        description="Merged",
        root_cause="API failure",
        example_indices=[0, 1],
        severity="high",
    )

    with (
        patch(
            "mlflow.genai.discovery.pipeline.cluster_by_llm",
            return_value=[[0, 1]],
        ) as mock_cluster,
        patch(
            "mlflow.genai.discovery.pipeline.summarize_cluster",
            return_value=merged_issue,
        ) as mock_summarize,
    ):
        result = _recluster_singletons(singletons, labels, analyses, "openai:/gpt-5", 25)

    mock_cluster.assert_called_once()
    mock_summarize.assert_called_once()
    assert len(result) == 1
    assert result[0].example_indices == [0, 1]


def test_recluster_keeps_unmerged_singletons():
    analyses = [
        _ConversationAnalysis(
            surface="error A",
            root_cause="A",
            affected_trace_ids=["t1"],
        ),
        _ConversationAnalysis(
            surface="error B",
            root_cause="B",
            affected_trace_ids=["t2"],
        ),
    ]
    singletons = [
        _IdentifiedIssue(
            name="Issue A",
            description="A",
            root_cause="A",
            example_indices=[0],
            severity="high",
        ),
        _IdentifiedIssue(
            name="Issue B",
            description="B",
            root_cause="B",
            example_indices=[1],
            severity="high",
        ),
    ]
    labels = ["[path_a] symptom a", "[path_b] symptom b"]

    with patch(
        "mlflow.genai.discovery.pipeline.cluster_by_llm",
        return_value=[[0], [1]],
    ) as mock_cluster:
        result = _recluster_singletons(singletons, labels, analyses, "openai:/gpt-5", 25)

    mock_cluster.assert_called_once()
    assert len(result) == 2


def test_recluster_single_singleton_returns_as_is():
    singletons = [
        _IdentifiedIssue(
            name="Solo",
            description="Only one",
            root_cause="N/A",
            example_indices=[0],
            severity="high",
        ),
    ]
    result = _recluster_singletons(singletons, ["label"], [], "m", 25)
    assert len(result) == 1
    assert result[0].name == "Solo"


def test_recluster_low_severity_merge_keeps_originals():
    analyses = [
        _ConversationAnalysis(
            surface="A",
            root_cause="A",
            affected_trace_ids=["t1"],
        ),
        _ConversationAnalysis(
            surface="B",
            root_cause="B",
            affected_trace_ids=["t2"],
        ),
    ]
    singletons = [
        _IdentifiedIssue(
            name="A",
            description="A",
            root_cause="A",
            example_indices=[0],
            severity="low",
        ),
        _IdentifiedIssue(
            name="B",
            description="B",
            root_cause="B",
            example_indices=[1],
            severity="low",
        ),
    ]
    labels = ["label a", "label b"]

    low_severity_merged = _IdentifiedIssue(
        name="Merged",
        description="M",
        root_cause="M",
        example_indices=[0, 1],
        severity="not_an_issue",
    )

    with (
        patch(
            "mlflow.genai.discovery.pipeline.cluster_by_llm",
            return_value=[[0, 1]],
        ),
        patch(
            "mlflow.genai.discovery.pipeline.summarize_cluster",
            return_value=low_severity_merged,
        ),
    ):
        result = _recluster_singletons(singletons, labels, analyses, "openai:/gpt-5", 25)

    assert len(result) == 2
    assert result[0].name == "A"
    assert result[1].name == "B"


def test_severity_helpers():
    assert severity_gte("high", "low")
    assert severity_gte("low", "low")
    assert severity_gte("medium", "low")
    assert not severity_gte("not_an_issue", "low")

    assert severity_max("high", "low") == "high"
    assert severity_max("not_an_issue", "low") == "low"
    assert severity_max("not_an_issue", "medium") == "medium"


# ---- verify_scorer ----


def test_verify_scorer_happy_path(make_trace):
    trace = make_trace()
    feedback = Feedback(
        name="test_scorer",
        value=True,
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="test"),
    )
    scorer = MagicMock(return_value=feedback)
    scorer.name = "test_scorer"

    verify_scorer(scorer, trace)

    scorer.assert_called_once_with(trace=trace)


def test_verify_scorer_with_session(make_trace):
    trace = make_trace()
    session = [trace, make_trace()]
    feedback = Feedback(
        name="test_scorer",
        value=True,
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="test"),
    )
    scorer = MagicMock(return_value=feedback)
    scorer.name = "test_scorer"

    verify_scorer(scorer, trace, session=session)

    scorer.assert_called_once_with(session=session)


def test_verify_scorer_non_feedback_raises(make_trace):
    trace = make_trace()
    scorer = MagicMock(return_value="not a Feedback")
    scorer.name = "test_scorer"

    with pytest.raises(Exception, match="returned str instead of Feedback"):
        verify_scorer(scorer, trace)


def test_verify_scorer_null_value_raises(make_trace):
    trace = make_trace()
    feedback = MagicMock(spec=Feedback)
    feedback.value = None
    feedback.error_message = "model API error"
    scorer = MagicMock(return_value=feedback)
    scorer.name = "test_scorer"

    with pytest.raises(Exception, match="returned null value"):
        verify_scorer(scorer, trace)
