import json
from unittest.mock import MagicMock, patch

import pytest

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.genai.discovery.clustering import recluster_singletons
from mlflow.genai.discovery.constants import (
    DEFAULT_MODEL,
    DEFAULT_SCORER_NAME,
    build_satisfaction_instructions,
)
from mlflow.genai.discovery.entities import (
    Issue,
    _ConversationAnalysis,
    _IdentifiedIssue,
    _TriageResult,
)
from mlflow.genai.discovery.pipeline import (
    _annotate_issue_traces,
    _dedup_issues,
    _is_non_issue,
    build_issue_discovery_scorer,
    discover_issues,
)
from mlflow.genai.discovery.utils import get_session_id, verify_scorer
from mlflow.genai.evaluation.context import NoneContext, _set_context
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.utils.mlflow_tags import MLFLOW_RUN_TYPE, MLFLOW_RUN_TYPE_ISSUE_DETECTION

from tests.genai.discovery.conftest import _TestScorer


@pytest.fixture(autouse=True)
def _mock_set_experiment():
    with patch("mlflow.genai.discovery.pipeline.mlflow.set_experiment"):
        yield
    _set_context(NoneContext())


def _mock_start_run(**kwargs):
    mock_run = MagicMock()
    mock_run.info.run_id = "run-id"
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_run)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _triage_eval(run_id="run-1"):
    return EvaluationResult(run_id=run_id, metrics={}, result_df=None)


def create_identified_issue(**kwargs) -> _IdentifiedIssue:
    defaults = {
        "name": "Issue: Test issue",
        "description": "A test issue",
        "root_cause": "A test root cause",
        "example_indices": [0],
        "severity": "high",
        "categories": [],
    }
    defaults.update(kwargs)
    return _IdentifiedIssue(**defaults)


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
            return_value=_TriageResult([], {}, {}),
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
        categories=[],
    )

    mock_issue = _make_issue(name="slow_response", description="Responses take too long")

    with mlflow.start_run() as run:
        triage_run_id = run.info.run_id

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="0"),
        patch("mlflow.genai.discovery.pipeline.sample_traces", return_value=traces),
        patch("mlflow.genai.discovery.pipeline.verify_scorer"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            return_value=_triage_eval(triage_run_id),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=_TriageResult(failing, rationale_map, {}),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failure_labels",
            return_value=(["label1", "label2", "label3"], [0, 1, 2]),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.cluster_by_llm",
            return_value=[[0, 1, 2]],
        ) as mock_cluster,
        patch(
            "mlflow.genai.discovery.pipeline.summarize_cluster",
            return_value=cluster_summary_issue,
        ) as mock_summarize,
        patch("mlflow.tracing.client.TracingClient._create_issue", return_value=mock_issue),
        patch("mlflow.genai.discovery.pipeline.mlflow.MlflowClient"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.start_run",
            side_effect=_mock_start_run,
        ),
        patch("mlflow.genai.discovery.pipeline._annotate_issue_traces") as mock_annotate,
    ):
        result = discover_issues()

    mock_cluster.assert_called_once()
    mock_summarize.assert_called_once()
    mock_annotate.assert_called_once()
    assert len(result.issues) == 1
    assert result.issues[0].name == "slow_response"
    assert result.triage_run_id == triage_run_id


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
        categories=[],
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
            return_value=_TriageResult(failing, rationale_map, {}),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failure_labels",
            return_value=(["label1", "label2"], [0, 1]),
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
        result = discover_issues()

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
            return_value=_TriageResult([], {}, {}),
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
            return_value=_TriageResult([], {}, {}),
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
    ("name", "severity", "expected"),
    [
        # Severity-based detection
        ("Some issue", "not_an_issue", True),
        ("Another issue", "not_an_issue", True),
        # Canonical keyword in name
        ("NO_ISSUE_DETECTED", "high", True),
        ("no_issue_detected", "medium", True),
        ("Issue: NO_ISSUE_DETECTED variant", "low", True),
        # Both severity and keyword
        ("NO_ISSUE_DETECTED", "not_an_issue", True),
        # Real issues — not filtered
        ("Slow response times [api]", "high", False),
        ("Data errors [db]", "medium", False),
        ("Missing data [api]", "low", False),
    ],
)
def test_is_non_issue(name, severity, expected):
    issue = _IdentifiedIssue(
        name=name,
        description="test",
        root_cause="test",
        example_indices=[0],
        severity=severity,
        categories=[],
    )
    assert _is_non_issue(issue) == expected


@pytest.mark.parametrize(
    "issue_name",
    ["No issues detected [general]", "NO_ISSUE_DETECTED"],
)
def test_discover_issues_filters_non_issues(make_trace, issue_name):
    traces = [make_trace() for _ in range(10)]
    failing = traces[:3]
    rationale_map = {t.info.trace_id: "bad" for t in failing}

    no_issue_result = _IdentifiedIssue(
        name=issue_name,
        description="Not a real failure.",
        root_cause="N/A",
        example_indices=[0, 1, 2],
        severity="not_an_issue",
        categories=[],
    )

    with mlflow.start_run() as run:
        triage_run_id = run.info.run_id

    with (
        patch("mlflow.genai.discovery.pipeline._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.pipeline.sample_traces", return_value=traces),
        patch("mlflow.genai.discovery.pipeline.verify_scorer"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            return_value=_triage_eval(triage_run_id),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=_TriageResult(failing, rationale_map, {}),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failure_labels",
            return_value=(["label1", "label2", "label3"], [0, 1, 2]),
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
        result = discover_issues()

    assert len(result.issues) == 0


def _make_litellm_response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _make_issue(**kwargs):
    defaults = {
        "issue_id": "test-id",
        "experiment_id": "0",
        "name": "Test issue",
        "description": "Test description",
        "status": "open",
        "created_timestamp": 0,
        "last_updated_timestamp": 0,
        "severity": "high",
        "root_causes": ["Test cause"],
    }
    defaults.update(kwargs)
    return Issue(**defaults)


def test_annotate_traces_annotates_each_trace_with_feedback():
    issues = [
        _make_issue(
            name="Slow responses [api]",
            description="Responses take too long",
            root_causes=["Complex queries"],
        ),
    ]
    issue_trace_ids = {issues[0].issue_id: ["trace-1", "trace-2"]}
    rationale_map = {
        "trace-1": "Response was slow and incomplete",
        "trace-2": "Timed out on user request",
    }

    with (
        patch(
            "mlflow.genai.discovery.pipeline._call_llm",
            return_value=_make_litellm_response("This trace shows slow response behavior."),
        ) as mock_completion,
        patch("mlflow.genai.discovery.pipeline.mlflow.log_issue") as mock_log_issue,
    ):
        _annotate_issue_traces(issues, issue_trace_ids, rationale_map, {}, "openai:/gpt-5-mini")

    assert mock_completion.call_count == 2
    assert mock_log_issue.call_count == 2
    for c in mock_log_issue.call_args_list:
        assert c.kwargs["issue_name"] == "Slow responses [api]"
        assert c.kwargs["issue_id"] == issues[0].issue_id
        assert "slow response" in c.kwargs["rationale"]


def test_annotate_traces_no_work_items_returns_early():
    issues = [
        _make_issue(
            name="Empty issue",
            description="No traces",
            root_causes=["N/A"],
        ),
    ]

    with patch("mlflow.genai.discovery.pipeline._call_llm") as mock_completion:
        _annotate_issue_traces(issues, {}, {}, {}, "openai:/gpt-5-mini")

    mock_completion.assert_not_called()


def test_annotate_traces_llm_failure_falls_back_to_triage_rationale():
    issues = [
        _make_issue(
            name="Timeout [api]",
            description="Request timed out",
            root_causes=["Upstream latency"],
            severity="low",
        ),
    ]
    issue_trace_ids = {issues[0].issue_id: ["t1"]}
    rationale_map = {"t1": "Original triage rationale"}

    with (
        patch(
            "mlflow.genai.discovery.pipeline._call_llm",
            side_effect=Exception("LLM unavailable"),
        ),
        patch("mlflow.genai.discovery.pipeline.mlflow.log_issue") as mock_log_issue,
    ):
        _annotate_issue_traces(issues, issue_trace_ids, rationale_map, {}, "openai:/gpt-5-mini")

    mock_log_issue.assert_called_once()
    rationale = mock_log_issue.call_args.kwargs["rationale"]
    assert "Original triage rationale" in rationale
    assert "Timeout [api]" in rationale


def test_annotate_traces_log_issue_failure_handled_gracefully():
    issues = [
        _make_issue(
            name="Error [db]",
            description="DB error",
            root_causes=["Connection pool"],
            severity="low",
        ),
    ]
    issue_trace_ids = {issues[0].issue_id: ["t1"]}

    with (
        patch(
            "mlflow.genai.discovery.pipeline._call_llm",
            return_value=_make_litellm_response("Annotation."),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.log_issue",
            side_effect=Exception("Tracking server down"),
        ),
    ):
        _annotate_issue_traces(
            issues, issue_trace_ids, {"t1": "rationale"}, {}, "openai:/gpt-5-mini"
        )

    # No error raised — failure handled gracefully


def test_annotate_traces_multiple_issues_annotated_independently():
    issue_a = _make_issue(
        issue_id="a", name="Issue A", description="Desc A", root_causes=["Cause A"]
    )
    issue_b = _make_issue(
        issue_id="b", name="Issue B", description="Desc B", root_causes=["Cause B"]
    )
    issues = [issue_a, issue_b]
    issue_trace_ids = {"a": ["t1"], "b": ["t2", "t3"]}
    rationale_map = {"t1": "r1", "t2": "r2", "t3": "r3"}

    with (
        patch(
            "mlflow.genai.discovery.pipeline._call_llm",
            return_value=_make_litellm_response("Annotated."),
        ),
        patch("mlflow.genai.discovery.pipeline.mlflow.log_issue") as mock_log_issue,
    ):
        _annotate_issue_traces(issues, issue_trace_ids, rationale_map, {}, "openai:/gpt-5-mini")

    assert mock_log_issue.call_count == 3
    issue_names = [c.kwargs["issue_name"] for c in mock_log_issue.call_args_list]
    assert issue_names.count("Issue A") == 1
    assert issue_names.count("Issue B") == 2


def test_annotate_traces_session_level_logs_on_first_trace():
    issues = [
        _make_issue(
            name="Slow responses [api]",
            description="Responses take too long",
            root_causes=["Complex queries"],
        ),
    ]
    issue_trace_ids = {issues[0].issue_id: ["trace-1", "trace-2", "trace-3"]}
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
            "mlflow.genai.discovery.pipeline._call_llm",
            return_value=_make_litellm_response("Session-level annotation."),
        ),
        patch("mlflow.genai.discovery.pipeline.mlflow.log_issue") as mock_log_issue,
    ):
        _annotate_issue_traces(
            issues,
            issue_trace_ids,
            rationale_map,
            {},
            "openai:/gpt-5-mini",
            trace_to_session=trace_to_session,
            session_first_trace=session_first_trace,
        )

    # 2 sessions -> 2 log_issue calls (not 3 per-trace)
    assert mock_log_issue.call_count == 2
    logged_trace_ids = {c.kwargs["trace_id"] for c in mock_log_issue.call_args_list}
    assert logged_trace_ids == {"trace-0", "trace-3"}
    for c in mock_log_issue.call_args_list:
        assert c.kwargs["issue_name"] == "Slow responses [api]"
        metadata = c.kwargs["metadata"]
        assert "mlflow.trace.session" in metadata
    session_ids = {
        c.kwargs["metadata"]["mlflow.trace.session"] for c in mock_log_issue.call_args_list
    }
    assert session_ids == {"session-A", "session-B"}


def test_recluster_merges_similar_singletons():
    analyses = [
        _ConversationAnalysis(
            full_rationale="API failure: tool error A",
            affected_trace_ids=["t1"],
        ),
        _ConversationAnalysis(
            full_rationale="API failure: tool error B",
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
            categories=[],
        ),
        _IdentifiedIssue(
            name="Issue B",
            description="Error B",
            root_cause="API failure",
            example_indices=[1],
            severity="high",
            categories=[],
        ),
    ]
    labels = {0: "[tool_call] API timeout", 1: "[tool_call] API timeout"}

    merged_issue = _IdentifiedIssue(
        name="Issue: API timeouts",
        description="Merged",
        root_cause="API failure",
        example_indices=[0, 1],
        severity="high",
        categories=[],
    )

    with (
        patch(
            "mlflow.genai.discovery.clustering.cluster_by_llm",
            return_value=[[0, 1]],
        ) as mock_cluster,
        patch(
            "mlflow.genai.discovery.clustering.summarize_cluster",
            return_value=merged_issue,
        ) as mock_summarize,
    ):
        result = recluster_singletons(
            singletons, labels, analyses, "openai:/gpt-5", 25, categories=[]
        )

    mock_cluster.assert_called_once()
    mock_summarize.assert_called_once()
    assert len(result) == 1
    assert result[0].example_indices == [0, 1]


def test_recluster_keeps_unmerged_singletons():
    analyses = [
        _ConversationAnalysis(
            full_rationale="error A",
            affected_trace_ids=["t1"],
        ),
        _ConversationAnalysis(
            full_rationale="error B",
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
            categories=[],
        ),
        _IdentifiedIssue(
            name="Issue B",
            description="B",
            root_cause="B",
            example_indices=[1],
            severity="high",
            categories=[],
        ),
    ]
    labels = {0: "[path_a] symptom a", 1: "[path_b] symptom b"}

    with patch(
        "mlflow.genai.discovery.clustering.cluster_by_llm",
        return_value=[[0], [1]],
    ) as mock_cluster:
        result = recluster_singletons(
            singletons, labels, analyses, "openai:/gpt-5", 25, categories=[]
        )

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
            categories=[],
        ),
    ]
    result = recluster_singletons(singletons, {0: "label"}, [], "m", 25, categories=[])
    assert len(result) == 1
    assert result[0].name == "Solo"


def test_recluster_low_severity_merge_keeps_originals():
    analyses = [
        _ConversationAnalysis(
            full_rationale="A",
            affected_trace_ids=["t1"],
        ),
        _ConversationAnalysis(
            full_rationale="B",
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
            categories=[],
        ),
        _IdentifiedIssue(
            name="B",
            description="B",
            root_cause="B",
            example_indices=[1],
            severity="low",
            categories=[],
        ),
    ]
    labels = {0: "label a", 1: "label b"}

    low_severity_merged = _IdentifiedIssue(
        name="Merged",
        description="M",
        root_cause="M",
        example_indices=[0, 1],
        severity="not_an_issue",
        categories=[],
    )

    with (
        patch(
            "mlflow.genai.discovery.clustering.cluster_by_llm",
            return_value=[[0, 1]],
        ),
        patch(
            "mlflow.genai.discovery.clustering.summarize_cluster",
            return_value=low_severity_merged,
        ),
    ):
        result = recluster_singletons(
            singletons, labels, analyses, "openai:/gpt-5", 25, categories=[]
        )

    assert len(result) == 2
    assert result[0].name == "A"
    assert result[1].name == "B"


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


def test_build_issue_discovery_scorer_returns_scorer_with_defaults():
    scorer = build_issue_discovery_scorer()
    assert scorer.name == DEFAULT_SCORER_NAME
    assert scorer.model == DEFAULT_MODEL


def test_build_issue_discovery_scorer_custom_model():
    scorer = build_issue_discovery_scorer(model="openai:/gpt-5")
    assert scorer.model == "openai:/gpt-5"


def test_build_satisfaction_instructions_categories_conversation():
    instructions = build_satisfaction_instructions(
        use_conversation=True, categories=["hallucination", "tool errors"]
    )
    assert "hallucination" in instructions
    assert "tool errors" in instructions
    assert "issue categories" in instructions


def test_build_satisfaction_instructions_categories_trace():
    instructions = build_satisfaction_instructions(use_conversation=False, categories=["latency"])
    assert "latency" in instructions
    assert "issue categories" in instructions


def test_discover_issues_with_custom_run_id(make_trace):
    traces = [make_trace()]

    with mlflow.start_run() as run:
        custom_run_id = run.info.run_id

    with (
        patch("mlflow.genai.discovery.pipeline.verify_scorer"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            return_value=_triage_eval(custom_run_id),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=_TriageResult([], {}, {}),
        ),
    ):
        result = discover_issues(traces=traces, run_id=custom_run_id)
        assert result.triage_run_id == custom_run_id
        mlflow.get_run(custom_run_id)


def test_discover_issues_tags_run_with_issue_detection_marker(make_trace):
    traces = [make_trace()]
    scorer = _TestScorer(name="test")

    with (
        patch("mlflow.genai.discovery.pipeline.verify_scorer"),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=_TriageResult([], {}, {}),
        ),
    ):
        result = discover_issues(traces=traces, scorers=[scorer])

    run = mlflow.get_run(result.triage_run_id)
    assert run.data.tags[MLFLOW_RUN_TYPE] == MLFLOW_RUN_TYPE_ISSUE_DETECTION


def test_discover_issues_returns_total_cost_usd_field(make_trace):
    from mlflow.entities.assessment import Feedback
    from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
    from mlflow.tracing.constant import AssessmentMetadataKey

    traces = [make_trace() for _ in range(5)]
    failing = traces[:2]
    rationale_map = {t.info.trace_id: "bad" for t in failing}

    with mlflow.start_run() as run:
        triage_run_id = run.info.run_id

    mock_traces_with_cost = []
    for trace in traces:
        mock_trace = MagicMock()
        mock_trace.info.trace_id = trace.info.trace_id
        mock_trace.info.assessments = [
            Feedback(
                name="test_scorer",
                value=True,
                source=AssessmentSource(source_type=AssessmentSourceType.CODE),
                trace_id=trace.info.trace_id,
                rationale="test",
                metadata={
                    AssessmentMetadataKey.SOURCE_RUN_ID: triage_run_id,
                    AssessmentMetadataKey.JUDGE_COST: 0.01,
                    AssessmentMetadataKey.JUDGE_INPUT_TOKENS: 100,
                    AssessmentMetadataKey.JUDGE_OUTPUT_TOKENS: 50,
                },
            )
        ]
        mock_traces_with_cost.append(mock_trace)

    cluster_summary_issue = _IdentifiedIssue(
        name="slow_response",
        description="Responses take too long",
        root_cause="Complex queries",
        example_indices=[0, 1],
        severity="high",
        categories=[],
    )

    mock_issue = _make_issue(name="slow_response", description="Responses take too long")

    with (
        patch("mlflow.genai.discovery.pipeline.verify_scorer"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            return_value=_triage_eval(triage_run_id),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.get_trace",
            side_effect=mock_traces_with_cost,
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=_TriageResult(failing, rationale_map, {}),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failure_labels",
            return_value=(["label1", "label2"], [0, 1]),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.cluster_by_llm",
            return_value=[[0, 1]],
        ),
        patch(
            "mlflow.genai.discovery.pipeline.summarize_cluster",
            return_value=cluster_summary_issue,
        ),
        patch("mlflow.tracing.client.TracingClient._create_issue", return_value=mock_issue),
        patch("mlflow.genai.discovery.pipeline._annotate_issue_traces"),
    ):
        result = discover_issues(traces=traces, experiment_id="0")

    assert hasattr(result, "total_cost_usd")
    assert result.total_cost_usd is not None
    assert isinstance(result.total_cost_usd, float)
    assert result.total_cost_usd > 0


def test_discover_issues_returns_none_cost_when_no_traces():
    result = discover_issues(traces=[])
    assert result.total_cost_usd == 0.0
    assert result.total_traces_analyzed == 0


def test_discover_issues_returns_cost_when_all_pass(make_trace):
    from mlflow.entities.assessment import Feedback
    from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
    from mlflow.tracing.constant import AssessmentMetadataKey

    traces = [make_trace() for _ in range(5)]

    with mlflow.start_run() as run:
        triage_run_id = run.info.run_id

    mock_traces_with_cost = []
    for trace in traces:
        mock_trace = MagicMock()
        mock_trace.info.trace_id = trace.info.trace_id
        mock_trace.info.assessments = [
            Feedback(
                name="test_scorer",
                value=True,
                source=AssessmentSource(source_type=AssessmentSourceType.CODE),
                trace_id=trace.info.trace_id,
                rationale="test",
                metadata={
                    AssessmentMetadataKey.SOURCE_RUN_ID: triage_run_id,
                    AssessmentMetadataKey.JUDGE_COST: 0.005,
                    AssessmentMetadataKey.JUDGE_INPUT_TOKENS: 50,
                    AssessmentMetadataKey.JUDGE_OUTPUT_TOKENS: 25,
                },
            )
        ]
        mock_traces_with_cost.append(mock_trace)

    with (
        patch("mlflow.genai.discovery.pipeline.verify_scorer"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            return_value=_triage_eval(triage_run_id),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.get_trace",
            side_effect=mock_traces_with_cost,
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=_TriageResult([], {}, {}),
        ),
    ):
        result = discover_issues(traces=traces)

    assert result.total_cost_usd is not None
    assert isinstance(result.total_cost_usd, float)
    assert result.total_cost_usd > 0


def test_discover_issues_filters_invalid_categories(make_trace):
    traces = [make_trace() for _ in range(5)]
    failing = traces[:2]
    rationale_map = {t.info.trace_id: "bad" for t in failing}

    # Issue returned from summarize_cluster has both valid and invalid categories
    cluster_summary_issue = _IdentifiedIssue(
        name="issue_with_categories",
        description="Issue with mixed categories",
        root_cause="Root cause",
        example_indices=[0, 1],
        severity="high",
        categories=["hallucination", "invalid_category", "tool_error", "another_invalid"],
    )

    mock_issue = _make_issue(
        name="issue_with_categories", description="Issue with mixed categories"
    )

    with mlflow.start_run() as run:
        triage_run_id = run.info.run_id

    with (
        patch("mlflow.genai.discovery.pipeline.verify_scorer"),
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            return_value=_triage_eval(triage_run_id),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=_TriageResult(failing, rationale_map, {}),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failure_labels",
            return_value=(["label1", "label2"], [0, 1]),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.cluster_by_llm",
            return_value=[[0, 1]],
        ),
        patch(
            "mlflow.genai.discovery.pipeline.summarize_cluster",
            return_value=cluster_summary_issue,
        ) as mock_summarize,
        patch("mlflow.tracing.client.TracingClient._create_issue", return_value=mock_issue),
        patch("mlflow.genai.discovery.pipeline._annotate_issue_traces"),
    ):
        discover_issues(
            traces=traces,
            experiment_id="0",
            categories=["hallucination", "tool_error"],
        )

    mock_summarize.assert_called_once()
    call_kwargs = mock_summarize.call_args.kwargs
    assert call_kwargs["categories"] == ["hallucination", "tool_error"]


def test_discover_issues_with_mixed_session_traces(make_trace):

    traces = [
        make_trace(session_id="session-1"),
        make_trace(session_id="session-1"),
        make_trace(session_id=None),
        make_trace(session_id=None),
    ]

    with (
        patch("mlflow.genai.discovery.pipeline.verify_scorer") as mock_verify,
        patch(
            "mlflow.genai.discovery.pipeline.mlflow.genai.evaluate",
            return_value=_triage_eval(),
        ),
        patch(
            "mlflow.genai.discovery.pipeline.extract_failing_traces",
            return_value=_TriageResult([], {}, {}),
        ),
    ):
        result = discover_issues(
            traces=traces,
        )

    assert result is not None
    assert result.total_traces_analyzed == 4

    mock_verify.assert_called_once()
    call_kwargs = mock_verify.call_args.kwargs
    assert call_kwargs["session"] is not None
    assert all(get_session_id(t) is not None for t in call_kwargs["session"])


def _make_dedup_response(
    groups: list[list[int]],
    names: list[str] | None = None,
    descriptions: list[str] | None = None,
    root_causes: list[str] | None = None,
):
    group_objects = [
        {
            "indices": indices,
            "name": names[i] if names and i < len(names) else "Issue: Merged issue",
            "description": descriptions[i]
            if descriptions and i < len(descriptions)
            else "Merged description",
            "root_cause": root_causes[i]
            if root_causes and i < len(root_causes)
            else "Merged root cause",
        }
        for i, indices in enumerate(groups)
    ]
    return _make_litellm_response(json.dumps({"groups": group_objects}))


def test_dedup_issues_empty():
    assert _dedup_issues([]) == []


def test_dedup_issues_single():
    issue = create_identified_issue()
    result = _dedup_issues([issue])
    assert result == [issue]


def test_dedup_issues_similar_issues_merged():
    issue1 = create_identified_issue(
        example_indices=[0], severity="low", categories=["correctness"]
    )
    issue2 = create_identified_issue(example_indices=[1], severity="high", categories=["latency"])
    with patch(
        "mlflow.genai.discovery.pipeline._call_llm",
        return_value=_make_dedup_response([[0, 1]]),
    ):
        result = _dedup_issues([issue1, issue2])
    assert len(result) == 1
    assert set(result[0].example_indices) == {0, 1}
    assert result[0].severity == "high"
    assert result[0].categories == ["correctness", "latency"]
    # LLM-generated consolidated fields are applied
    assert result[0].name == "Issue: Merged issue"
    assert result[0].description == "Merged description"
    assert result[0].root_cause == "Merged root cause"


def test_dedup_issues_consolidated_fields_from_llm():
    issue1 = create_identified_issue(name="Issue: Foo", description="desc 1", root_cause="rc 1")
    issue2 = create_identified_issue(name="Issue: Bar", description="desc 2", root_cause="rc 2")
    with patch(
        "mlflow.genai.discovery.pipeline._call_llm",
        return_value=_make_dedup_response(
            [[0, 1]],
            names=["Issue: Consolidated name"],
            descriptions=["Unified description"],
            root_causes=["Common root cause"],
        ),
    ):
        result = _dedup_issues([issue1, issue2])
    assert len(result) == 1
    assert result[0].name == "Issue: Consolidated name"
    assert result[0].description == "Unified description"
    assert result[0].root_cause == "Common root cause"


def test_dedup_issues_dissimilar_issues_not_merged():
    issue1 = create_identified_issue(example_indices=[0])
    issue2 = create_identified_issue(example_indices=[1])
    with patch(
        "mlflow.genai.discovery.pipeline._call_llm",
        return_value=_make_dedup_response([]),
    ):
        result = _dedup_issues([issue1, issue2])
    assert len(result) == 2


def test_dedup_issues_merges_example_indices_deduped():
    issue1 = create_identified_issue(example_indices=[0, 1])
    issue2 = create_identified_issue(example_indices=[1, 2])
    with patch(
        "mlflow.genai.discovery.pipeline._call_llm",
        return_value=_make_dedup_response([[0, 1]]),
    ):
        result = _dedup_issues([issue1, issue2])
    assert len(result) == 1
    assert set(result[0].example_indices) == {0, 1, 2}


def test_dedup_issues_categories_preserve_order_and_uniqueness():
    issue1 = create_identified_issue(categories=["correctness", "latency"])
    issue2 = create_identified_issue(categories=["latency", "hallucination"])
    with patch(
        "mlflow.genai.discovery.pipeline._call_llm",
        return_value=_make_dedup_response([[0, 1]]),
    ):
        result = _dedup_issues([issue1, issue2])
    assert len(result) == 1
    assert result[0].categories == ["correctness", "latency", "hallucination"]


def test_dedup_issues_overlapping_groups_merged_transitively():
    issue0 = create_identified_issue(example_indices=[0], severity="low")
    issue1 = create_identified_issue(example_indices=[1], severity="medium")
    issue2 = create_identified_issue(example_indices=[2], severity="high")
    with patch(
        "mlflow.genai.discovery.pipeline._call_llm",
        return_value=_make_dedup_response([[0, 1], [1, 2]]),
    ):
        result = _dedup_issues([issue0, issue1, issue2])
    assert len(result) == 1
    assert set(result[0].example_indices) == {0, 1, 2}
    assert result[0].severity == "high"


def test_dedup_issues_llm_failure_returns_original():
    issues = [create_identified_issue(example_indices=[i]) for i in range(3)]
    with patch("mlflow.genai.discovery.pipeline._call_llm", side_effect=Exception("API error")):
        result = _dedup_issues(issues)
    assert result == issues
