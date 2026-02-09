import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mlflow.entities import Trace, TraceData, TraceInfo
from mlflow.entities.span import Span
from mlflow.entities.span_status import SpanStatus, SpanStatusCode
from mlflow.genai.discovery import (
    Issue,
    _build_default_satisfaction_scorer,
    _build_trace_summary,
    _check_scorer_errors,
    _extract_scorer_error,
    _format_summaries_for_clustering,
    _IdentifiedIssue,
    _IssueClusteringResult,
    _phase1_triage,
    _phase2_cluster,
    _phase3_validate,
    _phase4_summarize,
    _SummaryResult,
    _TraceSummary,
    discover_issues,
)
from mlflow.genai.evaluation.entities import EvaluationResult


def _make_mock_span(name="test_span", status_code=SpanStatusCode.OK):
    span = MagicMock(spec=Span)
    span.name = name
    span.status = SpanStatus(status_code=status_code)
    return span


def _make_trace(
    trace_id="trace-1",
    request_preview="What is MLflow?",
    response_preview="MLflow is an ML platform.",
    execution_duration=500,
    spans=None,
):
    info = MagicMock(spec=TraceInfo)
    info.trace_id = trace_id
    info.request_preview = request_preview
    info.response_preview = response_preview
    info.execution_duration = execution_duration

    data = MagicMock(spec=TraceData)
    data.spans = spans or [_make_mock_span()]

    trace = MagicMock(spec=Trace)
    trace.info = info
    trace.data = data
    return trace


# ---- _build_trace_summary ----


def test_build_trace_summary_basic():
    trace = _make_trace(
        trace_id="t-1",
        request_preview="Hello",
        response_preview="Hi there",
        execution_duration=200,
        spans=[
            _make_mock_span("llm_call"),
            _make_mock_span("tool_call", SpanStatusCode.ERROR),
        ],
    )

    summary = _build_trace_summary(0, trace, "Response was incomplete")

    assert summary.index == 0
    assert summary.trace_id == "t-1"
    assert summary.inputs_preview == "Hello"
    assert summary.outputs_preview == "Hi there"
    assert summary.span_names == ["llm_call", "tool_call"]
    assert summary.error_spans == ["tool_call"]
    assert summary.satisfaction_rationale == "Response was incomplete"
    assert summary.execution_duration_ms == 200


def test_build_trace_summary_truncates_previews():
    long_text = "x" * 1000
    trace = _make_trace(request_preview=long_text, response_preview=long_text)

    summary = _build_trace_summary(0, trace, "")

    assert len(summary.inputs_preview) == 500
    assert len(summary.outputs_preview) == 500


def test_build_trace_summary_none_previews():
    trace = _make_trace()
    trace.info.request_preview = None
    trace.info.response_preview = None

    summary = _build_trace_summary(0, trace, "rationale")

    assert summary.inputs_preview == ""
    assert summary.outputs_preview == ""


# ---- _format_summaries_for_clustering ----


def test_format_summaries_for_clustering():
    summaries = [
        _TraceSummary(
            index=0,
            trace_id="t-1",
            inputs_preview="input1",
            outputs_preview="output1",
            span_names=["span1"],
            error_spans=[],
            satisfaction_rationale="bad",
            execution_duration_ms=100,
        ),
        _TraceSummary(
            index=1,
            trace_id="t-2",
            inputs_preview="input2",
            outputs_preview="output2",
            span_names=["span2", "span3"],
            error_spans=["span3"],
            satisfaction_rationale="also bad",
            execution_duration_ms=200,
        ),
    ]

    text = _format_summaries_for_clustering(summaries)

    assert "t-1" in text
    assert "t-2" in text
    assert "span3" in text
    assert "also bad" in text


# ---- _check_scorer_errors / _extract_scorer_error ----


def test_check_scorer_errors_all_none_raises():
    df = pd.DataFrame({"scorer/value": [None, None, None]})
    with pytest.raises(Exception, match="Scorer 'scorer' failed on all traces"):
        _check_scorer_errors(df, "scorer")


def test_check_scorer_errors_valid_values_passes():
    df = pd.DataFrame({"scorer/value": [True, False, True]})
    _check_scorer_errors(df, "scorer")


def test_check_scorer_errors_missing_column_passes():
    df = pd.DataFrame({"other_col": [1, 2, 3]})
    _check_scorer_errors(df, "scorer")


def test_check_scorer_errors_mixed_values_passes():
    df = pd.DataFrame({"scorer/value": [True, None, False]})
    _check_scorer_errors(df, "scorer")


def test_check_scorer_errors_extracts_error_from_assessments():
    assessments = json.dumps([
        {
            "assessment_name": "scorer",
            "feedback": {
                "error": {
                    "error_code": "SCORER_ERROR",
                    "error_message": "litellm not installed",
                }
            },
        }
    ])
    df = pd.DataFrame({
        "scorer/value": [None],
        "assessments": [assessments],
    })
    with pytest.raises(Exception, match="litellm not installed"):
        _check_scorer_errors(df, "scorer")


def test_extract_scorer_error_no_assessments_column():
    df = pd.DataFrame({"other": [1]})
    assert _extract_scorer_error(df, "scorer") == "No error details available."


def test_extract_scorer_error_from_dict_assessments():
    assessments = [
        {
            "assessment_name": "scorer",
            "feedback": {
                "error": {"error_message": "Connection refused"},
            },
        }
    ]
    df = pd.DataFrame({"assessments": [assessments]})
    assert _extract_scorer_error(df, "scorer") == "Connection refused"


# ---- _phase1_triage ----


def test_phase1_triage_identifies_failing_traces():
    traces = [_make_trace(trace_id=f"t-{i}") for i in range(3)]
    scorer = MagicMock()
    scorer.name = "satisfaction"

    result_df = pd.DataFrame({
        "satisfaction/value": [True, False, False],
        "satisfaction/rationale": ["good", "bad response", "incomplete"],
        "trace": traces,
    })
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=result_df)

    with patch("mlflow.genai.discovery.mlflow.genai.evaluate", return_value=eval_result):
        eval_r, failing, rationale_map = _phase1_triage(traces, scorer, None)

    assert eval_r is eval_result
    assert len(failing) == 2
    assert failing[0].info.trace_id == "t-1"
    assert failing[1].info.trace_id == "t-2"
    assert rationale_map["t-1"] == "bad response"
    assert rationale_map["t-2"] == "incomplete"


def test_phase1_triage_no_failures():
    traces = [_make_trace(trace_id="t-0")]
    scorer = MagicMock()
    scorer.name = "satisfaction"

    result_df = pd.DataFrame({
        "satisfaction/value": [True],
        "satisfaction/rationale": ["all good"],
        "trace": traces,
    })
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=result_df)

    with patch("mlflow.genai.discovery.mlflow.genai.evaluate", return_value=eval_result):
        _, failing, rationale_map = _phase1_triage(traces, scorer, None)

    assert failing == []
    assert rationale_map == {}


def test_phase1_triage_raises_on_all_scorer_errors():
    traces = [_make_trace(trace_id=f"t-{i}") for i in range(3)]
    scorer = MagicMock()
    scorer.name = "satisfaction"

    result_df = pd.DataFrame({
        "satisfaction/value": [None, None, None],
        "trace": traces,
    })
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=result_df)

    with (
        patch("mlflow.genai.discovery.mlflow.genai.evaluate", return_value=eval_result),
        pytest.raises(Exception, match="Scorer 'satisfaction' failed on all traces"),
    ):
        _phase1_triage(traces, scorer, None)


# ---- _phase2_cluster ----


def test_phase2_cluster_clusters_issues():
    failing_traces = [_make_trace(trace_id=f"t-{i}") for i in range(3)]
    rationale_map = {f"t-{i}": f"reason {i}" for i in range(3)}

    clustering_result = _IssueClusteringResult(
        issues=[
            _IdentifiedIssue(
                name="tool_failure",
                description="Tool calls fail intermittently",
                root_cause="API timeout",
                detection_instructions="Check if any tool spans have error status",
                example_indices=[0, 1],
            ),
            _IdentifiedIssue(
                name="hallucination",
                description="Model generates incorrect facts",
                root_cause="Insufficient context",
                detection_instructions="Check if output contradicts input context",
                example_indices=[2],
            ),
        ]
    )

    with patch(
        "mlflow.genai.discovery.get_chat_completions_with_structured_output",
        return_value=clustering_result,
    ):
        issues = _phase2_cluster(failing_traces, rationale_map, "openai:/gpt-4", 10)

    assert len(issues) == 2
    assert issues[0].name == "tool_failure"
    assert issues[1].name == "hallucination"


def test_phase2_cluster_empty_traces():
    issues = _phase2_cluster([], {}, "openai:/gpt-4", 10)
    assert issues == []


def test_phase2_cluster_respects_max_issues():
    failing_traces = [_make_trace(trace_id="t-0")]
    rationale_map = {"t-0": "reason"}

    clustering_result = _IssueClusteringResult(
        issues=[
            _IdentifiedIssue(
                name=f"issue_{i}",
                description=f"Issue {i}",
                root_cause=f"Cause {i}",
                detection_instructions=f"Detect issue {i}",
                example_indices=[0],
            )
            for i in range(5)
        ]
    )

    with patch(
        "mlflow.genai.discovery.get_chat_completions_with_structured_output",
        return_value=clustering_result,
    ):
        issues = _phase2_cluster(failing_traces, rationale_map, "openai:/gpt-4", 2)

    assert len(issues) == 2


# ---- _phase3_validate ----


def test_phase3_validate_computes_frequencies():
    validation_traces = [_make_trace(trace_id=f"t-{i}") for i in range(10)]
    identified_issues = [
        _IdentifiedIssue(
            name="tool_failure",
            description="Tool calls fail",
            root_cause="API timeout",
            detection_instructions="Check for tool errors",
            example_indices=[0],
        ),
    ]

    result_df = pd.DataFrame({
        "tool_failure/value": ([True, True, True] + [False] * 7),
        "tool_failure/rationale": ["err"] * 3 + ["ok"] * 7,
    })
    eval_result = EvaluationResult(run_id="run-2", metrics={}, result_df=result_df)

    with (
        patch("mlflow.genai.discovery.mlflow.genai.evaluate", return_value=eval_result),
        patch("mlflow.genai.discovery.make_judge", return_value=MagicMock()),
    ):
        eval_r, freq_map, rationale_map = _phase3_validate(
            validation_traces, identified_issues, "openai:/gpt-4", None
        )

    assert eval_r is eval_result
    assert freq_map["tool_failure"] == pytest.approx(0.3)
    assert len(rationale_map["tool_failure"]) == 3


def test_phase3_validate_empty_issues():
    eval_r, freq_map, rationale_map = _phase3_validate(
        [_make_trace()], [], "openai:/gpt-4", None
    )
    assert eval_r is None
    assert freq_map == {}


def test_phase3_validate_empty_traces():
    issues = [
        _IdentifiedIssue(
            name="issue1",
            description="desc",
            root_cause="cause",
            detection_instructions="detect",
            example_indices=[],
        ),
    ]
    eval_r, freq_map, rationale_map = _phase3_validate([], issues, "openai:/gpt-4", None)
    assert eval_r is None


# ---- _phase4_summarize ----


def test_phase4_summarize_generates_summary():
    issues = [
        Issue(
            name="tool_failure",
            description="Tool calls fail",
            root_cause="API timeout",
            example_trace_ids=["t-0"],
            scorer=MagicMock(),
            frequency=0.3,
        ),
    ]

    summary_result = _SummaryResult(summary="## Summary\n\nFound 1 issue.")

    with patch(
        "mlflow.genai.discovery.get_chat_completions_with_structured_output",
        return_value=summary_result,
    ):
        summary = _phase4_summarize(issues, 100, "openai:/gpt-4")

    assert "## Summary" in summary


def test_phase4_summarize_no_issues_returns_default():
    summary = _phase4_summarize([], 50, "openai:/gpt-4")
    assert "No significant issues" in summary
    assert "50" in summary


# ---- discover_issues ----


def test_discover_issues_no_experiment():
    with (
        patch("mlflow.genai.discovery._get_experiment_id", return_value=None),
        pytest.raises(Exception, match="No experiment specified"),
    ):
        discover_issues()


def test_discover_issues_empty_experiment():
    with (
        patch("mlflow.genai.discovery._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.mlflow.search_traces", return_value=[]),
    ):
        result = discover_issues()

    assert result.issues == []
    assert result.total_traces_analyzed == 0
    assert "No traces found" in result.summary


def test_discover_issues_all_traces_pass():
    traces = [_make_trace(trace_id=f"t-{i}") for i in range(5)]
    result_df = pd.DataFrame({
        "satisfaction/value": [True] * 5,
        "satisfaction/rationale": ["good"] * 5,
        "trace": traces,
    })
    triage_eval = EvaluationResult(run_id="run-1", metrics={}, result_df=result_df)

    with (
        patch("mlflow.genai.discovery._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.mlflow.search_traces", return_value=traces),
        patch("mlflow.genai.discovery.mlflow.genai.evaluate", return_value=triage_eval),
    ):
        result = discover_issues()

    assert result.issues == []
    assert "no issues found" in result.summary.lower()


def test_discover_issues_full_pipeline():
    traces = [_make_trace(trace_id=f"t-{i}") for i in range(10)]

    # Phase 1: triage result
    triage_df = pd.DataFrame({
        "satisfaction/value": [False] * 3 + [True] * 7,
        "satisfaction/rationale": ["bad"] * 3 + ["good"] * 7,
        "trace": traces,
    })
    triage_eval = EvaluationResult(run_id="run-triage", metrics={}, result_df=triage_df)

    # Phase 2: clustering result
    clustering_result = _IssueClusteringResult(
        issues=[
            _IdentifiedIssue(
                name="slow_response",
                description="Responses take too long",
                root_cause="Complex queries",
                detection_instructions=(
                    "Check the {{ trace }} execution duration for slow responses"
                ),
                example_indices=[0, 1],
            ),
        ]
    )

    # Phase 3: validation result
    validation_df = pd.DataFrame({
        "slow_response/value": [True] * 3 + [False] * 7,
        "slow_response/rationale": ["slow"] * 3 + ["fast"] * 7,
    })
    validation_eval = EvaluationResult(
        run_id="run-validate", metrics={}, result_df=validation_df
    )

    # Phase 4: summary
    summary_result = _SummaryResult(summary="## Found 1 issue\n\nslow_response: 30%")

    with (
        patch("mlflow.genai.discovery._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.mlflow.search_traces", return_value=traces),
        patch(
            "mlflow.genai.discovery.mlflow.genai.evaluate",
            side_effect=[triage_eval, validation_eval],
        ),
        patch(
            "mlflow.genai.discovery.get_chat_completions_with_structured_output",
            side_effect=[clustering_result, summary_result],
        ),
        patch("mlflow.genai.discovery._get_total_trace_count", return_value=50),
    ):
        result = discover_issues(sample_size=10)

    assert len(result.issues) == 1
    assert result.issues[0].name == "slow_response"
    assert result.issues[0].frequency == pytest.approx(0.3)
    assert result.triage_evaluation is triage_eval
    assert result.validation_evaluation is validation_eval
    assert "Found 1 issue" in result.summary


def test_discover_issues_low_frequency_issues_discarded():
    traces = [_make_trace(trace_id=f"t-{i}") for i in range(5)]

    triage_df = pd.DataFrame({
        "satisfaction/value": [False] * 2 + [True] * 3,
        "satisfaction/rationale": ["bad"] * 2 + ["good"] * 3,
        "trace": traces,
    })
    triage_eval = EvaluationResult(run_id="run-1", metrics={}, result_df=triage_df)

    clustering_result = _IssueClusteringResult(
        issues=[
            _IdentifiedIssue(
                name="rare_issue",
                description="Happens very rarely",
                root_cause="Unknown",
                detection_instructions="Check the {{ trace }} for rare errors",
                example_indices=[0],
            ),
        ]
    )

    # Frequency below 1% threshold (0 True values)
    validation_df = pd.DataFrame({
        "rare_issue/value": [False] * 200,
        "rare_issue/rationale": ["ok"] * 200,
    })
    validation_eval = EvaluationResult(run_id="run-2", metrics={}, result_df=validation_df)

    with (
        patch("mlflow.genai.discovery._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.mlflow.search_traces", return_value=traces),
        patch(
            "mlflow.genai.discovery.mlflow.genai.evaluate",
            side_effect=[triage_eval, validation_eval],
        ),
        patch(
            "mlflow.genai.discovery.get_chat_completions_with_structured_output",
            return_value=clustering_result,
        ),
        patch("mlflow.genai.discovery._get_total_trace_count", return_value=200),
    ):
        result = discover_issues(sample_size=5)

    assert len(result.issues) == 0


def test_discover_issues_explicit_experiment_id():
    with patch(
        "mlflow.genai.discovery.mlflow.search_traces",
        return_value=[],
    ) as mock_search:
        discover_issues(experiment_id="exp-42")

    mock_search.assert_called_once()
    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["locations"] == ["exp-42"]


def test_discover_issues_custom_satisfaction_scorer():
    custom_scorer = MagicMock()
    traces = [_make_trace()]

    result_df = pd.DataFrame({
        "satisfaction/value": [True],
        "satisfaction/rationale": ["good"],
        "trace": traces,
    })
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=result_df)

    with (
        patch("mlflow.genai.discovery._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.mlflow.search_traces", return_value=traces),
        patch(
            "mlflow.genai.discovery.mlflow.genai.evaluate",
            return_value=eval_result,
        ) as mock_eval,
    ):
        discover_issues(satisfaction_scorer=custom_scorer)

    mock_eval.assert_called_once()
    call_kwargs = mock_eval.call_args[1]
    assert call_kwargs["scorers"] == [custom_scorer]


def test_discover_issues_passes_filter_string():
    with (
        patch("mlflow.genai.discovery._get_experiment_id", return_value="exp-1"),
        patch(
            "mlflow.genai.discovery.mlflow.search_traces",
            return_value=[],
        ) as mock_search,
    ):
        discover_issues(filter_string="tag.env = 'prod'")

    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["filter_string"] == "tag.env = 'prod'"


def test_discover_issues_passes_model_id():
    with (
        patch("mlflow.genai.discovery._get_experiment_id", return_value="exp-1"),
        patch(
            "mlflow.genai.discovery.mlflow.search_traces",
            return_value=[],
        ) as mock_search,
    ):
        discover_issues(model_id="m-abc123")

    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["model_id"] == "m-abc123"


# ---- _build_default_satisfaction_scorer ----


def test_build_default_satisfaction_scorer():
    with patch("mlflow.genai.discovery.make_judge", return_value=MagicMock()) as mock_make_judge:
        _build_default_satisfaction_scorer("openai:/gpt-4")

    mock_make_judge.assert_called_once()
    call_kwargs = mock_make_judge.call_args[1]
    assert call_kwargs["name"] == "satisfaction"
    assert call_kwargs["feedback_value_type"] is bool
    assert call_kwargs["model"] == "openai:/gpt-4"
    assert "{{ conversation }}" in call_kwargs["instructions"]
