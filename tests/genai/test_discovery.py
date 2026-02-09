from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mlflow.entities import Trace, TraceData, TraceInfo
from mlflow.entities.span import Span
from mlflow.entities.span_status import SpanStatus, SpanStatusCode
from mlflow.genai.discovery import (
    Issue,
    _build_default_satisfaction_scorer,
    _build_summary,
    _compute_frequencies,
    _extract_failing_traces,
    _format_trace_for_clustering,
    _IdentifiedIssue,
    _IssueClusteringResult,
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


# ---- _format_trace_for_clustering ----


def test_format_trace_for_clustering():
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
    text = _format_trace_for_clustering(0, trace, "Response was incomplete")

    assert "[0] trace_id=t-1" in text
    assert "Hello" in text
    assert "Hi there" in text
    assert "tool_call" in text
    assert "Response was incomplete" in text


def test_format_trace_truncates_previews():
    long_text = "x" * 1000
    trace = _make_trace(request_preview=long_text, response_preview=long_text)
    text = _format_trace_for_clustering(0, trace, "")
    # Each preview should be truncated to 500 chars
    assert text.count("x") <= 1000


def test_format_trace_none_previews():
    trace = _make_trace()
    trace.info.request_preview = None
    trace.info.response_preview = None
    text = _format_trace_for_clustering(0, trace, "rationale")
    assert "Input: \n" in text
    assert "Output: \n" in text


# ---- _extract_failing_traces ----


def test_extract_failing_traces():
    traces = [_make_trace(trace_id=f"t-{i}") for i in range(3)]
    df = pd.DataFrame(
        {
            "satisfaction/value": [True, False, False],
            "satisfaction/rationale": ["good", "bad response", "incomplete"],
            "trace": traces,
        }
    )
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=df)

    failing, rationales = _extract_failing_traces(eval_result, "satisfaction")

    assert len(failing) == 2
    assert failing[0].info.trace_id == "t-1"
    assert failing[1].info.trace_id == "t-2"
    assert rationales["t-1"] == "bad response"
    assert rationales["t-2"] == "incomplete"


def test_extract_failing_traces_none_result_df():
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=None)
    failing, rationales = _extract_failing_traces(eval_result, "satisfaction")
    assert failing == []
    assert rationales == {}


def test_extract_failing_traces_missing_column():
    df = pd.DataFrame({"other/value": [True]})
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=df)
    failing, rationales = _extract_failing_traces(eval_result, "satisfaction")
    assert failing == []


def test_extract_failing_traces_no_failures():
    traces = [_make_trace(trace_id="t-0")]
    df = pd.DataFrame(
        {
            "satisfaction/value": [True],
            "satisfaction/rationale": ["good"],
            "trace": traces,
        }
    )
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=df)
    failing, rationales = _extract_failing_traces(eval_result, "satisfaction")
    assert failing == []
    assert rationales == {}


# ---- _compute_frequencies ----


def test_compute_frequencies():
    df = pd.DataFrame(
        {
            "issue_a/value": [True, True, True, False, False, False, False, False, False, False],
            "issue_a/rationale": ["r1", "r2", "r3"] + ["ok"] * 7,
            "issue_b/value": [True, False, False, False, False, False, False, False, False, False],
            "issue_b/rationale": ["r1"] + ["ok"] * 9,
        }
    )
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=df)

    freqs, examples = _compute_frequencies(eval_result, ["issue_a", "issue_b"])

    assert freqs["issue_a"] == pytest.approx(0.3)
    assert freqs["issue_b"] == pytest.approx(0.1)
    assert len(examples["issue_a"]) == 3
    assert len(examples["issue_b"]) == 1


def test_compute_frequencies_none_df():
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=None)
    freqs, examples = _compute_frequencies(eval_result, ["issue_a"])
    assert freqs == {}
    assert examples == {}


# ---- _build_summary ----


def test_build_summary_no_issues():
    summary = _build_summary([], 50)
    assert "50 traces" in summary
    assert "No issues found" in summary


def test_build_summary_with_issues():
    issues = [
        Issue(
            name="tool_failure",
            description="Tool calls fail intermittently",
            root_cause="API timeout",
            example_trace_ids=["t-0"],
            scorer=MagicMock(),
            frequency=0.3,
        ),
    ]
    summary = _build_summary(issues, 100)
    assert "tool_failure" in summary
    assert "30%" in summary
    assert "API timeout" in summary


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


# ---- discover_issues (integration) ----


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


def test_discover_issues_all_traces_pass():
    traces = [_make_trace(trace_id=f"t-{i}") for i in range(5)]
    result_df = pd.DataFrame(
        {
            "satisfaction/value": [True] * 5,
            "satisfaction/rationale": ["good"] * 5,
            "trace": traces,
        }
    )
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

    triage_df = pd.DataFrame(
        {
            "satisfaction/value": [False] * 3 + [True] * 7,
            "satisfaction/rationale": ["bad"] * 3 + ["good"] * 7,
            "trace": traces,
        }
    )
    triage_eval = EvaluationResult(run_id="run-triage", metrics={}, result_df=triage_df)

    clustering_result = _IssueClusteringResult(
        issues=[
            _IdentifiedIssue(
                name="slow_response",
                description="Responses take too long",
                root_cause="Complex queries",
                detection_instructions="Check the {{ trace }} execution duration",
                example_indices=[0, 1],
            ),
        ]
    )

    validation_df = pd.DataFrame(
        {
            "slow_response/value": [True] * 3 + [False] * 7,
            "slow_response/rationale": ["slow"] * 3 + ["fast"] * 7,
        }
    )
    validation_eval = EvaluationResult(run_id="run-validate", metrics={}, result_df=validation_df)

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
    ):
        result = discover_issues(sample_size=10)

    assert len(result.issues) == 1
    assert result.issues[0].name == "slow_response"
    assert result.issues[0].frequency == pytest.approx(0.3)
    assert result.issues[0].example_trace_ids == ["t-0", "t-1"]
    assert result.triage_evaluation is triage_eval
    assert result.validation_evaluation is validation_eval


def test_discover_issues_low_frequency_issues_discarded():
    traces = [_make_trace(trace_id=f"t-{i}") for i in range(5)]

    triage_df = pd.DataFrame(
        {
            "satisfaction/value": [False] * 2 + [True] * 3,
            "satisfaction/rationale": ["bad"] * 2 + ["good"] * 3,
            "trace": traces,
        }
    )
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

    validation_df = pd.DataFrame(
        {
            "rare_issue/value": [False] * 200,
            "rare_issue/rationale": ["ok"] * 200,
        }
    )
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


def test_discover_issues_passes_filter_and_model_id():
    with (
        patch("mlflow.genai.discovery._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.mlflow.search_traces", return_value=[]) as mock_search,
    ):
        discover_issues(filter_string="tag.env = 'prod'", model_id="m-abc")

    call_kwargs = mock_search.call_args[1]
    assert call_kwargs["filter_string"] == "tag.env = 'prod'"
    assert call_kwargs["model_id"] == "m-abc"


def test_discover_issues_custom_satisfaction_scorer():
    custom_scorer = MagicMock()
    custom_scorer.name = "custom"
    traces = [_make_trace()]

    result_df = pd.DataFrame(
        {
            "custom/value": [True],
            "custom/rationale": ["good"],
            "trace": traces,
        }
    )
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
