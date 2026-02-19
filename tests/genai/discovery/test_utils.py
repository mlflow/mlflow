from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatusCode
from mlflow.genai.discovery.constants import _DEFAULT_SCORER_NAME
from mlflow.genai.discovery.entities import Issue
from mlflow.genai.discovery.schemas import (
    _BatchTraceAnalysisResult,
    _IdentifiedIssue,
    _ScorerInstructionsResult,
    _ScorerSpec,
    _TraceAnalysis,
)
from mlflow.genai.discovery.utils import (
    _build_default_satisfaction_scorer,
    _build_enriched_trace_summary,
    _build_span_tree,
    _build_summary,
    _compute_frequencies,
    _extract_failing_traces,
    _format_analysis_for_clustering,
    _generate_scorer_specs,
    _get_existing_score,
    _has_session_ids,
    _partition_by_existing_scores,
    _run_deep_analysis,
)
from mlflow.genai.evaluation.entities import EvaluationResult

# ---- _build_span_tree ----


def test_build_span_tree_simple(make_mock_span):
    root = make_mock_span(
        name="agent",
        span_id="s1",
        parent_id=None,
        span_type="AGENT",
        start_time_ns=0,
        end_time_ns=1_500_000_000,
    )
    child = make_mock_span(
        name="llm_call",
        span_id="s2",
        parent_id="s1",
        span_type="LLM",
        start_time_ns=100_000_000,
        end_time_ns=900_000_000,
        model_name="gpt-4",
    )
    tree = _build_span_tree([root, child])

    assert "agent (AGENT, OK, 1500ms)" in tree
    assert "llm_call (LLM, OK, 800ms, model=gpt-4)" in tree


def test_build_span_tree_error_with_exception(make_mock_span):
    exc_event = MagicMock(spec=SpanEvent)
    exc_event.name = "exception"
    exc_event.attributes = {
        "exception.type": "ConnectionTimeout",
        "exception.message": "API unreachable",
    }
    span = make_mock_span(
        name="weather_tool",
        span_id="s1",
        span_type="TOOL",
        status_code=SpanStatusCode.ERROR,
        status_description="Connection failed",
        events=[exc_event],
    )
    tree = _build_span_tree([span])

    assert "TOOL, ERROR" in tree
    assert "ERROR: Connection failed" in tree
    assert "EXCEPTION: ConnectionTimeout: API unreachable" in tree


def test_build_span_tree_empty():
    assert "(no spans)" in _build_span_tree([])


def test_build_span_tree_with_io(make_mock_span):
    span = make_mock_span(
        name="tool",
        span_id="s1",
        inputs={"query": "test"},
        outputs={"result": "ok"},
    )
    tree = _build_span_tree([span])

    assert "in: " in tree
    assert "out: " in tree


# ---- _build_enriched_trace_summary ----


def test_build_enriched_trace_summary(make_mock_span, make_trace):
    root = make_mock_span(name="agent", span_id="s1", span_type="AGENT")
    child = make_mock_span(
        name="tool_call",
        span_id="s2",
        parent_id="s1",
        span_type="TOOL",
        status_code=SpanStatusCode.ERROR,
        status_description="Failed",
    )
    trace = make_trace(
        trace_id="t-1",
        request_preview="Hello",
        response_preview="Hi there",
        execution_duration=200,
        spans=[root, child],
    )
    text = _build_enriched_trace_summary(0, trace, "Response was incomplete")

    assert "[0] trace_id=t-1" in text
    assert "Hello" in text
    assert "Hi there" in text
    assert "200ms" in text
    assert "Response was incomplete" in text
    assert "Span tree:" in text
    assert "tool_call" in text


def test_build_enriched_trace_summary_truncates_previews(make_trace):
    long_text = "x" * 1000
    trace = make_trace(request_preview=long_text, response_preview=long_text)
    text = _build_enriched_trace_summary(0, trace, "")
    assert text.count("x") <= 1000


def test_build_enriched_trace_summary_none_previews(make_trace):
    trace = make_trace()
    trace.info.request_preview = None
    trace.info.response_preview = None
    text = _build_enriched_trace_summary(0, trace, "rationale")
    assert "Input: \n" in text
    assert "Output: \n" in text


# ---- _run_deep_analysis ----


def test_run_deep_analysis():
    mock_result = _BatchTraceAnalysisResult(
        analyses=[
            _TraceAnalysis(
                trace_index=0,
                failure_category="tool_error",
                failure_summary="Tool API call failed",
                root_cause_hypothesis="External API was unreachable",
                affected_spans=["weather_tool"],
                severity=4,
            )
        ]
    )
    with patch(
        "mlflow.genai.discovery.utils.get_chat_completions_with_structured_output",
        return_value=mock_result,
    ) as mock_llm:
        analyses = _run_deep_analysis(["[0] trace summary..."], "openai:/gpt-5")

    assert len(analyses) == 1
    assert analyses[0].failure_category == "tool_error"
    assert analyses[0].affected_spans == ["weather_tool"]
    mock_llm.assert_called_once()
    assert mock_llm.call_args[1]["model_uri"] == "openai:/gpt-5"


# ---- _format_analysis_for_clustering ----


def test_format_analysis_for_clustering():
    analysis = _TraceAnalysis(
        trace_index=0,
        failure_category="tool_error",
        failure_summary="Tool failed",
        root_cause_hypothesis="API down",
        affected_spans=["weather_tool"],
        severity=4,
    )
    text = _format_analysis_for_clustering(0, analysis, "[0] trace summary...")

    assert "Category: tool_error" in text
    assert "Severity: 4/5" in text
    assert "Tool failed" in text
    assert "API down" in text
    assert "weather_tool" in text
    assert "[0] trace summary..." in text


# ---- _generate_scorer_specs ----


def test_generate_scorer_specs():
    issue = _IdentifiedIssue(
        name="tool_error",
        description="Tool calls fail",
        root_cause="API timeout",
        example_indices=[0, 1],
        confidence=90,
    )
    analyses = [
        _TraceAnalysis(
            trace_index=0,
            failure_category="tool_error",
            failure_summary="Tool API call failed",
            root_cause_hypothesis="External API unreachable",
            affected_spans=["weather_tool"],
            severity=4,
        )
    ]
    mock_result = _ScorerInstructionsResult(
        scorers=[
            _ScorerSpec(
                name="tool_error",
                detection_instructions="Analyze the {{ trace }} for tool failures",
            )
        ]
    )
    with patch(
        "mlflow.genai.discovery.utils.get_chat_completions_with_structured_output",
        return_value=mock_result,
    ) as mock_llm:
        specs = _generate_scorer_specs(issue, analyses, "openai:/gpt-5-mini")

    assert len(specs) == 1
    assert "{{ trace }}" in specs[0].detection_instructions
    mock_llm.assert_called_once()
    assert mock_llm.call_args[1]["model_uri"] == "openai:/gpt-5-mini"


def test_generate_scorer_specs_splits_compound_criteria():
    issue = _IdentifiedIssue(
        name="compound_issue",
        description="Response is truncated and uses wrong API",
        root_cause="Multiple failures",
        example_indices=[0],
        confidence=90,
    )
    mock_result = _ScorerInstructionsResult(
        scorers=[
            _ScorerSpec(
                name="response_truncation",
                detection_instructions="Analyze the {{ trace }} for truncated responses",
            ),
            _ScorerSpec(
                name="wrong_api_usage",
                detection_instructions="Analyze the {{ trace }} for wrong API calls",
            ),
        ]
    )
    with patch(
        "mlflow.genai.discovery.utils.get_chat_completions_with_structured_output",
        return_value=mock_result,
    ):
        specs = _generate_scorer_specs(issue, [], "openai:/gpt-5-mini")

    assert len(specs) == 2
    assert specs[0].name == "response_truncation"
    assert specs[1].name == "wrong_api_usage"


def test_generate_scorer_specs_adds_template_var():
    issue = _IdentifiedIssue(
        name="test",
        description="Test",
        root_cause="Test",
        example_indices=[0],
        confidence=90,
    )
    mock_result = _ScorerInstructionsResult(
        scorers=[_ScorerSpec(name="test", detection_instructions="Check if the response is bad")]
    )
    with patch(
        "mlflow.genai.discovery.utils.get_chat_completions_with_structured_output",
        return_value=mock_result,
    ):
        specs = _generate_scorer_specs(issue, [], "openai:/gpt-5-mini")

    assert "{{ trace }}" in specs[0].detection_instructions


# ---- _extract_failing_traces ----


def test_extract_failing_traces(make_trace):
    traces = [make_trace(trace_id=f"t-{i}") for i in range(3)]
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


def test_extract_failing_traces_no_failures(make_trace):
    traces = [make_trace(trace_id="t-0")]
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
            "issue_a/value": [False, False, False, True, True, True, True, True, True, True],
            "issue_a/rationale": ["r1", "r2", "r3"] + ["ok"] * 7,
            "issue_b/value": [False, True, True, True, True, True, True, True, True, True],
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
            confidence=85,
        ),
    ]
    summary = _build_summary(issues, 100)
    assert "tool_failure" in summary
    assert "30%" in summary
    assert "API timeout" in summary


# ---- _get_existing_score ----


def test_get_existing_score_true(make_trace, make_assessment):
    trace = make_trace(assessments=[make_assessment("my_scorer", True)])
    assert _get_existing_score(trace, "my_scorer") is True


def test_get_existing_score_false(make_trace, make_assessment):
    trace = make_trace(assessments=[make_assessment("my_scorer", False)])
    assert _get_existing_score(trace, "my_scorer") is False


def test_get_existing_score_none_when_no_assessments(make_trace):
    trace = make_trace(assessments=[])
    assert _get_existing_score(trace, "my_scorer") is None


def test_get_existing_score_filters_by_name(make_trace, make_assessment):
    trace = make_trace(assessments=[make_assessment("other_scorer", True)])
    assert _get_existing_score(trace, "my_scorer") is None


def test_get_existing_score_ignores_non_bool(make_trace):
    fb = Feedback(
        name="my_scorer",
        value="some_string",
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="test"),
    )
    trace = make_trace(assessments=[fb])
    assert _get_existing_score(trace, "my_scorer") is None


# ---- _partition_by_existing_scores ----


def test_partition_by_existing_scores(make_trace, make_assessment):
    neg_trace = make_trace(trace_id="neg", assessments=[make_assessment("scorer", False)])
    pos_trace = make_trace(trace_id="pos", assessments=[make_assessment("scorer", True)])
    unscored_trace = make_trace(trace_id="unscored", assessments=[])

    negative, positive, needs_scoring = _partition_by_existing_scores(
        [neg_trace, pos_trace, unscored_trace], "scorer"
    )

    assert len(negative) == 1
    assert negative[0].info.trace_id == "neg"
    assert len(positive) == 1
    assert positive[0].info.trace_id == "pos"
    assert len(needs_scoring) == 1
    assert needs_scoring[0].info.trace_id == "unscored"


# ---- _has_session_ids ----


def test_has_session_ids_true_via_tag(make_trace):
    trace = make_trace()
    trace.info.tags = {"mlflow.trace.session_id": "session-1"}
    trace.info.trace_metadata = {}
    assert _has_session_ids([trace]) is True


def test_has_session_ids_true_via_metadata(make_trace):
    trace = make_trace()
    trace.info.tags = {}
    trace.info.trace_metadata = {"mlflow.trace.session": "session-1"}
    assert _has_session_ids([trace]) is True


def test_has_session_ids_false(make_trace):
    trace = make_trace()
    trace.info.tags = {}
    trace.info.trace_metadata = {}
    assert _has_session_ids([trace]) is False


def test_has_session_ids_mixed(make_trace):
    t1 = make_trace(trace_id="t-1")
    t1.info.tags = {}
    t1.info.trace_metadata = {}
    t2 = make_trace(trace_id="t-2")
    t2.info.tags = {"mlflow.trace.session_id": "session-1"}
    t2.info.trace_metadata = {}
    assert _has_session_ids([t1, t2]) is True


# ---- _build_default_satisfaction_scorer ----


@pytest.mark.parametrize(
    ("use_conversation", "expected_var"),
    [
        (True, "{{ conversation }}"),
        (False, "{{ trace }}"),
    ],
)
def test_build_default_satisfaction_scorer(use_conversation, expected_var):
    with patch(
        "mlflow.genai.discovery.utils.make_judge", return_value=MagicMock()
    ) as mock_make_judge:
        _build_default_satisfaction_scorer("openai:/gpt-4", use_conversation=use_conversation)

    mock_make_judge.assert_called_once()
    call_kwargs = mock_make_judge.call_args[1]
    assert call_kwargs["name"] == _DEFAULT_SCORER_NAME
    assert call_kwargs["feedback_value_type"] is bool
    assert call_kwargs["model"] == "openai:/gpt-4"
    assert expected_var in call_kwargs["instructions"]
