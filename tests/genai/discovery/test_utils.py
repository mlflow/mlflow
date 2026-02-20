from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.genai.discovery.constants import _DEFAULT_SCORER_NAME
from mlflow.genai.discovery.entities import (
    Issue,
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
    _sample_traces,
)
from mlflow.genai.evaluation.entities import EvaluationResult

# ---- _build_span_tree ----


def test_build_span_tree(make_trace):
    trace = make_trace()
    tree = _build_span_tree(trace.data.spans)

    assert "agent" in tree
    assert "llm_call" in tree
    assert "LLM" in tree
    assert "OK" in tree


def test_build_span_tree_with_error_span(make_trace):
    trace = make_trace(error_span=True)
    tree = _build_span_tree(trace.data.spans)

    assert "tool_call" in tree
    assert "ERROR" in tree
    assert "Connection failed" in tree


def test_build_span_tree_empty():
    assert "(no spans)" in _build_span_tree([])


def test_build_span_tree_io(make_trace):
    trace = make_trace()
    tree = _build_span_tree(trace.data.spans)

    assert "in:" in tree
    assert "out:" in tree


# ---- _build_enriched_trace_summary ----


def test_build_enriched_trace_summary(make_trace):
    trace = make_trace(
        request_input="Hello",
        response_output="Hi there",
        error_span=True,
    )
    text = _build_enriched_trace_summary(0, trace, "Response was incomplete")

    assert f"[0] trace_id={trace.info.trace_id}" in text
    assert "Response was incomplete" in text
    assert "Span tree:" in text
    assert "tool_call" in text


def test_build_enriched_trace_summary_truncates_previews(make_trace):
    trace = make_trace(request_input="x" * 10000, response_output="y" * 10000)
    text = _build_enriched_trace_summary(0, trace, "")
    assert "[..TRIMMED BY ANALYSIS TOOL]" in text
    input_line = next(line for line in text.split("\n") if line.strip().startswith("Input:"))
    output_line = next(line for line in text.split("\n") if line.strip().startswith("Output:"))
    assert len(input_line) < 10000
    assert len(output_line) < 10000


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
    ) as mock_llm:
        specs = _generate_scorer_specs(issue, [], "openai:/gpt-5-mini")

    assert len(specs) == 2
    assert specs[0].name == "response_truncation"
    assert specs[1].name == "wrong_api_usage"
    mock_llm.assert_called_once()


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
    ) as mock_llm:
        specs = _generate_scorer_specs(issue, [], "openai:/gpt-5-mini")

    assert "{{ trace }}" in specs[0].detection_instructions
    mock_llm.assert_called_once()


# ---- _extract_failing_traces ----


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

    failing, rationales = _extract_failing_traces(eval_result, "satisfaction")

    assert len(failing) == 2
    assert failing[0].info.trace_id == traces[1].info.trace_id
    assert failing[1].info.trace_id == traces[2].info.trace_id
    assert rationales[traces[1].info.trace_id] == "bad response"
    assert rationales[traces[2].info.trace_id] == "incomplete"


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
    traces = [make_trace()]
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
    trace = make_trace()
    trace.info.assessments = [make_assessment("my_scorer", True)]
    assert _get_existing_score(trace, "my_scorer") is True


def test_get_existing_score_false(make_trace, make_assessment):
    trace = make_trace()
    trace.info.assessments = [make_assessment("my_scorer", False)]
    assert _get_existing_score(trace, "my_scorer") is False


def test_get_existing_score_none_when_no_assessments(make_trace):
    trace = make_trace()
    trace.info.assessments = []
    assert _get_existing_score(trace, "my_scorer") is None


def test_get_existing_score_filters_by_name(make_trace, make_assessment):
    trace = make_trace()
    trace.info.assessments = [make_assessment("other_scorer", True)]
    assert _get_existing_score(trace, "my_scorer") is None


def test_get_existing_score_ignores_non_bool(make_trace):
    fb = Feedback(
        name="my_scorer",
        value="some_string",
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="test"),
    )
    trace = make_trace()
    trace.info.assessments = [fb]
    assert _get_existing_score(trace, "my_scorer") is None


# ---- _partition_by_existing_scores ----


def test_partition_by_existing_scores(make_trace, make_assessment):
    neg_trace = make_trace()
    neg_trace.info.assessments = [make_assessment("scorer", False)]
    pos_trace = make_trace()
    pos_trace.info.assessments = [make_assessment("scorer", True)]
    unscored_trace = make_trace()
    unscored_trace.info.assessments = []

    negative, positive, needs_scoring = _partition_by_existing_scores(
        [neg_trace, pos_trace, unscored_trace], "scorer"
    )

    assert len(negative) == 1
    assert negative[0].info.trace_id == neg_trace.info.trace_id
    assert len(positive) == 1
    assert positive[0].info.trace_id == pos_trace.info.trace_id
    assert len(needs_scoring) == 1
    assert needs_scoring[0].info.trace_id == unscored_trace.info.trace_id


# ---- _has_session_ids ----


def test_has_session_ids_true(make_trace):
    trace = make_trace(session_id="session-1")
    assert _has_session_ids([trace]) is True


def test_has_session_ids_false(make_trace):
    trace = make_trace()
    assert _has_session_ids([trace]) is False


def test_has_session_ids_mixed(make_trace):
    t1 = make_trace()
    t2 = make_trace(session_id="session-1")
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


# ---- _sample_traces ----


def test_sample_traces_no_sessions(make_trace):
    traces = [make_trace() for _ in range(20)]
    search_kwargs = {"filter_string": None, "return_type": "list", "locations": ["exp-1"]}

    with patch(
        "mlflow.genai.discovery.utils.mlflow.search_traces", return_value=traces
    ) as mock_search:
        result = _sample_traces(5, search_kwargs)

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
        result = _sample_traces(2, search_kwargs)

    mock_search.assert_called_once()
    session_ids = {
        (t.info.tags or {}).get("mlflow.trace.session_id")
        or (t.info.trace_metadata or {}).get("mlflow.trace.session")
        for t in result
    }
    assert len(session_ids) == 2


def test_sample_traces_empty_pool():
    search_kwargs = {"filter_string": None, "return_type": "list", "locations": ["exp-1"]}

    with patch("mlflow.genai.discovery.utils.mlflow.search_traces", return_value=[]) as mock_search:
        result = _sample_traces(10, search_kwargs)

    mock_search.assert_called_once()
    assert result == []


def test_sample_traces_fewer_than_requested(make_trace):
    traces = [make_trace() for _ in range(3)]
    search_kwargs = {"filter_string": None, "return_type": "list", "locations": ["exp-1"]}

    with patch(
        "mlflow.genai.discovery.utils.mlflow.search_traces", return_value=traces
    ) as mock_search:
        result = _sample_traces(10, search_kwargs)

    mock_search.assert_called_once()
    assert len(result) == 3
