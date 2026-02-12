from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mlflow.entities import Trace, TraceData, TraceInfo
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span import Span
from mlflow.entities.span_event import SpanEvent
from mlflow.entities.span_status import SpanStatus, SpanStatusCode
from mlflow.genai.discovery import (
    _DEFAULT_SCORER_NAME,
    Issue,
    _BatchTraceAnalysisResult,
    _ScorerInstructions,
    _TraceAnalysis,
    _build_default_satisfaction_scorer,
    _build_enriched_trace_summary,
    _build_span_tree,
    _build_summary,
    _compute_frequencies,
    _extract_failing_traces,
    _format_analysis_for_clustering,
    _generate_scorer_instructions,
    _get_existing_score,
    _IdentifiedIssue,
    _IssueClusteringResult,
    _partition_by_existing_scores,
    _run_deep_analysis,
    discover_issues,
)
from mlflow.genai.evaluation.entities import EvaluationResult


def _make_mock_span(
    name="test_span",
    status_code=SpanStatusCode.OK,
    span_id="span-1",
    parent_id=None,
    span_type="UNKNOWN",
    start_time_ns=0,
    end_time_ns=100_000_000,
    model_name=None,
    events=None,
    inputs=None,
    outputs=None,
    status_description="",
):
    span = MagicMock(spec=Span)
    span.name = name
    span.span_id = span_id
    span.parent_id = parent_id
    span.span_type = span_type
    span.start_time_ns = start_time_ns
    span.end_time_ns = end_time_ns
    span.model_name = model_name
    span.status = SpanStatus(status_code=status_code, description=status_description)
    span.events = events or []
    span.inputs = inputs
    span.outputs = outputs
    return span


def _make_trace(
    trace_id="trace-1",
    request_preview="What is MLflow?",
    response_preview="MLflow is an ML platform.",
    execution_duration=500,
    spans=None,
    assessments=None,
):
    info = MagicMock(spec=TraceInfo)
    info.trace_id = trace_id
    info.request_preview = request_preview
    info.response_preview = response_preview
    info.execution_duration = execution_duration
    info.assessments = assessments or []

    data = MagicMock(spec=TraceData)
    data.spans = spans or [_make_mock_span()]

    trace = MagicMock(spec=Trace)
    trace.info = info
    trace.data = data
    return trace


def _make_assessment(name, value):
    return Feedback(
        name=name,
        value=value,
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="test"),
    )


# ---- _build_span_tree ----


def test_build_span_tree_simple():
    root = _make_mock_span(
        name="agent",
        span_id="s1",
        parent_id=None,
        span_type="AGENT",
        start_time_ns=0,
        end_time_ns=1_500_000_000,
    )
    child = _make_mock_span(
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


def test_build_span_tree_error_with_exception():
    exc_event = MagicMock(spec=SpanEvent)
    exc_event.name = "exception"
    exc_event.attributes = {
        "exception.type": "ConnectionTimeout",
        "exception.message": "API unreachable",
    }
    span = _make_mock_span(
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


def test_build_span_tree_with_io():
    span = _make_mock_span(
        name="tool",
        span_id="s1",
        inputs={"query": "test"},
        outputs={"result": "ok"},
    )
    tree = _build_span_tree([span])

    assert "in: " in tree
    assert "out: " in tree


# ---- _build_enriched_trace_summary ----


def test_build_enriched_trace_summary():
    root = _make_mock_span(name="agent", span_id="s1", span_type="AGENT")
    child = _make_mock_span(
        name="tool_call",
        span_id="s2",
        parent_id="s1",
        span_type="TOOL",
        status_code=SpanStatusCode.ERROR,
        status_description="Failed",
    )
    trace = _make_trace(
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


def test_build_enriched_trace_summary_truncates_previews():
    long_text = "x" * 1000
    trace = _make_trace(request_preview=long_text, response_preview=long_text)
    text = _build_enriched_trace_summary(0, trace, "")
    # Each preview should be truncated to 500 chars
    assert text.count("x") <= 1000


def test_build_enriched_trace_summary_none_previews():
    trace = _make_trace()
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
        "mlflow.genai.discovery.get_chat_completions_with_structured_output",
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


# ---- _generate_scorer_instructions ----


def test_generate_scorer_instructions():
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
    mock_result = _ScorerInstructions(
        detection_instructions="Analyze the {{ trace }} for tool failures"
    )
    with patch(
        "mlflow.genai.discovery.get_chat_completions_with_structured_output",
        return_value=mock_result,
    ) as mock_llm:
        instructions = _generate_scorer_instructions(issue, analyses, "openai:/gpt-5-mini")

    assert "{{ trace }}" in instructions
    mock_llm.assert_called_once()
    assert mock_llm.call_args[1]["model_uri"] == "openai:/gpt-5-mini"


def test_generate_scorer_instructions_adds_template_var():
    issue = _IdentifiedIssue(
        name="test",
        description="Test",
        root_cause="Test",
        example_indices=[0],
        confidence=90,
    )
    mock_result = _ScorerInstructions(
        detection_instructions="Check if the response is bad"
    )
    with patch(
        "mlflow.genai.discovery.get_chat_completions_with_structured_output",
        return_value=mock_result,
    ):
        instructions = _generate_scorer_instructions(issue, [], "openai:/gpt-5-mini")

    assert "{{ trace }}" in instructions


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
            confidence=85,
        ),
    ]
    summary = _build_summary(issues, 100)
    assert "tool_failure" in summary
    assert "30%" in summary
    assert "API timeout" in summary


# ---- _get_existing_score ----


def test_get_existing_score_true():
    trace = _make_trace(assessments=[_make_assessment("my_scorer", True)])
    assert _get_existing_score(trace, "my_scorer") is True


def test_get_existing_score_false():
    trace = _make_trace(assessments=[_make_assessment("my_scorer", False)])
    assert _get_existing_score(trace, "my_scorer") is False


def test_get_existing_score_none_when_no_assessments():
    trace = _make_trace(assessments=[])
    assert _get_existing_score(trace, "my_scorer") is None


def test_get_existing_score_filters_by_name():
    trace = _make_trace(assessments=[_make_assessment("other_scorer", True)])
    assert _get_existing_score(trace, "my_scorer") is None


def test_get_existing_score_ignores_non_bool():
    fb = Feedback(
        name="my_scorer",
        value="some_string",
        source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="test"),
    )
    trace = _make_trace(assessments=[fb])
    assert _get_existing_score(trace, "my_scorer") is None


# ---- _partition_by_existing_scores ----


def test_partition_by_existing_scores():
    neg_trace = _make_trace(trace_id="neg", assessments=[_make_assessment("scorer", False)])
    pos_trace = _make_trace(trace_id="pos", assessments=[_make_assessment("scorer", True)])
    unscored_trace = _make_trace(trace_id="unscored", assessments=[])

    negative, positive, needs_scoring = _partition_by_existing_scores(
        [neg_trace, pos_trace, unscored_trace], "scorer"
    )

    assert len(negative) == 1
    assert negative[0].info.trace_id == "neg"
    assert len(positive) == 1
    assert positive[0].info.trace_id == "pos"
    assert len(needs_scoring) == 1
    assert needs_scoring[0].info.trace_id == "unscored"


# ---- _build_default_satisfaction_scorer ----


def test_build_default_satisfaction_scorer():
    with patch("mlflow.genai.discovery.make_judge", return_value=MagicMock()) as mock_make_judge:
        _build_default_satisfaction_scorer("openai:/gpt-4")

    mock_make_judge.assert_called_once()
    call_kwargs = mock_make_judge.call_args[1]
    assert call_kwargs["name"] == _DEFAULT_SCORER_NAME
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
        patch("mlflow.genai.discovery._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.mlflow.search_traces", return_value=traces),
        patch(
            "mlflow.genai.discovery.mlflow.genai.evaluate",
            side_effect=[test_eval, triage_eval],
        ),
    ):
        result = discover_issues()

    assert result.issues == []
    assert "no issues found" in result.summary.lower()


def test_discover_issues_full_pipeline():
    traces = [_make_trace(trace_id=f"t-{i}") for i in range(10)]

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

    # Phase 2: Deep analysis result
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

    # Phase 3: Clustering result (no detection_instructions field)
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

    # Phase 4: Scorer instructions result
    scorer_instructions_result = _ScorerInstructions(
        detection_instructions="Check the {{ trace }} execution duration"
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
            side_effect=[test_eval, triage_eval, validation_eval],
        ),
        patch(
            "mlflow.genai.discovery.get_chat_completions_with_structured_output",
            side_effect=[deep_analysis_result, clustering_result, scorer_instructions_result],
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

    # Phase 2: Deep analysis
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

    # Phase 3: Clustering — only 1 example (below minimum of 2)
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

    with (
        patch("mlflow.genai.discovery._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.mlflow.search_traces", return_value=traces),
        patch(
            "mlflow.genai.discovery.mlflow.genai.evaluate",
            side_effect=[test_eval, triage_eval],
        ),
        patch(
            "mlflow.genai.discovery.get_chat_completions_with_structured_output",
            side_effect=[deep_analysis_result, clustering_result],
        ),
    ):
        result = discover_issues(sample_size=5)

    # Filtered out: only 1 example (below minimum of 2)
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
        patch("mlflow.genai.discovery._get_experiment_id", return_value="exp-1"),
        patch("mlflow.genai.discovery.mlflow.search_traces", return_value=traces),
        patch(
            "mlflow.genai.discovery.mlflow.genai.evaluate",
            side_effect=[test_eval, triage_eval],
        ) as mock_eval,
    ):
        discover_issues(satisfaction_scorer=custom_scorer)

    # First call is the test scorer run, second is the full triage
    assert mock_eval.call_count == 2
    triage_call_kwargs = mock_eval.call_args_list[1][1]
    assert triage_call_kwargs["scorers"] == [custom_scorer]
