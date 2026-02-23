from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.genai.discovery.constants import _DEFAULT_SCORER_NAME
from mlflow.genai.discovery.entities import (
    Issue,
    _ConversationAnalysis,
    _ConversationAnalysisLLMResult,
    _IdentifiedIssue,
    _ScorerInstructionsResult,
    _ScorerSpec,
)
from mlflow.genai.discovery.utils import (
    _build_default_satisfaction_scorer,
    _build_enriched_trace_summary,
    _build_span_tree,
    _build_summary,
    _cluster_analyses,
    _compute_frequencies,
    _embed_texts,
    _extract_failing_traces,
    _generate_scorer_specs,
    _get_existing_score,
    _group_traces_by_session,
    _has_session_ids,
    _partition_by_existing_scores,
    _run_deep_analysis,
    _sample_traces,
    _summarize_cluster,
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


def test_build_enriched_trace_summary_includes_assessments(make_trace, make_assessment):
    trace = make_trace()
    trace.info.assessments = [
        make_assessment("correctness", False, rationale="Answer was wrong"),
        make_assessment("relevance", True, rationale="On topic"),
    ]
    text = _build_enriched_trace_summary(0, trace, "bad")

    assert "Assessments:" in text
    assert "correctness: False" in text
    assert "Answer was wrong" in text
    assert "relevance: True" in text
    assert "On topic" in text


def test_build_enriched_trace_summary_no_assessments(make_trace):
    trace = make_trace()
    trace.info.assessments = []
    text = _build_enriched_trace_summary(0, trace, "bad")

    assert "Assessments:" not in text


# ---- _run_deep_analysis ----


def test_run_deep_analysis(make_trace):
    mock_llm_result = _ConversationAnalysisLLMResult(
        surface="tool API call failure handling",
        root_cause="External API was unreachable. The system did not retry.",
        symptoms="Tool call returned an error. User saw no result.",
        domain="API integration",
        severity=4,
    )

    trace = make_trace(session_id="session-1")
    session_groups = {"session-1": [trace]}
    rationale_map = {trace.info.trace_id: "Tool failed"}

    with (
        patch(
            "mlflow.genai.discovery.utils.get_chat_completions_with_structured_output",
            return_value=mock_llm_result,
        ) as mock_llm,
        patch("mlflow.genai.judges.tools.list_judge_tools", return_value=[]),
    ):
        analyses = _run_deep_analysis(session_groups, rationale_map, "openai:/gpt-5")

    assert len(analyses) == 1
    assert analyses[0].surface == "tool API call failure handling"
    assert analyses[0].symptoms == "Tool call returned an error. User saw no result."
    assert analyses[0].domain == "API integration"
    assert analyses[0].severity == 4
    assert analyses[0].affected_trace_ids == [trace.info.trace_id]
    mock_llm.assert_called_once()
    assert mock_llm.call_args[1]["model_uri"] == "openai:/gpt-5"


# ---- _embed_texts ----


def test_embed_texts():
    mock_response = MagicMock()
    mock_response.data = [
        {"embedding": [0.1, 0.2, 0.3]},
        {"embedding": [0.4, 0.5, 0.6]},
    ]
    mock_litellm = MagicMock()
    mock_litellm.embedding.return_value = mock_response

    with patch.dict("sys.modules", {"litellm": mock_litellm}):
        result = _embed_texts(["hello", "world"], "openai:/text-embedding-3-small")

    mock_litellm.embedding.assert_called_once_with(
        model="openai/text-embedding-3-small", input=["hello", "world"]
    )
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


# ---- _cluster_analyses ----


def test_cluster_analyses_single_analysis():
    analyses = [
        _ConversationAnalysis(
            surface="response generation via LLM pipeline",
            root_cause="Model produced incorrect output.",
            symptoms="User received wrong answer.",
            domain="question answering",
            affected_trace_ids=["t-1"],
            severity=3,
        )
    ]
    result = _cluster_analyses(analyses, "openai:/text-embedding-3-small", max_issues=5)

    assert result == [[0]]


def test_cluster_analyses_groups_similar():
    analyses = [
        _ConversationAnalysis(
            surface="response generation via LLM pipeline",
            root_cause="Model hallucinated facts.",
            symptoms="User received incorrect information.",
            domain="question answering",
            affected_trace_ids=["t-1"],
            severity=4,
        ),
        _ConversationAnalysis(
            surface="response generation via LLM pipeline",
            root_cause="Model hallucinated different facts.",
            symptoms="User received fabricated details.",
            domain="question answering",
            affected_trace_ids=["t-2"],
            severity=4,
        ),
        _ConversationAnalysis(
            surface="database query execution timeout",
            root_cause="Query took too long.",
            symptoms="User waited and received no result.",
            domain="data retrieval",
            affected_trace_ids=["t-3"],
            severity=3,
        ),
    ]

    # Mock _embed_texts to return vectors that make first two similar, third different
    def mock_embed(texts, model):
        result = []
        for text in texts:
            if "response generation" in text.lower():
                result.append([1.0, 0.0, 0.0])
            else:
                result.append([0.0, 1.0, 0.0])
        return result

    with patch("mlflow.genai.discovery.utils._embed_texts", side_effect=mock_embed):
        groups = _cluster_analyses(analyses, "openai:/text-embedding-3-small", max_issues=5)

    # Should produce 2 groups: [0,1] and [2]
    assert len(groups) == 2
    flat = [idx for g in groups for idx in g]
    assert sorted(flat) == [0, 1, 2]


def test_cluster_analyses_respects_max_issues():
    analyses = [
        _ConversationAnalysis(
            surface=f"unique issue number {i}",
            root_cause=f"Unique root cause {i}.",
            symptoms=f"User observed issue {i}.",
            domain=f"domain {i}",
            affected_trace_ids=[f"t-{i}"],
            severity=3,
        )
        for i in range(5)
    ]

    # Each analysis gets a very different embedding
    def mock_embed(texts, model):
        import numpy as np

        rng = np.random.RandomState(42)
        return [rng.randn(128).tolist() for _ in texts]

    with patch("mlflow.genai.discovery.utils._embed_texts", side_effect=mock_embed):
        groups = _cluster_analyses(
            analyses, "openai:/text-embedding-3-small", max_issues=2
        )

    assert len(groups) <= 2
    # Only the largest clusters are kept; smallest are dropped
    flat = [idx for g in groups for idx in g]
    assert len(flat) <= 5


# ---- _summarize_cluster ----


def test_summarize_cluster():
    analyses = [
        _ConversationAnalysis(
            surface="response generation via LLM",
            root_cause="Model hallucinated.",
            symptoms="User received incorrect facts.",
            domain="question answering",
            affected_trace_ids=["t-1"],
            severity=4,
        ),
        _ConversationAnalysis(
            surface="response generation via LLM",
            root_cause="Model made up facts.",
            symptoms="User received fabricated information.",
            domain="question answering",
            affected_trace_ids=["t-2"],
            severity=4,
        ),
    ]

    mock_issue = _IdentifiedIssue(
        name="hallucination",
        description="LLM generates incorrect facts",
        root_cause="Model confabulation",
        example_indices=[],
        confidence=85,
    )

    with patch(
        "mlflow.genai.discovery.utils.get_chat_completions_with_structured_output",
        return_value=mock_issue,
    ) as mock_llm:
        result = _summarize_cluster([0, 1], analyses, "openai:/gpt-5")

    mock_llm.assert_called_once()
    assert mock_llm.call_args[1]["model_uri"] == "openai:/gpt-5"
    assert result.name == "hallucination"
    assert result.example_indices == [0, 1]


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
        _ConversationAnalysis(
            surface="tool API call failure handling",
            root_cause="External API was unreachable. The system did not retry.",
            symptoms="Tool call returned an error. User saw no result.",
            domain="API integration",
            affected_trace_ids=["trace-1"],
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
    # Verify the new field format is used in the prompt
    user_msg = mock_llm.call_args[1]["messages"][1].content
    assert "surface:" in user_msg
    assert "root cause:" in user_msg


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
        (False, "{{ inputs }}"),
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

    with patch(
        "mlflow.genai.discovery.utils.mlflow.search_traces", return_value=[]
    ) as mock_search:
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


# ---- _group_traces_by_session ----


def test_group_traces_by_session_with_sessions(make_trace):
    t1 = make_trace(session_id="s1")
    t2 = make_trace(session_id="s1")
    t3 = make_trace(session_id="s2")

    groups = _group_traces_by_session([t1, t2, t3])

    assert len(groups) == 2
    assert len(groups["s1"]) == 2
    assert len(groups["s2"]) == 1


def test_group_traces_by_session_no_sessions(make_trace):
    t1 = make_trace()
    t2 = make_trace()

    groups = _group_traces_by_session([t1, t2])

    assert len(groups) == 2
    # Each trace is its own "session" keyed by trace_id
    assert t1.info.trace_id in groups
    assert t2.info.trace_id in groups


def test_group_traces_by_session_mixed(make_trace):
    t1 = make_trace(session_id="s1")
    t2 = make_trace()

    groups = _group_traces_by_session([t1, t2])

    assert len(groups) == 2
    assert len(groups["s1"]) == 1
    assert t2.info.trace_id in groups
