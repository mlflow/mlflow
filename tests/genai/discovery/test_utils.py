from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mlflow.genai.discovery.constants import _DEFAULT_SCORER_NAME
from mlflow.genai.discovery.entities import (
    Issue,
    _ConversationAnalysis,
    _IdentifiedIssue,
)
from mlflow.genai.discovery.utils import (
    _build_default_satisfaction_scorer,
    _build_summary,
    _cluster_analyses,
    _embed_texts,
    _extract_failing_traces,
    _group_traces_by_session,
    _has_session_ids,
    _sample_traces,
    _summarize_cluster,
)
from mlflow.genai.evaluation.entities import EvaluationResult

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
        groups = _cluster_analyses(analyses, "openai:/text-embedding-3-small", max_issues=2)

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


def test_extract_failing_traces_with_list_of_scorer_names(make_trace):
    traces = [make_trace() for _ in range(3)]
    df = pd.DataFrame(
        {
            "satisfaction/value": [True, False, True],
            "satisfaction/rationale": ["good", "bad response", "good"],
            "quality/value": [True, True, False],
            "quality/rationale": ["ok", "ok", "poor quality"],
            "trace": traces,
        }
    )
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=df)

    failing, rationales = _extract_failing_traces(eval_result, ["satisfaction", "quality"])

    assert len(failing) == 2
    assert failing[0].info.trace_id == traces[1].info.trace_id
    assert failing[1].info.trace_id == traces[2].info.trace_id
    assert rationales[traces[1].info.trace_id] == "bad response"
    assert rationales[traces[2].info.trace_id] == "poor quality"


def test_extract_failing_traces_multiple_scorers_fail_same_row(make_trace):
    traces = [make_trace() for _ in range(2)]
    df = pd.DataFrame(
        {
            "scorer_a/value": [False, True],
            "scorer_a/rationale": ["reason a", "ok"],
            "scorer_b/value": [False, True],
            "scorer_b/rationale": ["reason b", "ok"],
            "trace": traces,
        }
    )
    eval_result = EvaluationResult(run_id="run-1", metrics={}, result_df=df)

    failing, rationales = _extract_failing_traces(eval_result, ["scorer_a", "scorer_b"])

    assert len(failing) == 1
    assert failing[0].info.trace_id == traces[0].info.trace_id
    assert "reason a" in rationales[traces[0].info.trace_id]
    assert "reason b" in rationales[traces[0].info.trace_id]


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
