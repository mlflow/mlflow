from unittest.mock import patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import (
    chunk_relevance,
    context_sufficiency,
    correctness,
    get_all_scorers,
    get_rag_scorers,
    groundedness,
    guideline_adherence,
    relevance_to_query,
    safety,
)
from mlflow.genai.scorers.builtin_scorers import GENAI_CONFIG_NAME


def normalize_config(config):
    config = config.copy()
    metrics = config.get(GENAI_CONFIG_NAME, {}).get("metrics", [])
    config.setdefault(GENAI_CONFIG_NAME, {})["metrics"] = sorted(metrics)
    return config


ALL_SCORERS = [
    guideline_adherence.with_config(name="politeness", global_guidelines=["Be polite", "Be kind"]),
    *get_all_scorers(),
]

expected = {
    GENAI_CONFIG_NAME: {
        "metrics": [
            "correctness",
            "chunk_relevance",
            "context_sufficiency",
            "groundedness",
            "guideline_adherence",
            "relevance_to_query",
            "safety",
        ],
        "global_guidelines": {
            "politeness": ["Be polite", "Be kind"],
        },
    }
}


@pytest.mark.parametrize(
    "scorers",
    [
        ALL_SCORERS,
        ALL_SCORERS + ALL_SCORERS,  # duplicate scorers
        get_rag_scorers()
        + [
            guideline_adherence.with_config(
                name="politeness", global_guidelines=["Be polite", "Be kind"]
            ),
            guideline_adherence,
            correctness,
            safety,
        ],
    ],
)
def test_scorers_and_rag_scorers_config(scorers):
    evaluation_config = {}
    for scorer in scorers:
        evaluation_config = scorer.update_evaluation_config(evaluation_config)

    assert normalize_config(evaluation_config) == normalize_config(expected)


@pytest.mark.parametrize(
    ("scorer", "expected_metric"),
    [
        (chunk_relevance, "chunk_relevance"),
        (context_sufficiency, "context_sufficiency"),
        (correctness, "correctness"),
        (groundedness, "groundedness"),
        (guideline_adherence, "guideline_adherence"),
        (relevance_to_query, "relevance_to_query"),
        (safety, "safety"),
    ],
)
def test_individual_scorers(scorer, expected_metric):
    """Test that each individual scorer correctly updates the evaluation config."""
    evaluation_config = {}
    evaluation_config = scorer.update_evaluation_config(evaluation_config)

    expected_conf = {
        GENAI_CONFIG_NAME: {
            "metrics": [expected_metric],
        }
    }

    assert normalize_config(evaluation_config) == normalize_config(expected_conf)


def test_global_guideline_adherence():
    """Test that the global guideline adherence scorer correctly updates the evaluation config."""
    evaluation_config = {}
    scorer = guideline_adherence.with_config(global_guidelines=["Be polite", "Be kind"])
    evaluation_config = scorer.update_evaluation_config(evaluation_config)

    expected_conf = {
        GENAI_CONFIG_NAME: {
            "metrics": ["guideline_adherence"],
            "global_guidelines": {
                "guideline_adherence": ["Be polite", "Be kind"],
            },
        }
    }

    assert normalize_config(evaluation_config) == normalize_config(expected_conf)


def test_multiple_global_guideline_adherence():
    """Test passing multiple global guideline adherence scorers with different names."""
    evaluation_config = {}

    guideline = guideline_adherence.with_config(
        global_guidelines=["Be polite", "Be kind"]
    )  # w/ default name
    english = guideline_adherence.with_config(
        name="english",
        global_guidelines=["The response must be in English"],
    )
    clarify = guideline_adherence.with_config(
        name="clarify",
        global_guidelines=["The response must be clear, coherent, and concise"],
    )

    scorers = [guideline, english, clarify]
    for scorer in scorers:
        evaluation_config = scorer.update_evaluation_config(evaluation_config)

    expected_conf = {
        GENAI_CONFIG_NAME: {
            "metrics": ["guideline_adherence"],
            "global_guidelines": {
                "guideline_adherence": ["Be polite", "Be kind"],
                "english": ["The response must be in English"],
                "clarify": ["The response must be clear, coherent, and concise"],
            },
        }
    }
    assert normalize_config(evaluation_config) == normalize_config(expected_conf)


@pytest.mark.parametrize(
    "scorers",
    [
        [
            guideline_adherence.with_config(global_guidelines=["Be polite", "Be kind"]),
            guideline_adherence,
        ],
        [
            guideline_adherence,
            guideline_adherence.with_config(global_guidelines=["Be polite", "Be kind"]),
        ],
    ],
)
def test_guideline_adherence_scorers(scorers):
    evaluation_config = {}
    for scorer in scorers:
        evaluation_config = scorer.update_evaluation_config(evaluation_config)

    expected_conf = {
        GENAI_CONFIG_NAME: {
            "metrics": [
                "guideline_adherence",
            ],
            "global_guidelines": {
                "guideline_adherence": ["Be polite", "Be kind"],
            },
        }
    }

    assert normalize_config(evaluation_config) == normalize_config(expected_conf)


def test_builtin_scorer_block_mutations():
    """Test that the built-in scorers are immutable."""
    with pytest.raises(MlflowException, match=r"Built-in scorer fields are immutable"):
        chunk_relevance.name = "new_name"


@pytest.mark.parametrize(
    ("scorer", "updates"),
    [
        (chunk_relevance, {"name": "custom_name"}),
        (context_sufficiency, {"name": "custom_name"}),
        (groundedness, {"name": "custom_name"}),
        (relevance_to_query, {"name": "custom_name"}),
        (safety, {"name": "custom_name"}),
        (correctness, {"name": "custom_name"}),
        (
            guideline_adherence,
            {"name": "custom_name", "global_guidelines": ["Be polite", "Be kind"]},
        ),
    ],
    ids=lambda x: x.__class__.__name__,
)
def test_configure_builtin_scorers(scorer, updates):
    updated_scorer = scorer.with_config(**updates)

    assert updated_scorer is not scorer  # with_config() should return a new instance
    assert isinstance(updated_scorer, scorer.__class__)
    for key, value in updates.items():
        assert getattr(updated_scorer, key) == value

    # Positional argument should not be allowed
    with pytest.raises(TypeError, match=rf"{scorer.__class__.__name__}.with_config\(\) takes"):
        scorer.with_config("custom_name")


def test_groundedness(sample_rag_trace):
    with patch("databricks.agents.evals.judges.groundedness") as mock_groundedness:
        groundedness(trace=sample_rag_trace)

    mock_groundedness.assert_called_once_with(
        request="query",
        response="answer",
        retrieved_context=[
            {"content": "content_1", "doc_uri": "url_1"},
            {"content": "content_2", "doc_uri": "url_2"},
            {"content": "content_3"},
        ],
        assessment_name="groundedness",
    )


def test_chunk_relevance(sample_rag_trace):
    with patch("databricks.agents.evals.judges.chunk_relevance") as mock_chunk_relevance:
        chunk_relevance(trace=sample_rag_trace)

    mock_chunk_relevance.assert_called_once_with(
        request="query",
        retrieved_context=[
            {"content": "content_1", "doc_uri": "url_1"},
            {"content": "content_2", "doc_uri": "url_2"},
            {"content": "content_3"},
        ],
        assessment_name="chunk_relevance",
    )


def test_context_sufficiency(sample_rag_trace):
    with patch("databricks.agents.evals.judges.context_sufficiency") as mock_context_sufficiency:
        context_sufficiency(trace=sample_rag_trace)

    mock_context_sufficiency.assert_called_once_with(
        request="query",
        retrieved_context=[
            {"content": "content_1", "doc_uri": "url_1"},
            {"content": "content_2", "doc_uri": "url_2"},
            {"content": "content_3"},
        ],
        expected_response="expected answer",
        expected_facts=["fact1", "fact2"],
        assessment_name="context_sufficiency",
    )


def test_guideline_adherence():
    # 1. Called with per-row guidelines
    with patch("databricks.agents.evals.judges.guideline_adherence") as mock_guideline_adherence:
        guideline_adherence(
            inputs={"question": "query"},
            outputs="answer",
            expectations={"guidelines": ["guideline1", "guideline2"]},
        )

    mock_guideline_adherence.assert_called_once_with(
        request="query",
        response="answer",
        guidelines=["guideline1", "guideline2"],
        assessment_name="guideline_adherence",
    )

    # 2. Called with global guidelines
    is_english = guideline_adherence.with_config(
        name="is_english",
        global_guidelines=["The response should be in English."],
    )

    with patch("databricks.agents.evals.judges.guideline_adherence") as mock_guideline_adherence:
        is_english(
            inputs={"question": "query"},
            outputs="answer",
        )

    mock_guideline_adherence.assert_called_once_with(
        request="query",
        response="answer",
        guidelines=["The response should be in English."],
        assessment_name="is_english",
    )


def test_relevance_to_query():
    with patch("databricks.agents.evals.judges.relevance_to_query") as mock_relevance_to_query:
        relevance_to_query(
            inputs={"question": "query"},
            outputs="answer",
        )

    mock_relevance_to_query.assert_called_once_with(
        request="query",
        response="answer",
        assessment_name="relevance_to_query",
    )


def test_safety():
    with patch("databricks.agents.evals.judges.safety") as mock_safety:
        safety(
            inputs={"question": "query"},
            outputs="answer",
        )

    mock_safety.assert_called_once_with(
        request="query",
        response="answer",
        assessment_name="safety",
    )


def test_correctness():
    with patch("databricks.agents.evals.judges.correctness") as mock_correctness:
        correctness(
            inputs={"question": "query"},
            outputs="answer",
            expectations={"expected_facts": ["fact1", "fact2"]},
        )

    mock_correctness.assert_called_once_with(
        request="query",
        response="answer",
        expected_facts=["fact1", "fact2"],
        expected_response=None,
        assessment_name="correctness",
    )
