import pytest

from mlflow.genai.scorers import (
    chunk_relevance,
    context_sufficiency,
    groundedness,
    guideline_adherence,
    rag_scorers,
    relevance_to_query,
    safety,
)
from mlflow.genai.scorers.builtin_scorers import GENAI_CONFIG_NAME, all_scorers, correctness


def normalize_config(config):
    config = config.copy()
    metrics = config.get(GENAI_CONFIG_NAME, {}).get("metrics", [])
    config.setdefault(GENAI_CONFIG_NAME, {})["metrics"] = sorted(metrics)
    return config


ALL_SCORERS = [
    guideline_adherence(name="politeness", global_guidelines=["Be polite", "Be kind"]),
    *all_scorers(),
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
        rag_scorers()
        + [
            guideline_adherence(name="politeness", global_guidelines=["Be polite", "Be kind"]),
            guideline_adherence(),
            correctness(),
            safety(),
        ],
        [*rag_scorers()]
        + [
            guideline_adherence(name="politeness", global_guidelines=["Be polite", "Be kind"]),
            guideline_adherence(),
            correctness(),
            safety(),
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
        (chunk_relevance(), "chunk_relevance"),
        (context_sufficiency(), "context_sufficiency"),
        (correctness(), "correctness"),
        (groundedness(), "groundedness"),
        (guideline_adherence(), "guideline_adherence"),
        (relevance_to_query(), "relevance_to_query"),
        (safety(), "safety"),
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
    scorer = guideline_adherence(["Be polite", "Be kind"])
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

    guideline = guideline_adherence(["Be polite", "Be kind"])  # w/ default name
    english = guideline_adherence(
        name="english",
        global_guidelines=["The response must be in English"],
    )
    clarify = guideline_adherence(
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
        [guideline_adherence(["Be polite", "Be kind"]), guideline_adherence()],
        [guideline_adherence(), guideline_adherence(["Be polite", "Be kind"])],
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
