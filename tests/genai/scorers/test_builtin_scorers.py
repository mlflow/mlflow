import pytest

from mlflow.genai.scorers import (
    chunk_relevance,
    context_sufficiency,
    global_guideline_adherence,
    groundedness,
    guideline_adherence,
    rag_scorers,
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
    chunk_relevance(),
    context_sufficiency(),
    groundedness(),
    guideline_adherence(),
    global_guideline_adherence(["Be polite", "Be kind"]),
    relevance_to_query(),
    safety(),
]

expected = {
    GENAI_CONFIG_NAME: {
        "metrics": [
            "chunk_relevance",
            "context_sufficiency",
            "groundedness",
            "guideline_adherence",
            "relevance_to_query",
            "safety",
        ],
        "global_guidelines": {
            "guideline_adherence": ["Be polite", "Be kind"],
        },
    }
}


@pytest.mark.parametrize(
    "scorers",
    [
        ALL_SCORERS,
        [*ALL_SCORERS] + [*ALL_SCORERS],  # duplicate scorers
        rag_scorers()
        + [global_guideline_adherence(["Be polite", "Be kind"]), guideline_adherence(), safety()],
        [*rag_scorers()]
        + [global_guideline_adherence(["Be polite", "Be kind"]), guideline_adherence(), safety()],
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
    scorer = global_guideline_adherence(["Be polite", "Be kind"])
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

    guideline = global_guideline_adherence(["Be polite", "Be kind"])  # w/ default name
    english = global_guideline_adherence(
        name="english",
        guidelines=["The response must be in English"],
    )
    clarify = global_guideline_adherence(
        name="clarify",
        guidelines=["The response must be clear, coherent, and concise"],
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
        [global_guideline_adherence(["Be polite", "Be kind"]), guideline_adherence()],
        [guideline_adherence(), global_guideline_adherence(["Be polite", "Be kind"])],
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
