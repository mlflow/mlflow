import pytest

from mlflow.genai.scorers import (
    chunk_relevance,
    context_sufficiency,
    document_recall,
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
    document_recall(),
    global_guideline_adherence("Be polite"),
    groundedness(),
    guideline_adherence(),
    relevance_to_query(),
    safety(),
]


@pytest.mark.parametrize(
    "scorers",
    [
        ALL_SCORERS,
        [*ALL_SCORERS] + [*ALL_SCORERS],  # duplicate scorers
        rag_scorers() + [global_guideline_adherence("Be polite"), guideline_adherence(), safety()],
        [*rag_scorers()]
        + [global_guideline_adherence("Be polite"), guideline_adherence(), safety()],
    ],
)
def test_scorers_and_rag_scorers_config(scorers):
    evaluation_config = {}
    for scorer in scorers:
        evaluation_config = scorer.update_evaluation_config(evaluation_config)

    expected = {
        GENAI_CONFIG_NAME: {
            "metrics": [
                "chunk_relevance",
                "context_sufficiency",
                "document_recall",
                "global_guideline_adherence",
                "groundedness",
                "guideline_adherence",
                "relevance_to_query",
                "safety",
            ],
            "global_guideline": "Be polite",
        }
    }

    assert normalize_config(evaluation_config) == normalize_config(expected)
