from unittest.mock import patch

import pytest

import mlflow.genai
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

    assert normalize_config(evaluation_config) == normalize_config(expected)


def test_evaluate_parameters():
    data = []
    with (
        patch("mlflow.get_tracking_uri", return_value="databricks"),
        patch("mlflow.genai.evaluation.base.is_model_traced", return_value=True),
        patch("mlflow.genai.evaluation.base._convert_to_legacy_eval_set", return_value=data),
        patch("mlflow.evaluate") as mock_evaluate,
    ):
        mlflow.genai.evaluate(
            data=data,
            scorers=ALL_SCORERS,
        )

        # Verify the call was made with the right parameters
        mock_evaluate.assert_called_once_with(
            model=None,
            data=data,
            evaluator_config=expected,
            extra_metrics=[],
            model_type=GENAI_CONFIG_NAME,
        )
