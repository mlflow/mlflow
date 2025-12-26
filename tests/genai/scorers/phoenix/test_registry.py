import sys
from unittest.mock import MagicMock, patch

import pytest

from mlflow.exceptions import MlflowException


@pytest.fixture(autouse=True)
def clear_phoenix_modules():
    """Clear phoenix module caches before each test."""
    mods_to_remove = [k for k in sys.modules if k.startswith("mlflow.genai.scorers.phoenix")]
    for mod in mods_to_remove:
        del sys.modules[mod]
    return


@pytest.mark.parametrize(
    ("metric_name", "evaluator_name"),
    [
        ("Hallucination", "HallucinationEvaluator"),
        ("Relevance", "RelevanceEvaluator"),
        ("Toxicity", "ToxicityEvaluator"),
        ("QA", "QAEvaluator"),
        ("Summarization", "SummarizationEvaluator"),
    ],
)
def test_get_evaluator_class(metric_name, evaluator_name):
    # Create a shared mock module for phoenix.evals
    mock_evals = MagicMock()
    mock_phoenix = MagicMock()
    mock_phoenix.evals = mock_evals

    with (
        patch("mlflow.genai.scorers.phoenix.utils.check_phoenix_installed"),
        patch.dict("sys.modules", {"phoenix": mock_phoenix, "phoenix.evals": mock_evals}),
    ):
        from mlflow.genai.scorers.phoenix.registry import get_evaluator_class

        result = get_evaluator_class(metric_name)

        # Verify we got back the mocked evaluator class
        assert result is getattr(mock_evals, evaluator_name)


def test_get_evaluator_class_invalid_metric():
    mock_evals = MagicMock()
    mock_phoenix = MagicMock()
    mock_phoenix.evals = mock_evals

    with (
        patch("mlflow.genai.scorers.phoenix.utils.check_phoenix_installed"),
        patch.dict("sys.modules", {"phoenix": mock_phoenix, "phoenix.evals": mock_evals}),
    ):
        from mlflow.genai.scorers.phoenix.registry import get_evaluator_class

        with pytest.raises(MlflowException, match="Unknown Phoenix metric"):
            get_evaluator_class("InvalidMetric")


def test_get_metric_config_hallucination():
    from mlflow.genai.scorers.phoenix.registry import get_metric_config

    config = get_metric_config("Hallucination")
    assert config["evaluator_class"] == "HallucinationEvaluator"
    assert config["positive_label"] == "factual"
    assert config["negative_label"] == "hallucinated"
    assert "input" in config["required_fields"]
    assert "output" in config["required_fields"]
    assert "reference" in config["required_fields"]


def test_get_metric_config_toxicity():
    from mlflow.genai.scorers.phoenix.registry import get_metric_config

    config = get_metric_config("Toxicity")
    assert config["evaluator_class"] == "ToxicityEvaluator"
    assert config["positive_label"] == "non-toxic"
    assert config["negative_label"] == "toxic"
    assert "input" in config["required_fields"]


def test_get_metric_config_invalid_metric():
    from mlflow.genai.scorers.phoenix.registry import get_metric_config

    with pytest.raises(MlflowException, match="Unknown Phoenix metric"):
        get_metric_config("InvalidMetric")
