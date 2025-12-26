import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.trulens.registry import get_feedback_method_name, get_metric_config


@pytest.mark.parametrize(
    ("metric_name", "expected_method"),
    [
        ("Groundedness", "groundedness_measure_with_cot_reasons"),
        ("ContextRelevance", "context_relevance_with_cot_reasons"),
        ("AnswerRelevance", "relevance_with_cot_reasons"),
        ("Coherence", "coherence_with_cot_reasons"),
    ],
)
def test_get_feedback_method_name(metric_name, expected_method):
    result = get_feedback_method_name(metric_name)
    assert result == expected_method


def test_get_feedback_method_name_invalid_metric():
    with pytest.raises(MlflowException, match="Unknown TruLens metric"):
        get_feedback_method_name("InvalidMetric")


def test_get_metric_config_groundedness():
    config = get_metric_config("Groundedness")
    assert config["method"] == "groundedness_measure_with_cot_reasons"
    assert config["args"] == ["source", "statement"]
    assert "description" in config


def test_get_metric_config_context_relevance():
    config = get_metric_config("ContextRelevance")
    assert config["method"] == "context_relevance_with_cot_reasons"
    assert config["args"] == ["question", "context"]


def test_get_metric_config_answer_relevance():
    config = get_metric_config("AnswerRelevance")
    assert config["method"] == "relevance_with_cot_reasons"
    assert config["args"] == ["prompt", "response"]


def test_get_metric_config_coherence():
    config = get_metric_config("Coherence")
    assert config["method"] == "coherence_with_cot_reasons"
    assert config["args"] == ["text"]


def test_get_metric_config_invalid_metric():
    with pytest.raises(MlflowException, match="Unknown TruLens metric"):
        get_metric_config("InvalidMetric")
