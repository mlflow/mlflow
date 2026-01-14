import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.trulens.registry import get_feedback_method_name


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
