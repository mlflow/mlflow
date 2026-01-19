import pytest

from mlflow.exceptions import MlflowException

phoenix_evals = pytest.importorskip("phoenix.evals")


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
    from mlflow.genai.scorers.phoenix.registry import get_evaluator_class

    result = get_evaluator_class(metric_name)
    expected = getattr(phoenix_evals, evaluator_name)
    assert result is expected


def test_get_evaluator_class_invalid_metric():
    from mlflow.genai.scorers.phoenix.registry import get_evaluator_class

    with pytest.raises(MlflowException, match="Unknown Phoenix metric"):
        get_evaluator_class("InvalidMetric")
