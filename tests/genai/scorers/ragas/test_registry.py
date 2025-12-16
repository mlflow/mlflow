import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.ragas.registry import (
    get_metric_class,
    is_deterministic_metric,
)


def test_get_metric_class_returns_valid_class():
    metric_class = get_metric_class("Faithfulness")
    assert metric_class.__name__ == "Faithfulness"


def test_get_metric_class_raises_error_for_invalid_name():
    with pytest.raises(MlflowException, match="Unknown metric: 'InvalidMetric'"):
        get_metric_class("InvalidMetric")


@pytest.mark.parametrize(
    ("metric_name", "expected"),
    [
        ("ExactMatch", True),
        ("BleuScore", True),
        ("RougeScore", True),
        ("NonLLMStringSimilarity", True),
        ("StringPresence", True),
        ("ChrfScore", True),
        ("Faithfulness", False),
        ("ContextPrecision", False),
    ],
)
def test_is_deterministic_metric(metric_name, expected):
    assert is_deterministic_metric(metric_name) is expected
