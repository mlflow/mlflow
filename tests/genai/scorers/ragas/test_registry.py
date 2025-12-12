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


def test_is_deterministic_metric():
    assert is_deterministic_metric("ExactMatch") is True
    assert is_deterministic_metric("BleuScore") is True
    assert is_deterministic_metric("RougeScore") is True
    assert is_deterministic_metric("Faithfulness") is False
    assert is_deterministic_metric("ContextPrecision") is False
