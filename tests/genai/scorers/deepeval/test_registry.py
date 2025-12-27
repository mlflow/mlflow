import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.deepeval.registry import get_metric_class


def test_get_metric_class_returns_valid_class():
    metric_class = get_metric_class("AnswerRelevancy")
    assert metric_class.__name__ == "AnswerRelevancyMetric"


def test_get_metric_class_raises_error_for_invalid_name():
    with pytest.raises(MlflowException, match="Unknown metric: 'InvalidMetric'"):
        get_metric_class("InvalidMetric")
