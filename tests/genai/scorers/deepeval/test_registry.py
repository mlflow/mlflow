from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.deepeval.registry import get_metric_class, is_deterministic_metric


def test_get_metric_class_returns_valid_class():
    metric_class = get_metric_class("AnswerRelevancy")
    assert metric_class.__name__ == "AnswerRelevancyMetric"


def test_get_metric_class_raises_error_for_invalid_name():
    with pytest.raises(MlflowException, match="Unknown metric: 'InvalidMetric'"):
        get_metric_class("InvalidMetric")


def test_get_metric_class_dynamic_import_success():
    mock_metric_class = mock.MagicMock()
    mock_metric_class.__name__ = "NewMetricMetric"

    mock_module = mock.MagicMock()
    mock_module.NewMetricMetric = mock_metric_class

    with mock.patch.dict("sys.modules", {"deepeval.metrics": mock_module}):
        result = get_metric_class("NewMetric")
        assert result is mock_metric_class


def test_is_deterministic_metric_returns_false_for_unknown():
    # Unknown metrics should default to non-deterministic (requires model)
    assert not is_deterministic_metric("UnknownMetric")


def test_is_deterministic_metric_returns_true_for_deterministic():
    assert is_deterministic_metric("ExactMatch")
    assert is_deterministic_metric("PatternMatch")


def test_is_deterministic_metric_returns_false_for_non_deterministic():
    assert not is_deterministic_metric("AnswerRelevancy")
    assert not is_deterministic_metric("Faithfulness")
