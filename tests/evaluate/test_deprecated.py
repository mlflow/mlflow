from unittest.mock import patch

import pandas as pd
import pytest

import mlflow

_TEST_DATA = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


@pytest.fixture
def mock_warnings():
    with patch("mlflow.models.evaluation.deprecated.warnings") as mock_warnings:
        yield mock_warnings


def test_global_evaluate_warn_in_databricks(mock_warnings):
    with patch("mlflow.get_tracking_uri", return_value="databricks"):
        mlflow.evaluate(
            data=_TEST_DATA,
            model=lambda x: x["x"] * 2,
            extra_metrics=[mlflow.metrics.latency()],
        )

    mock_warnings.warn.assert_called_once()
    assert mock_warnings.warn.call_args[0][0].startswith(
        "The `mlflow.evaluate` API has been deprecated"
    )


def test_global_evaluate_does_not_warn_outside_databricks(mock_warnings):
    mlflow.evaluate(
        data=_TEST_DATA,
        model=lambda x: x["x"] * 2,
        extra_metrics=[mlflow.metrics.latency()],
    )
    mock_warnings.warn.assert_not_called()


@pytest.mark.parametrize("tracking_uri", ["databricks", "sqlite://"])
def test_models_evaluate_does_not_warn(tracking_uri, mock_warnings):
    with patch("mlflow.get_tracking_uri", return_value=tracking_uri):
        mlflow.models.evaluate(
            data=_TEST_DATA,
            model=lambda x: x["x"] * 2,
            extra_metrics=[mlflow.metrics.latency()],
        )

    mock_warnings.warn.assert_not_called()
