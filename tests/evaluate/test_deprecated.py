from unittest.mock import patch

import pandas as pd
import pytest

import mlflow


@pytest.fixture(params=["databricks", "local"])
def tracking_uri(request):
    if request.param == "databricks":
        uri = "databricks"
        with patch("mlflow.get_tracking_uri", return_value=uri):
            yield uri
    else:
        uri = "sqlite://"
        with patch("mlflow.get_tracking_uri", return_value=uri):
            yield uri


@patch("mlflow.models.evaluation.deprecated._logger")
def test_global_evaluate_deprecation(mock_logger, tracking_uri):
    data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    # Issue a warning when mlflow.evaluate is used in Databricks
    mlflow.evaluate(
        data=data,
        model=lambda x: x["x"] * 2,
        extra_metrics=[mlflow.metrics.latency()],
    )
    if tracking_uri.startswith("databricks"):
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args[0][0].startswith(
            "The `mlflow.evaluate` API has been deprecated"
        )
    else:
        mock_logger.warning.assert_not_called()
    mock_logger.reset_mock()

    # No warning when mlflow.models.evaluate is used
    mlflow.models.evaluate(
        data=data,
        model=lambda x: x["x"] * 2,
        extra_metrics=[mlflow.metrics.latency()],
    )
    mock_logger.warning.assert_not_called()
