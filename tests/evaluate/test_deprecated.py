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


@patch("mlflow.models.evaluation.deprecated.warnings")
def test_global_evaluate_deprecation(mock_warnings, tracking_uri):
    data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    # Issue a warning when mlflow.evaluate is used in Databricks
    mlflow.evaluate(
        data=data,
        model=lambda x: x["x"] * 2,
        extra_metrics=[mlflow.metrics.latency()],
    )
    if tracking_uri.startswith("databricks"):
        mock_warnings.warn.assert_called_once()
        assert mock_warnings.warn.call_args[0][0].startswith(
            "The `mlflow.evaluate` API is deprecated"
        )
    else:
        mock_warnings.assert_not_called()
    mock_warnings.reset_mock()

    # No warning when mlflow.models.evaluate is used
    mlflow.models.evaluate(
        data=data,
        model=lambda x: x["x"] * 2,
        extra_metrics=[mlflow.metrics.latency()],
    )
    mock_warnings.assert_not_called()


def test_databricks_agent_evaluate_deprecation():
    data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    # Warning should be shown when "databricks-agent" model type is used
    with (
        patch("mlflow.models.evaluation.base.warnings") as mock_warnings,
        patch("mlflow.models.evaluation.base._evaluate") as mock_evaluate_impl,
    ):
        mlflow.models.evaluate(
            data=data,
            model=lambda x: x["x"] * 2,
            model_type="databricks-agent",
            extra_metrics=[mlflow.metrics.latency()],
        )
    mock_warnings.warn.assert_called_once()
    assert mock_warnings.warn.call_args[0][0].startswith(
        "'databricks-agent' model type is deprecated"
    )
    mock_evaluate_impl.assert_called_once()
