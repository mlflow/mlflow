import warnings
from contextlib import contextmanager
from unittest.mock import patch

import pandas as pd
import pytest

import mlflow

from tests.helper_functions import validate_telemetry_record

_TEST_DATA = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


def test_global_evaluate_warn_in_databricks():
    with patch("mlflow.get_tracking_uri", return_value="databricks"):
        with pytest.warns(FutureWarning, match="The `mlflow.evaluate` API has been deprecated"):
            mlflow.evaluate(
                data=_TEST_DATA,
                model=lambda x: x["x"] * 2,
                extra_metrics=[mlflow.metrics.latency()],
            )


@contextmanager
def no_future_warning():
    with warnings.catch_warnings():
        # Translate future warning into an exception
        warnings.simplefilter("error", FutureWarning)
        yield


def test_global_evaluate_does_not_warn_outside_databricks():
    with no_future_warning():
        mlflow.evaluate(
            data=_TEST_DATA,
            model=lambda x: x["x"] * 2,
            extra_metrics=[mlflow.metrics.latency()],
        )


@pytest.mark.parametrize("tracking_uri", ["databricks", "sqlite://"])
def test_models_evaluate_does_not_warn(tracking_uri):
    with patch("mlflow.get_tracking_uri", return_value=tracking_uri):
        with no_future_warning():
            mlflow.models.evaluate(
                data=_TEST_DATA,
                model=lambda x: x["x"] * 2,
                extra_metrics=[mlflow.metrics.latency()],
            )


def test_deprecated_evaluate_sends_telemetry_record(mock_requests):
    """Test that deprecated evaluate sends telemetry records."""
    # Use non-databricks tracking URI to avoid deprecation warning
    with patch("mlflow.get_tracking_uri", return_value="sqlite://"):
        mlflow.evaluate(
            data=_TEST_DATA,
            model=lambda x: x["x"] * 2,
            extra_metrics=[mlflow.metrics.latency()],
        )

    validate_telemetry_record(mock_requests, mlflow.evaluate)
