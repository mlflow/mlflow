import warnings
from contextlib import contextmanager
from unittest.mock import patch

import pandas as pd
import pytest

import mlflow

_TEST_DATA = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


@pytest.mark.parametrize("tracking_uri", ["databricks", "http://localhost:5000"])
def test_global_evaluate_warn_in_tracking_uri(tracking_uri):
    with patch("mlflow.get_tracking_uri", return_value=tracking_uri):
        with pytest.warns(FutureWarning, match="The `mlflow.evaluate` API has been deprecated"):
            mlflow.evaluate(
                data=_TEST_DATA,
                model=lambda x: x["x"] * 2,
                extra_metrics=[mlflow.metrics.latency()],
            )


def test_global_evaluate_hints_for_genai_model_type():
    with (
        patch(
            "mlflow.agent.hint.maybe_append_agent_hint",
            side_effect=lambda _issue_id, message: f"{message}\nAgent skill hint.",
        ) as append_hint,
        # Patch the implementation because this test only covers dispatch from
        # the deprecated public entry point.
        patch("mlflow.models.evaluation.deprecated.model_evaluate"),
        pytest.warns(FutureWarning, match="(?s)deprecated.*Agent skill hint"),
    ):
        mlflow.evaluate(data=_TEST_DATA, model_type="question-answering")

    append_hint.assert_called_once()


@contextmanager
def no_future_warning():
    with warnings.catch_warnings():
        # Translate future warning into an exception
        warnings.simplefilter("error", FutureWarning)
        yield


@pytest.mark.parametrize("tracking_uri", ["databricks", "sqlite://"])
def test_models_evaluate_does_not_warn(tracking_uri):
    with patch("mlflow.get_tracking_uri", return_value=tracking_uri):
        with no_future_warning():
            mlflow.models.evaluate(
                data=_TEST_DATA,
                model=lambda x: x["x"] * 2,
                extra_metrics=[mlflow.metrics.mse()],
            )
