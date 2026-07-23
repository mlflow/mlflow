"""
Tests for CatBoost autologging with CatBoostRegressor.

These tests verify that mlflow.catboost.autolog() correctly logs:
- Model parameters (iterations, loss_function, depth, etc.)
- Per-iteration metrics (RMSE on train and validation sets)
- Trained model artifacts
- Dataset inputs (training and evaluation data)

See test_catboost_classifier_autolog.py for CatBoostClassifier tests.
"""

from typing import Any, Generator

import catboost as cb
import numpy as np
import pandas as pd
import pytest
from pandas.core.frame import DataFrame
from pandas.core.series import Series

import mlflow
import mlflow.catboost


def get_regression_data() -> tuple[DataFrame, Series]:
    """Return simple regression data for CatBoostRegressor tests."""
    np.random.seed(seed=42)
    X = pd.DataFrame(data=np.random.randn(100, 3), columns=["f1", "f2", "f3"])  # type: ignore
    y = pd.Series(data=np.random.randn(100))
    return X, y


REGRESSOR_PARAMS = {
    "allow_writing_files": False,
    "iterations": 10,
    "loss_function": "RMSE",
}


@pytest.fixture(autouse=True)
def reset_autolog() -> Generator[None, Any, None]:
    """Ensure autologging is disabled after each test to avoid cross-test interference."""
    yield
    mlflow.catboost.autolog(disable=True)


def test_regressor_autolog_logs_params() -> None:
    """Verify that model parameters are logged for CatBoostRegressor."""
    mlflow.catboost.autolog()
    X, y = get_regression_data()

    with mlflow.start_run() as run:
        model = cb.CatBoostRegressor(**REGRESSOR_PARAMS)
        model.fit(X, y)

    run_data = mlflow.get_run(run.info.run_id).data
    assert "iterations" in run_data.params
    assert run_data.params["iterations"] == "10"
    assert "loss_function" in run_data.params
    assert run_data.params["loss_function"] == "RMSE"


def test_regressor_autolog_logs_metrics_with_eval_set() -> None:
    """Verify that per-iteration RMSE metrics are logged for regressor with eval_set."""
    mlflow.catboost.autolog()
    X, y = get_regression_data()
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    with mlflow.start_run() as run:
        model = cb.CatBoostRegressor(**REGRESSOR_PARAMS)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

    client = mlflow.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, key="learn-RMSE")
    assert len(metric_history) == REGRESSOR_PARAMS["iterations"]

    val_history = client.get_metric_history(run.info.run_id, key="validation-RMSE")
    assert len(val_history) == REGRESSOR_PARAMS["iterations"]


def test_regressor_autolog_logs_model() -> None:
    """Verify that CatBoostRegressor model is logged and loadable."""
    mlflow.catboost.autolog(log_models=True)
    X, y = get_regression_data()

    with mlflow.start_run() as run:
        model = cb.CatBoostRegressor(**REGRESSOR_PARAMS)
        model.fit(X, y)

    model_uri = f"runs:/{run.info.run_id}/model"
    loaded_model = mlflow.catboost.load_model(model_uri)
    assert isinstance(loaded_model, cb.CatBoostRegressor)

    original_preds = model.predict(data=X)
    loaded_preds = loaded_model.predict(data=X)
    np.testing.assert_array_almost_equal(actual=original_preds, desired=loaded_preds)


def test_regressor_autolog_logs_datasets() -> None:
    """Verify that training and eval datasets are logged for CatBoostRegressor."""
    mlflow.catboost.autolog(log_datasets=True)
    X, y = get_regression_data()
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    with mlflow.start_run() as run:
        model = cb.CatBoostRegressor(**REGRESSOR_PARAMS)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

    run_data = mlflow.get_run(run.info.run_id)
    dataset_inputs = run_data.inputs.dataset_inputs
    assert len(dataset_inputs) >= 1
