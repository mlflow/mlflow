"""
Tests for CatBoost autologging with CatBoostClassifier.

These tests verify that mlflow.catboost.autolog() correctly logs:
- Model parameters (iterations, depth, learning_rate, etc.)
- Per-iteration metrics (MultiClass/Logloss on train and validation sets)
- Trained model artifacts with signatures and input examples
- Dataset inputs (training and evaluation data, including Pool objects)
- Preservation of user-provided callbacks

See test_catboost_regressor_autolog.py for CatBoostRegressor tests.
"""

import catboost as cb
import numpy as np
import pandas as pd
import pytest
from sklearn import datasets

import mlflow
import mlflow.catboost


def get_iris():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = pd.Series(iris.target)
    return X, y


MODEL_PARAMS = {"allow_writing_files": False, "iterations": 10}


@pytest.fixture(autouse=True)
def reset_autolog():
    """Ensure autologging is disabled after each test to avoid cross-test interference."""
    yield
    mlflow.catboost.autolog(disable=True)


def test_autolog_logs_params():
    """Verify that model parameters (iterations, depth, etc.) are logged after training.

    CatBoost resolves many default parameters internally, so we call
    model.get_all_params() after fit() to capture the full resolved config.
    """
    mlflow.catboost.autolog()
    X, y = get_iris()

    with mlflow.start_run() as run:
        model = cb.CatBoostClassifier(**MODEL_PARAMS)
        model.fit(X, y)

    run_data = mlflow.get_run(run.info.run_id).data
    assert "iterations" in run_data.params
    assert run_data.params["iterations"] == "10"
    assert "depth" in run_data.params


def test_autolog_logs_metrics_with_eval_set():
    """Verify that per-iteration metrics are logged for both train and validation sets.

    When eval_set is provided, CatBoost computes metrics on both the training
    data ('learn') and validation data ('validation') at each iteration. The
    autolog callback should capture all of these as MLflow step-metrics.
    """
    mlflow.catboost.autolog()
    X, y = get_iris()
    X_train, X_val = X[:120], X[120:]
    y_train, y_val = y[:120], y[120:]

    with mlflow.start_run() as run:
        model = cb.CatBoostClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

    client = mlflow.MlflowClient()
    # Iris is a 3-class dataset, so CatBoost uses MultiClass loss
    metric_history = client.get_metric_history(run.info.run_id, "learn-MultiClass")
    assert len(metric_history) == MODEL_PARAMS["iterations"]

    val_history = client.get_metric_history(run.info.run_id, "validation-MultiClass")
    assert len(val_history) == MODEL_PARAMS["iterations"]


def test_autolog_logs_metrics_without_eval_set():
    """Verify that training metrics are still logged when no eval_set is provided.

    Even without a validation set, CatBoost computes training metrics ('learn')
    at each iteration, and the callback should capture them.
    """
    mlflow.catboost.autolog()
    X, y = get_iris()

    with mlflow.start_run() as run:
        model = cb.CatBoostClassifier(**MODEL_PARAMS)
        model.fit(X, y)

    client = mlflow.MlflowClient()
    # Iris is a 3-class dataset, so CatBoost uses MultiClass loss
    metric_history = client.get_metric_history(run.info.run_id, "learn-MultiClass")
    assert len(metric_history) == MODEL_PARAMS["iterations"]


def test_autolog_logs_model():
    """Verify that the trained model is logged and can be loaded back with identical predictions.

    The logged model should be a valid CatBoostClassifier that produces the same
    predictions as the original model.
    """
    mlflow.catboost.autolog(log_models=True)
    X, y = get_iris()

    with mlflow.start_run() as run:
        model = cb.CatBoostClassifier(**MODEL_PARAMS)
        model.fit(X, y)

    model_uri = f"runs:/{run.info.run_id}/model"
    loaded_model = mlflow.catboost.load_model(model_uri)
    assert isinstance(loaded_model, cb.CatBoostClassifier)

    original_preds = model.predict(X)
    loaded_preds = loaded_model.predict(X)
    np.testing.assert_array_equal(original_preds, loaded_preds)


def test_autolog_does_not_log_model_when_disabled():
    """Verify that no model artifact is logged when log_models=False."""
    mlflow.catboost.autolog(log_models=False)
    X, y = get_iris()

    with mlflow.start_run() as run:
        model = cb.CatBoostClassifier(**MODEL_PARAMS)
        model.fit(X, y)

    client = mlflow.MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "model" not in artifacts


def test_autolog_logs_datasets():
    """Verify that training and evaluation datasets are logged as MLflow dataset inputs.

    When X is a pandas DataFrame (not a Pool), the data should be logged using
    MLflow's dataset tracking (from_pandas / from_numpy).
    """
    mlflow.catboost.autolog(log_datasets=True)
    X, y = get_iris()
    X_train, X_val = X[:120], X[120:]
    y_train, y_val = y[:120], y[120:]

    with mlflow.start_run() as run:
        model = cb.CatBoostClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

    run_data = mlflow.get_run(run.info.run_id)
    dataset_inputs = run_data.inputs.dataset_inputs
    assert len(dataset_inputs) >= 1


def test_autolog_logs_pool_as_dataset():
    """Verify that when X is a catboost.Pool, features and labels are extracted and logged as a dataset.

    Pool.get_features() and Pool.get_label() are used to extract the data, which is then
    converted to a DataFrame with columns 'feature_0', 'feature_1', ..., 'label' and logged.
    """
    mlflow.catboost.autolog(log_datasets=True)
    X, y = get_iris()
    train_pool = cb.Pool(X, y)

    with mlflow.start_run() as run:
        model = cb.CatBoostClassifier(**MODEL_PARAMS)
        model.fit(train_pool)

    run_data = mlflow.get_run(run.info.run_id)
    dataset_inputs = run_data.inputs.dataset_inputs
    assert len(dataset_inputs) >= 1


def test_autolog_disable():
    """Verify that disable=True completely prevents any params or metrics from being logged."""
    mlflow.catboost.autolog(disable=True)
    X, y = get_iris()

    with mlflow.start_run() as run:
        model = cb.CatBoostClassifier(**MODEL_PARAMS)
        model.fit(X, y)

    run_data = mlflow.get_run(run.info.run_id).data
    assert len(run_data.params) == 0
    assert len(run_data.metrics) == 0


def test_autolog_preserves_user_callbacks():
    """Verify that user-provided callbacks are not overwritten by the autolog callback.

    The autolog callback is appended to the user's callback list, so both should
    fire on each iteration. This test confirms the user callback runs for every
    iteration AND that MLflow metrics are still logged.
    """
    mlflow.catboost.autolog()
    X, y = get_iris()

    callback_called = []

    class UserCallback:
        def after_iteration(self, info):
            callback_called.append(info.iteration)
            return True

    with mlflow.start_run() as run:
        model = cb.CatBoostClassifier(**MODEL_PARAMS)
        model.fit(X, y, callbacks=[UserCallback()])

    assert len(callback_called) == MODEL_PARAMS["iterations"]

    client = mlflow.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "learn-MultiClass")
    assert len(metric_history) == MODEL_PARAMS["iterations"]


def test_autolog_creates_run_if_none_active():
    """Verify that autologging creates a managed run when no run is explicitly started.

    The safe_patch mechanism with manage_run=True should automatically create
    and close a run around the fit() call.
    """
    mlflow.catboost.autolog()
    X, y = get_iris()

    model = cb.CatBoostClassifier(**MODEL_PARAMS)
    model.fit(X, y)

    runs = mlflow.search_runs()
    assert len(runs) == 1
    assert "params.iterations" in runs.iloc[0].to_dict()


def test_autolog_with_numpy_data():
    """Verify that autologging works when X is a numpy array instead of a DataFrame."""
    mlflow.catboost.autolog()
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target

    with mlflow.start_run() as run:
        model = cb.CatBoostClassifier(**MODEL_PARAMS)
        model.fit(X, y)

    run_data = mlflow.get_run(run.info.run_id).data
    assert "iterations" in run_data.params


def test_autolog_with_input_example():
    """Verify that an input example and model signature are logged when enabled.

    When log_input_examples=True and log_model_signatures=True (default),
    the first few rows of X are captured and used to infer the model signature.
    """
    mlflow.catboost.autolog(log_models=True, log_input_examples=True)
    X, y = get_iris()

    with mlflow.start_run() as run:
        model = cb.CatBoostClassifier(**MODEL_PARAMS)
        model.fit(X, y)

    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow_model = mlflow.models.get_model_info(model_uri)
    assert mlflow_model.signature is not None
