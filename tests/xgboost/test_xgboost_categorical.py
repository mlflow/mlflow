import pytest
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
from mlflow.models import infer_signature
from mlflow.pyfunc import PyFuncModel
from mlflow.types.schema import DataType, ColSpec, Schema
from mlflow.exceptions import MlflowException
import os


@pytest.fixture(scope="module")
def xgb_model_with_categorical():
    """Fixture for a simple XGBoost model trained with a categorical feature."""
    # Create a DataFrame with a categorical column
    data = {
        "feature1": [10, 20, 30, 40, 50],
        "categorical_feature": pd.Series(["A", "B", "A", "C", "B"], dtype="category"),
        "target": [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    X = df[["feature1", "categorical_feature"]]
    y = df["target"]

    # Convert categorical to codes for XGBoost if not using native categorical support
    # For simplicity, we'll use one-hot encoding or label encoding if XGBoost version
    # doesn't natively support pandas categorical.
    # However, with recent XGBoost versions, it can handle pandas categorical directly
    # if `enable_categorical=True` is set in DMatrix or model parameters.
    # For testing schema enforcement, we just need the DataFrame to have the dtype.

    # Simple XGBoost model (Booster API)
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "enable_categorical": True}
    model = xgb.train(params, dtrain, num_boost_round=5)
    return model, X, y


@pytest.fixture(scope="module")
def xgb_sklearn_model_with_categorical():
    """Fixture for an XGBoost scikit-learn model trained with a categorical feature."""
    data = {
        "feature1": [10, 20, 30, 40, 50],
        "categorical_feature": pd.Series(["A", "B", "A", "C", "B"], dtype="category"),
        "target": [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    X = df[["feature1", "categorical_feature"]]
    y = df["target"]

    # XGBClassifier (scikit-learn API)
    model = xgb.XGBClassifier(objective="binary:logistic", enable_categorical=True, random_state=42)
    model.fit(X, y)
    return model, X, y


@pytest.mark.categorical
def test_log_and_load_xgboost_model_with_categorical_feature(tmp_path, xgb_model_with_categorical):
    """
    Tests logging and loading an XGBoost model with a categorical feature,
    and verifies schema inference and pyfunc prediction.
    """
    model, X, y = xgb_model_with_categorical
    model_path = os.path.join(tmp_path, "xgb_cat_model")

    # Log the model
    with mlflow.start_run() as run:
        # Pass enable_categorical=True when creating DMatrix for prediction during signature inference
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            input_example=X.head(2),
            signature=infer_signature(X, model.predict(xgb.DMatrix(X, enable_categorical=True))),
        )
        model_uri = f"runs:/{run.info.run_id}/model"

    # Load the model as pyfunc
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    assert isinstance(loaded_pyfunc_model, PyFuncModel)

    # Verify the signature
    signature = loaded_pyfunc_model.metadata.signature
    assert signature is not None
    assert len(signature.inputs.inputs) == 2
    assert signature.inputs.inputs[0].name == "feature1"
    assert signature.inputs.inputs[0].type == DataType.long
    assert signature.inputs.inputs[1].name == "categorical_feature"
    assert signature.inputs.inputs[1].type == DataType.category

    # Prepare inference data
    inference_data = pd.DataFrame({
        "feature1": [15, 25],
        "categorical_feature": pd.Series(["A", "C"], dtype="category")
    })

    # Perform prediction
    predictions = loaded_pyfunc_model.predict(inference_data)
    assert predictions is not None
    assert len(predictions) == 2


@pytest.mark.categorical
def test_log_and_load_xgboost_sklearn_model_with_categorical_feature(tmp_path, xgb_sklearn_model_with_categorical):
    """
    Tests logging and loading an XGBoost scikit-learn model with a categorical feature,
    and verifies schema inference and pyfunc prediction.
    """
    model, X, y = xgb_sklearn_model_with_categorical
    model_path = os.path.join(tmp_path, "xgb_sklearn_cat_model")

    # Log the model
    with mlflow.start_run() as run:
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            input_example=X.head(2),
            signature=infer_signature(X, model.predict(X)),
        )
        model_uri = f"runs:/{run.info.run_id}/model"

    # Load the model as pyfunc
    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    assert isinstance(loaded_pyfunc_model, PyFuncModel)

    # Verify the signature
    signature = loaded_pyfunc_model.metadata.signature
    assert signature is not None
    assert len(signature.inputs.inputs) == 2
    assert signature.inputs.inputs[0].name == "feature1"
    assert signature.inputs.inputs[0].type == DataType.long
    assert signature.inputs.inputs[1].name == "categorical_feature"
    assert signature.inputs.inputs[1].type == DataType.category

    # Prepare inference data
    inference_data = pd.DataFrame({
        "feature1": [15, 25],
        "categorical_feature": pd.Series(["A", "C"], dtype="category")
    })

    # Perform prediction
    predictions = loaded_pyfunc_model.predict(inference_data)
    assert predictions is not None
    assert len(predictions) == 2


@pytest.mark.categorical
def test_pyfunc_predict_with_unseen_categories(tmp_path, xgb_model_with_categorical):
    """
    Tests pyfunc prediction with unseen categories in the input data.
    Expects pandas' default behavior (adding new categories to the CategoricalDtype).
    """
    model, X, y = xgb_model_with_categorical
    model_path = os.path.join(tmp_path, "xgb_unseen_cat_model")

    with mlflow.start_run() as run:
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            input_example=X.head(2),
            signature=infer_signature(X, model.predict(xgb.DMatrix(X, enable_categorical=True))),
        )
        model_uri = f"runs:/{run.info.run_id}/model"

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri)

    # Inference data with an unseen category 'D'
    inference_data = pd.DataFrame({
        "feature1": [100],
        "categorical_feature": pd.Series(["D"], dtype=object) # Use object dtype to simulate raw input
    })

    predictions = loaded_pyfunc_model.predict(inference_data)
    assert predictions is not None
    assert len(predictions) == 1

    # Verify that the categorical column in the enforced input (internal to pyfunc)
    # now contains 'D' in its categories. This is pandas' default behavior.
    # To inspect this, we'd need to mock or inspect internal calls, but for now,
    # successful prediction implies the conversion happened.
    # A more robust test would involve a custom pyfunc model that logs the dtype of its input.


@pytest.mark.categorical
def test_pyfunc_predict_with_categorical_nan_values(tmp_path, xgb_model_with_categorical):
    """
    Tests pyfunc prediction with NaN values in a categorical column.
    """
    model, X, y = xgb_model_with_categorical
    model_path = os.path.join(tmp_path, "xgb_nan_cat_model")

    # Add NaN to the original categorical feature for logging
    X_with_nan = X.copy()
    X_with_nan.loc[0, "categorical_feature"] = np.nan
    X_with_nan["categorical_feature"] = X_with_nan["categorical_feature"].astype("category")

    with mlflow.start_run() as run:
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            input_example=X_with_nan.head(2),
            signature=infer_signature(X_with_nan, model.predict(xgb.DMatrix(X_with_nan, enable_categorical=True))),
        )
        model_uri = f"runs:/{run.info.run_id}/model"

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri)

    # Inference data with NaN
    inference_data = pd.DataFrame({
        "feature1": [150],
        "categorical_feature": pd.Series([np.nan], dtype="category")
    })

    predictions = loaded_pyfunc_model.predict(inference_data)
    assert predictions is not None
    assert len(predictions) == 1


@pytest.mark.categorical
def test_pyfunc_predict_with_mixed_data_types(tmp_path, xgb_model_with_categorical):
    """
    Tests pyfunc prediction with mixed numerical and categorical data types.
    """
    model, X, y = xgb_model_with_categorical
    model_path = os.path.join(tmp_path, "xgb_mixed_model")

    with mlflow.start_run() as run:
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            input_example=X.head(2),
            signature=infer_signature(X, model.predict(xgb.DMatrix(X, enable_categorical=True))),
        )
        model_uri = f"runs:/{run.info.run_id}/model"

    loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri)

    inference_data = pd.DataFrame({
        "feature1": [55, 65],
        "categorical_feature": pd.Series(["A", "B"], dtype="category")
    })

    predictions = loaded_pyfunc_model.predict(inference_data)
    assert predictions is not None
    assert len(predictions) == 2

