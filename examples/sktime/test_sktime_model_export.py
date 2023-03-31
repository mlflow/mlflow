"""Tests for sktime custom model flavor."""

import flavor
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes import convert
from sktime.datasets import load_airline, load_longley
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
import boto3
import moto
from botocore.config import Config
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.models import Model, infer_signature
from mlflow.models.utils import _read_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
import mlflow
from mlflow import pyfunc
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.exceptions import MlflowException

FH = [1, 2, 3]
COVERAGE = [0.1, 0.5, 0.9]
ALPHA = [0.1, 0.5, 0.9]
COV = False


@pytest.fixture
def model_path(tmp_path):
    """Create a temporary path to save/log model."""
    return tmp_path.joinpath("model")


@pytest.fixture(scope="module")
def mock_s3_bucket():
    """Create a mock S3 bucket using moto.

    Returns
    -------
    string with name of mock S3 bucket
    """
    with moto.mock_s3():
        bucket_name = "mock-bucket"
        my_config = Config(region_name="us-east-1")
        s3_client = boto3.client("s3", config=my_config)
        s3_client.create_bucket(Bucket=bucket_name)
        yield bucket_name


@pytest.fixture
def sktime_custom_env(tmp_path):
    """Create a conda environment and returns path to conda environment yml file."""
    conda_env = tmp_path.joinpath("conda_env.yml")
    _mlflow_conda_env(conda_env, additional_pip_deps=["sktime"])
    return conda_env


@pytest.fixture(scope="module")
def data_airline():
    """Create sample data for univariate model without exogenous regressor."""
    return load_airline()


@pytest.fixture(scope="module")
def data_longley():
    """Create sample data for univariate model with exogenous regressor."""
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    return y_train, y_test, X_train, X_test


@pytest.fixture(scope="module")
def auto_arima_model(data_airline):
    """Create instance of fitted auto arima model."""
    return AutoARIMA(sp=12, d=0, max_p=2, max_q=2, suppress_warnings=True).fit(data_airline)


@pytest.fixture(scope="module")
def naive_forecaster_model_with_regressor(data_longley):
    """Create instance of fitted naive forecaster model."""
    y_train, _, X_train, _ = data_longley
    model = NaiveForecaster()
    return model.fit(y_train, X_train)


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_auto_arima_model_save_and_load(auto_arima_model, model_path, serialization_format):
    """Test saving and loading of native sktime auto_arima_model."""
    flavor.save_model(
        sktime_model=auto_arima_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_model = flavor.load_model(
        model_uri=model_path,
    )

    np.testing.assert_array_equal(auto_arima_model.predict(fh=FH), loaded_model.predict(fh=FH))


@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_auto_arima_model_pyfunc_output(auto_arima_model, model_path, serialization_format):
    """Test auto arima prediction of loaded pyfunc model."""
    flavor.save_model(
        sktime_model=auto_arima_model,
        path=model_path,
        serialization_format=serialization_format,
    )
    loaded_pyfunc = flavor.pyfunc.load_model(model_uri=model_path)

    model_predict = auto_arima_model.predict(fh=FH)
    predict_conf = pd.DataFrame([{"fh": FH, "predict_method": "predict"}])
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)
    np.testing.assert_array_equal(model_predict, pyfunc_predict)

    model_predict_interval = auto_arima_model.predict_interval(fh=FH, coverage=COVERAGE)
    predict_interval_conf = pd.DataFrame(
        [
            {
                "fh": FH,
                "predict_method": "predict_interval",
                "coverage": COVERAGE,
            }
        ]
    )
    pyfunc_predict_interval = loaded_pyfunc.predict(predict_interval_conf)
    np.testing.assert_array_equal(model_predict_interval.values, pyfunc_predict_interval.values)

    model_predict_quantiles = auto_arima_model.predict_quantiles(fh=FH, alpha=ALPHA)
    predict_quantiles_conf = pd.DataFrame(
        [
            {
                "fh": FH,
                "predict_method": "predict_quantiles",
                "alpha": ALPHA,
            }
        ]
    )
    pyfunc_predict_quantiles = loaded_pyfunc.predict(predict_quantiles_conf)
    np.testing.assert_array_equal(model_predict_quantiles.values, pyfunc_predict_quantiles.values)

    model_predict_var = auto_arima_model.predict_var(fh=FH, cov=COV)
    predict_var_conf = pd.DataFrame([{"fh": FH, "predict_method": "predict_var", "cov": COV}])
    pyfunc_predict_var = loaded_pyfunc.predict(predict_var_conf)
    np.testing.assert_array_equal(model_predict_var.values, pyfunc_predict_var.values)


def test_naive_forecaster_model_with_regressor_pyfunc_output(
    naive_forecaster_model_with_regressor, model_path, data_longley
):
    """Test naive forecaster prediction of loaded pyfunc model."""
    _, _, _, X_test = data_longley

    flavor.save_model(sktime_model=naive_forecaster_model_with_regressor, path=model_path)
    loaded_pyfunc = flavor.pyfunc.load_model(model_uri=model_path)

    X_test_array = convert(X_test, "pd.DataFrame", "np.ndarray")

    model_predict = naive_forecaster_model_with_regressor.predict(fh=FH, X=X_test)
    predict_conf = pd.DataFrame([{"fh": FH, "predict_method": "predict", "X": X_test_array}])
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)
    np.testing.assert_array_equal(model_predict, pyfunc_predict)

    model_predict_interval = naive_forecaster_model_with_regressor.predict_interval(
        fh=FH, coverage=COVERAGE, X=X_test
    )
    predict_interval_conf = pd.DataFrame(
        [
            {
                "fh": FH,
                "predict_method": "predict_interval",
                "coverage": COVERAGE,
                "X": X_test_array,
            }
        ]
    )
    pyfunc_predict_interval = loaded_pyfunc.predict(predict_interval_conf)
    np.testing.assert_array_equal(model_predict_interval.values, pyfunc_predict_interval.values)

    model_predict_quantiles = naive_forecaster_model_with_regressor.predict_quantiles(
        fh=FH, alpha=ALPHA, X=X_test
    )
    predict_quantiles_conf = pd.DataFrame(
        [
            {
                "fh": FH,
                "predict_method": "predict_quantiles",
                "alpha": ALPHA,
                "X": X_test_array,
            }
        ]
    )
    pyfunc_predict_quantiles = loaded_pyfunc.predict(predict_quantiles_conf)
    np.testing.assert_array_equal(model_predict_quantiles.values, pyfunc_predict_quantiles.values)

    model_predict_var = naive_forecaster_model_with_regressor.predict_var(fh=FH, cov=COV, X=X_test)
    predict_var_conf = pd.DataFrame(
        [
            {
                "fh": FH,
                "predict_method": "predict_var",
                "cov": COV,
                "X": X_test_array,
            }
        ]
    )
    pyfunc_predict_var = loaded_pyfunc.predict(predict_var_conf)
    np.testing.assert_array_equal(model_predict_var.values, pyfunc_predict_var.values)


@pytest.mark.parametrize("use_signature", [True, False])
@pytest.mark.parametrize("use_example", [True, False])
def test_signature_and_examples_saved_correctly(
    auto_arima_model, data_airline, model_path, use_signature, use_example
):
    """Test saving of mlflow signature and example for native sktime predict method."""
    # Note: Signature inference fails on native model predict_interval/predict_quantiles
    prediction = auto_arima_model.predict(fh=FH)
    signature = infer_signature(data_airline, prediction) if use_signature else None
    example = pd.DataFrame(data_airline[0:5].copy(deep=False)) if use_example else None
    flavor.save_model(auto_arima_model, path=model_path, signature=signature, input_example=example)
    mlflow_model = Model.load(model_path)
    assert signature == mlflow_model.signature
    if example is None:
        assert mlflow_model.saved_input_example_info is None
    else:
        r_example = _read_example(mlflow_model, model_path).copy(deep=False)
        np.testing.assert_array_equal(r_example, example)


@pytest.mark.parametrize("use_signature", [True, False])
def test_predict_var_signature_saved_correctly(
    auto_arima_model, data_airline, model_path, use_signature
):
    """Test saving of mlflow signature for native sktime predict_var method."""
    prediction = auto_arima_model.predict_var(fh=FH)
    signature = infer_signature(data_airline, prediction) if use_signature else None
    flavor.save_model(auto_arima_model, path=model_path, signature=signature)
    mlflow_model = Model.load(model_path)
    assert signature == mlflow_model.signature


@pytest.mark.parametrize("use_signature", [True, False])
@pytest.mark.parametrize("use_example", [True, False])
def test_signature_and_example_for_pyfunc_predict_inteval(
    auto_arima_model, model_path, data_airline, use_signature, use_example
):
    """Test saving of mlflow signature and example for pyfunc predict."""
    model_path_primary = model_path.joinpath("primary")
    model_path_secondary = model_path.joinpath("secondary")
    flavor.save_model(sktime_model=auto_arima_model, path=model_path_primary)
    loaded_pyfunc = flavor.pyfunc.load_model(model_uri=model_path_primary)
    predict_conf = pd.DataFrame(
        [
            {
                "fh": FH,
                "predict_method": "predict_interval",
                "coverage": COVERAGE,
            }
        ]
    )
    forecast = loaded_pyfunc.predict(predict_conf)
    signature = infer_signature(data_airline, forecast) if use_signature else None
    example = pd.DataFrame(data_airline[0:5].copy(deep=False)) if use_example else None
    flavor.save_model(
        auto_arima_model,
        path=model_path_secondary,
        signature=signature,
        input_example=example,
    )
    mlflow_model = Model.load(model_path_secondary)
    assert signature == mlflow_model.signature
    if example is None:
        assert mlflow_model.saved_input_example_info is None
    else:
        r_example = _read_example(mlflow_model, model_path_secondary).copy(deep=False)
        np.testing.assert_array_equal(r_example, example)


@pytest.mark.parametrize("use_signature", [True, False])
def test_signature_for_pyfunc_predict_quantiles(
    auto_arima_model, model_path, data_airline, use_signature
):
    """Test saving of mlflow signature for pyfunc sktime predict_quantiles method."""
    model_path_primary = model_path.joinpath("primary")
    model_path_secondary = model_path.joinpath("secondary")
    flavor.save_model(sktime_model=auto_arima_model, path=model_path_primary)
    loaded_pyfunc = flavor.pyfunc.load_model(model_uri=model_path_primary)
    predict_conf = pd.DataFrame(
        [
            {
                "fh": FH,
                "predict_method": "predict_quantiles",
                "alpha": ALPHA,
            }
        ]
    )
    forecast = loaded_pyfunc.predict(predict_conf)
    signature = infer_signature(data_airline, forecast) if use_signature else None
    flavor.save_model(auto_arima_model, path=model_path_secondary, signature=signature)
    mlflow_model = Model.load(model_path_secondary)
    assert signature == mlflow_model.signature


def test_load_from_remote_uri_succeeds(auto_arima_model, model_path, mock_s3_bucket):
    """Test loading native sktime model from mock S3 bucket."""
    flavor.save_model(sktime_model=auto_arima_model, path=model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = os.path.join(artifact_root, artifact_path)
    reloaded_sktime_model = flavor.load_model(model_uri=model_uri)

    np.testing.assert_array_equal(
        auto_arima_model.predict(fh=FH),
        reloaded_sktime_model.predict(fh=FH),
    )


@pytest.mark.parametrize("should_start_run", [True, False])
@pytest.mark.parametrize("serialization_format", ["pickle", "cloudpickle"])
def test_log_model(auto_arima_model, tmp_path, should_start_run, serialization_format):
    """Test logging and reloading sktime model."""
    try:
        if should_start_run:
            mlflow.start_run()
        artifact_path = "sktime"
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["sktime"])
        model_info = flavor.log_model(
            sktime_model=auto_arima_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            serialization_format=serialization_format,
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        assert model_info.model_uri == model_uri
        reloaded_model = flavor.load_model(
            model_uri=model_uri,
        )
        np.testing.assert_array_equal(auto_arima_model.predict(), reloaded_model.predict())
        model_path = Path(_download_artifact_from_uri(artifact_uri=model_uri))
        model_config = Model.load(str(model_path.joinpath("MLmodel")))
        assert pyfunc.FLAVOR_NAME in model_config.flavors
    finally:
        mlflow.end_run()


def test_log_model_calls_register_model(auto_arima_model, tmp_path):
    """Test log model calls register model."""
    artifact_path = "sktime"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["sktime"])
        flavor.log_model(
            sktime_model=auto_arima_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
            registered_model_name="SktimeModel",
        )
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model.assert_called_once_with(
            model_uri,
            "SktimeModel",
            await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        )


def test_log_model_no_registered_model_name(auto_arima_model, tmp_path):
    """Test log model calls register model without registered model name."""
    artifact_path = "sktime"
    register_model_patch = mock.patch("mlflow.register_model")
    with mlflow.start_run(), register_model_patch:
        conda_env = tmp_path.joinpath("conda_env.yaml")
        _mlflow_conda_env(conda_env, additional_pip_deps=["sktime"])
        flavor.log_model(
            sktime_model=auto_arima_model,
            artifact_path=artifact_path,
            conda_env=str(conda_env),
        )
        mlflow.register_model.assert_not_called()


def test_sktime_pyfunc_raises_invalid_df_input(auto_arima_model, model_path):
    """Test pyfunc call raises error with invalid dataframe configuration."""
    flavor.save_model(sktime_model=auto_arima_model, path=model_path)
    loaded_pyfunc = flavor.pyfunc.load_model(model_uri=model_path)

    with pytest.raises(MlflowException, match="The provided prediction pd.DataFrame "):
        loaded_pyfunc.predict(pd.DataFrame([{"predict_method": "predict"}, {"fh": FH}]))

    with pytest.raises(MlflowException, match="The provided prediction configuration "):
        loaded_pyfunc.predict(pd.DataFrame([{"invalid": True}]))

    with pytest.raises(MlflowException, match="Invalid `predict_method` value"):
        loaded_pyfunc.predict(pd.DataFrame([{"predict_method": "predict_proba"}]))


def test_sktime_save_model_raises_invalid_serialization_format(auto_arima_model, model_path):
    """Test save_model call raises error with invalid serialization format."""
    with pytest.raises(MlflowException, match="Unrecognized serialization format: "):
        flavor.save_model(
            sktime_model=auto_arima_model, path=model_path, serialization_format="json"
        )
