import os
import pytest

import pmdarima
import numpy as np
import pandas as pd

import mlflow.pmdarima
from mlflow.models import infer_signature, Model
from mlflow.models.utils import _read_example
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.utils.file_utils import TempDir
from tests.prophet.test_prophet_model_export import DataGeneration

from tests.helper_functions import mock_s3_bucket  # pylint: disable=unused-import

# pytestmark = pytest.mark.large


@pytest.fixture(scope="function")
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.fixture(scope="module")
def test_data():

    data_conf = {
        "shift": False,
        "start": "2016-01-01",
        "size": 365 * 3,
        "seasonal_period": 7,
        "seasonal_freq": 0.1,
        "date_field": "date",
        "target_field": "orders",
    }
    raw = DataGeneration(**data_conf).create_series_df()
    return raw.set_index("date")


@pytest.fixture(scope="module")
def auto_arima_model(test_data):

    return pmdarima.auto_arima(
        test_data["orders"], max_d=1, suppress_warnings=True, error_action="raise"
    )


@pytest.fixture(scope="module")
def auto_arima_object_module(test_data):

    model = pmdarima.arima.ARIMA(order=(2, 1, 3), maxiter=25)
    return model.fit(test_data["orders"])


def test_pmdarima_auto_arima_save_and_load(auto_arima_model, model_path):

    mlflow.pmdarima.save_model(pmdarima_model=auto_arima_model, path=model_path)

    loaded_model = mlflow.pmdarima.load_model(model_uri=model_path)

    np.testing.assert_array_equal(auto_arima_model.predict(10), loaded_model.predict(10))


def test_pmdarima_arima_object_save_and_load(auto_arima_object_module, model_path):

    mlflow.pmdarima.save_model(pmdarima_model=auto_arima_object_module, path=model_path)

    loaded_model = mlflow.pmdarima.load_model(model_uri=model_path)

    np.testing.assert_array_equal(auto_arima_object_module.predict(30), loaded_model.predict(30))


def test_pmdarima_autoarima_pyfunc_save_and_load(auto_arima_model, model_path):

    mlflow.pmdarima.save_model(pmdarima_model=auto_arima_model, path=model_path)
    loaded_pyfunc = mlflow.pyfunc.load_model(model_uri=model_path)

    predict_conf = pd.DataFrame({"n_periods": 60, "return_conf_int": True, "alpha": 0.1}, index=[0])

    model_predict = auto_arima_model.predict(n_periods=60, return_conf_int=True, alpha=0.1)
    pyfunc_predict = loaded_pyfunc.predict(predict_conf)

    for idx, arr in enumerate(model_predict):
        np.testing.assert_array_equal(arr, pyfunc_predict[idx])


def test_pmdarima_signature_and_examples_saved_correctly(auto_arima_model, test_data):

    # NB: with return_conf_int=True, the return type of pmdarima models is a tuple.
    prediction = auto_arima_model.predict(n_periods=20, return_conf_int=True, alpha=0.05)
    signature_ = infer_signature(test_data, prediction[0])
    example_ = test_data[0:5].copy(deep=False)
    for signature in (None, signature_):
        for example in (None, example_):
            with TempDir() as tmp:
                path = tmp.path("model")
                mlflow.pmdarima.save_model(
                    auto_arima_model, path=path, signature=signature, input_example=example
                )
                mlflow_model = Model.load(path)
                assert signature == mlflow_model.signature
                if example is None:
                    assert mlflow_model.saved_input_example_info is None
                else:
                    r_example = _read_example(mlflow_model, path).copy(deep=False)
                    np.testing.assert_array_equal(r_example, example)


def test_pmdarima_load_from_remote_uri_succeeds(
    auto_arima_object_module, model_path, mock_s3_bucket
):

    mlflow.pmdarima.save_model(pmdarima_model=auto_arima_object_module, path=model_path)

    artifact_root = f"s3://{mock_s3_bucket}"
    artifact_path = "model"
    artifact_repo = S3ArtifactRepository(artifact_root)
    artifact_repo.log_artifacts(model_path, artifact_path=artifact_path)

    model_uri = os.path.join(artifact_root, artifact_path)
    reloaded_pmdarima_model = mlflow.pmdarima.load_model(model_uri=model_uri)

    np.testing.assert_array_equal(
        auto_arima_object_module.predict(30), reloaded_pmdarima_model.predict(30)
    )
