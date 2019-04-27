from __future__ import print_function

import os
import pickle

import numpy as np
import pytest
import six
import sklearn.datasets
import sklearn.linear_model
import sklearn.neighbors

import mlflow
import mlflow.pyfunc
import mlflow.pyfunc.cli
import mlflow.pyfunc.model
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.tracking.artifact_utils import _get_model_log_dir


def _load_pyfunc(path):
    with open(path, "rb") as f:
        if six.PY2:
            return pickle.load(f)
        else:
            return pickle.load(f, encoding='latin1')  # pylint: disable=unexpected-keyword-arg


@pytest.fixture(scope="module")
def iris_data():
    iris = sklearn.datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    return x, y


@pytest.fixture(scope="module")
def sklearn_knn_model(iris_data):
    x, y = iris_data
    knn_model = sklearn.neighbors.KNeighborsClassifier()
    knn_model.fit(x, y)
    return knn_model


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


@pytest.mark.large
def test_model_save_load(sklearn_knn_model, iris_data, tmpdir, model_path):
    sk_model_path = os.path.join(str(tmpdir), "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    model_config = Model(run_id="test", artifact_path="testtest")
    mlflow.pyfunc.save_model(dst_path=model_path,
                             data_path=sk_model_path,
                             loader_module=os.path.basename(__file__)[:-3],
                             code_path=[__file__],
                             model=model_config)

    reloaded_model_config = Model.load(os.path.join(model_path, "MLmodel"))
    assert model_config.__dict__ == reloaded_model_config.__dict__
    assert mlflow.pyfunc.FLAVOR_NAME in reloaded_model_config.flavors
    assert mlflow.pyfunc.PY_VERSION in reloaded_model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    reloaded_model = mlflow.pyfunc.load_pyfunc(model_path)
    np.testing.assert_array_equal(
        sklearn_knn_model.predict(iris_data[0]), reloaded_model.predict(iris_data[0]))


@pytest.mark.large
def test_model_log_load(sklearn_knn_model, iris_data, tmpdir):
    sk_model_path = os.path.join(str(tmpdir), "knn.pkl")
    with open(sk_model_path, "wb") as f:
        pickle.dump(sklearn_knn_model, f)

    pyfunc_artifact_path = "pyfunc_model"
    with mlflow.start_run():
        mlflow.pyfunc.log_model(artifact_path=pyfunc_artifact_path,
                                data_path=sk_model_path,
                                loader_module=os.path.basename(__file__)[:-3],
                                code_path=[__file__])
        pyfunc_run_id = mlflow.active_run().info.run_uuid

    pyfunc_model_path = _get_model_log_dir(pyfunc_artifact_path, pyfunc_run_id)
    model_config = Model.load(os.path.join(pyfunc_model_path, "MLmodel"))
    assert mlflow.pyfunc.FLAVOR_NAME in model_config.flavors
    assert mlflow.pyfunc.PY_VERSION in model_config.flavors[mlflow.pyfunc.FLAVOR_NAME]
    reloaded_model = mlflow.pyfunc.load_pyfunc(pyfunc_model_path)
    np.testing.assert_array_equal(
        sklearn_knn_model.predict(iris_data[0]), reloaded_model.predict(iris_data[0]))


@pytest.mark.large
def test_save_model_with_unsupported_argument_combinations_throws_exception(model_path):
    with pytest.raises(MlflowException) as exc_info:
        mlflow.pyfunc.save_model(dst_path=model_path,
                                 data_path="/path/to/data")
    assert "Either `loader_module` or `python_model` must be specified" in str(exc_info)


@pytest.mark.large
def test_log_model_with_unsupported_argument_combinations_throws_exception():
    with mlflow.start_run(), pytest.raises(MlflowException) as exc_info:
        mlflow.pyfunc.log_model(artifact_path="pyfunc_model",
                                data_path="/path/to/data")
    assert "Either `loader_module` or `python_model` must be specified" in str(exc_info)
