from __future__ import print_function

import sys
import os
import pickle
import pytest
from collections import namedtuple

import numpy as np
import sklearn.datasets as datasets
import sklearn.linear_model as glm
import sklearn.neighbors as knn
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import FunctionTransformer as SKFunctionTransformer

import mlflow.sklearn
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.models import Model
from mlflow.tracking.utils import _get_model_log_dir
from mlflow.utils.file_utils import TempDir
from mlflow.utils.environment import _mlflow_conda_env


ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


@pytest.fixture(scope="session")
def sklearn_knn_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    knn_model = knn.KNeighborsClassifier()
    knn_model.fit(X, y)
    return ModelWithData(model=knn_model, inference_data=X)


@pytest.fixture(scope="session")
def sklearn_logreg_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    linear_lr = glm.LogisticRegression()
    linear_lr.fit(X, y)
    return ModelWithData(model=linear_lr, inference_data=X)


@pytest.fixture(scope="session")
def sklearn_custom_transformer_model(sklearn_knn_model):
    def transform(vec):
        print("Invoking custom transformer!")
        return vec + 1

    transformer = SKFunctionTransformer(transform, validate=True)
    pipeline = SKPipeline([("custom_transformer", transformer), ("knn", sklearn_knn_model.model)])
    return ModelWithData(pipeline, inference_data=datasets.load_iris().data[:, :2])


@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")


def test_model_save_load(sklearn_knn_model, model_path):
    knn_model = sklearn_knn_model.model

    mlflow.sklearn.save_model(sk_model=knn_model, path=model_path)
    reloaded_knn_model = mlflow.sklearn.load_model(path=model_path)
    reloaded_knn_pyfunc = pyfunc.load_pyfunc(path=model_path)

    np.testing.assert_array_equal(
            knn_model.predict(sklearn_knn_model.inference_data),
            reloaded_knn_model.predict(sklearn_knn_model.inference_data))

    np.testing.assert_array_equal(
            reloaded_knn_model.predict(sklearn_knn_model.inference_data),
            reloaded_knn_pyfunc.predict(sklearn_knn_model.inference_data))


def test_model_log(sklearn_logreg_model, model_path):
    old_uri = mlflow.get_tracking_uri()
    with TempDir(chdr=True, remove_on_exit=True) as tmp:
        for should_start_run in [False, True]:
            try:
                mlflow.set_tracking_uri("test")
                if should_start_run:
                    mlflow.start_run()

                artifact_path = "linear"
                conda_env = os.path.join(tmp.path(), "conda_env.yaml")
                _mlflow_conda_env(conda_env, additional_pip_deps=["sklearn"])

                mlflow.sklearn.log_model(
                        sk_model=sklearn_logreg_model.model,
                        artifact_path=artifact_path,
                        conda_env=conda_env)
                run_id = mlflow.active_run().info.run_uuid

                reloaded_logreg_model = mlflow.sklearn.load_model(artifact_path, run_id)
                np.testing.assert_array_equal(
                        sklearn_logreg_model.model.predict(sklearn_logreg_model.inference_data),
                        reloaded_logreg_model.predict(sklearn_logreg_model.inference_data))

                model_path = _get_model_log_dir(
                        artifact_path,
                        run_id=run_id)
                model_config = Model.load(os.path.join(model_path, "MLmodel"))
                assert pyfunc.FLAVOR_NAME in model_config.flavors
                assert pyfunc.ENV in model_config.flavors[pyfunc.FLAVOR_NAME]
                env_path = model_config.flavors[pyfunc.FLAVOR_NAME][pyfunc.ENV]
                assert os.path.exists(os.path.join(model_path, env_path))

            finally:
                mlflow.end_run()
                mlflow.set_tracking_uri(old_uri)


def test_custom_transformer_can_be_saved_and_loaded_with_cloudpickle_format(
        sklearn_custom_transformer_model, tmpdir):
    custom_transformer_model = sklearn_custom_transformer_model.model

    # Because the model contains a customer transformer that is not defined at the top level of the
    # current test module, we expect pickle to fail when attempting to serialize it. In contrast,
    # we expect cloudpickle to successfully locate the transformer definition and serialize the
    # model successfully.
    if sys.version_info >= (3, 0):
        expect_exception_context = pytest.raises(AttributeError)
    else:
        expect_exception_context = pytest.raises(pickle.PicklingError)
    with expect_exception_context:
        pickle_format_model_path = os.path.join(str(tmpdir), "pickle_model")
        mlflow.sklearn.save_model(sk_model=custom_transformer_model,
                                  path=pickle_format_model_path,
                                  serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)

    cloudpickle_format_model_path = os.path.join(str(tmpdir), "cloud_pickle_model")
    mlflow.sklearn.save_model(sk_model=custom_transformer_model,
                              path=cloudpickle_format_model_path,
                              serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

    reloaded_custom_transformer_model = mlflow.sklearn.load_model(
            path=cloudpickle_format_model_path)

    np.testing.assert_array_equal(
            custom_transformer_model.predict(sklearn_custom_transformer_model.inference_data),
            reloaded_custom_transformer_model.predict(
                sklearn_custom_transformer_model.inference_data))


def test_save_model_throws_exception_if_serialization_format_is_unrecognized(
        sklearn_knn_model, model_path):
    with pytest.raises(MlflowException) as exc:
        mlflow.sklearn.save_model(sk_model=sklearn_knn_model.model, path=model_path,
                                  serialization_format="not a valid format")

    # The unsupported serialization format should have been detected prior to the execution of
    # any directory creation or state-mutating persistence logic that would prevent a second
    # serialization call with the same model path from succeeding
    assert not os.path.exists(model_path)
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model.model, path=model_path)
