from collections import namedtuple
from unittest import mock

import pytest
from sklearn import datasets
import sklearn.neighbors as knn
import mlflow
import random

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.models import add_libraries_to_model
from mlflow.models.utils import get_model_version_from_model_uri

ModelWithData = namedtuple("ModelWithData", ["model", "inference_data"])


@pytest.fixture(scope="module")
def sklearn_knn_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    knn_model = knn.KNeighborsClassifier()
    knn_model.fit(X, y)
    return ModelWithData(model=knn_model, inference_data=X)


def random_int(lo=1, hi=1000000000):
    return random.randint(lo, hi)


def test_adding_libraries_to_model_default(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"

    # Log a model
    with mlflow.start_run():
        run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model.model,
            artifact_path=artifact_path,
            registered_model_name=model_name,
        )

    wheeled_model_info = add_libraries_to_model(model_uri)
    assert wheeled_model_info.run_id == run_id

    # Verify new model version created
    wheeled_model_version = get_model_version_from_model_uri(wheeled_model_uri)
    assert wheeled_model_version.run_id == run_id
    assert wheeled_model_version.name == model_name


def test_adding_libraries_to_model_new_run(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"

    # Log a model
    with mlflow.start_run():
        original_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model.model,
            artifact_path=artifact_path,
            registered_model_name=model_name,
        )

    with mlflow.start_run():
        wheeled_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        wheeled_model_info = add_libraries_to_model(model_uri)
    assert original_run_id != wheeled_run_id
    assert wheeled_model_info.run_id == wheeled_run_id

    # Verify new model version created
    wheeled_model_version = get_model_version_from_model_uri(wheeled_model_uri)
    assert wheeled_model_version.run_id == wheeled_run_id
    assert wheeled_model_version.name == model_name


def test_adding_libraries_to_model_run_id_passed(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{model_name}/2"

    # Log a model
    with mlflow.start_run():
        original_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model.model,
            artifact_path=artifact_path,
            registered_model_name=model_name,
        )

    with mlflow.start_run():
        wheeled_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        pass

    wheeled_model_info = add_libraries_to_model(model_uri, run_id=wheeled_run_id)
    assert original_run_id != wheeled_run_id
    assert wheeled_model_info.run_id == wheeled_run_id

    # Verify new model version created
    wheeled_model_version = get_model_version_from_model_uri(wheeled_model_uri)
    assert wheeled_model_version.run_id == wheeled_run_id
    assert wheeled_model_version.name == model_name


def test_adding_libraries_to_model_new_model_name(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    wheeled_model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"
    wheeled_model_uri = f"models:/{wheeled_model_name}/1"

    # Log a model
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model.model,
            artifact_path=artifact_path,
            registered_model_name=model_name,
        )

    with mlflow.start_run():
        new_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        wheeled_model_info = add_libraries_to_model(
            model_uri, registered_model_name=wheeled_model_name
        )
    assert wheeled_model_info.run_id == new_run_id

    # Verify new model version created
    wheeled_model_version = get_model_version_from_model_uri(wheeled_model_uri)
    assert wheeled_model_version.run_id == new_run_id
    assert wheeled_model_version.name == wheeled_model_name
    assert wheeled_model_name != model_name


def test_adding_libraries_to_model_when_version_source_None(sklearn_knn_model):
    model_name = f"wheels-test-{random_int()}"
    artifact_path = "model"
    model_uri = f"models:/{model_name}/1"

    # Log a model
    with mlflow.start_run():
        original_run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
        mlflow.sklearn.log_model(
            sk_model=sklearn_knn_model.model,
            artifact_path=artifact_path,
            registered_model_name=model_name,
        )

    model_version_without_source = ModelVersion(name=model_name, version=1, creation_timestamp=124)
    assert model_version_without_source.run_id is None
    with mock.patch.object(
        MlflowClient, "get_model_version", return_value=model_version_without_source
    ) as mlflow_client_mock:
        wheeled_model_info = add_libraries_to_model(model_uri)
        assert wheeled_model_info.run_id is not None
        assert wheeled_model_info.run_id != original_run_id
        mlflow_client_mock.assert_called_once_with(model_name, "1")
