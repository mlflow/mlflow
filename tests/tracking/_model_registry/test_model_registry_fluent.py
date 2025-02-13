from unittest import mock

import pytest

import mlflow
from mlflow import MlflowClient, register_model
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    ALREADY_EXISTS,
    INTERNAL_ERROR,
    RESOURCE_ALREADY_EXISTS,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS


def test_register_model_with_runs_uri():
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, model_input):
            return model_input

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel())

    register_model(f"runs:/{run.info.run_id}/model", "Model 1")
    mv = MlflowClient().get_model_version("Model 1", "1")
    assert mv.name == "Model 1"


def test_register_model_with_non_runs_uri():
    create_model_patch = mock.patch.object(
        MlflowClient, "create_registered_model", return_value=RegisteredModel("Model 1")
    )
    create_version_patch = mock.patch.object(
        MlflowClient,
        "_create_model_version",
        return_value=ModelVersion("Model 1", "1", creation_timestamp=123),
    )
    with create_model_patch, create_version_patch:
        register_model("s3:/some/path/to/model", "Model 1")
        MlflowClient.create_registered_model.assert_called_once_with("Model 1")
        MlflowClient._create_model_version.assert_called_once_with(
            name="Model 1",
            run_id=None,
            tags=None,
            source="s3:/some/path/to/model",
            await_creation_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
            local_model_path=None,
        )


@pytest.mark.parametrize("error_code", [RESOURCE_ALREADY_EXISTS, ALREADY_EXISTS])
def test_register_model_with_existing_registered_model(error_code):
    create_model_patch = mock.patch.object(
        MlflowClient,
        "create_registered_model",
        side_effect=MlflowException("Some Message", error_code),
    )
    create_version_patch = mock.patch.object(
        MlflowClient,
        "_create_model_version",
        return_value=ModelVersion("Model 1", "1", creation_timestamp=123),
    )
    with create_model_patch, create_version_patch:
        register_model("s3:/some/path/to/model", "Model 1")
        MlflowClient.create_registered_model.assert_called_once_with("Model 1")
        MlflowClient._create_model_version.assert_called_once_with(
            name="Model 1",
            run_id=None,
            source="s3:/some/path/to/model",
            tags=None,
            await_creation_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
            local_model_path=None,
        )


def test_register_model_with_unexpected_mlflow_exception_in_create_registered_model():
    with mock.patch.object(
        MlflowClient,
        "create_registered_model",
        side_effect=MlflowException("Dunno", INTERNAL_ERROR),
    ) as mock_create_registered_model:
        with pytest.raises(MlflowException, match="Dunno"):
            register_model("s3:/some/path/to/model", "Model 1")
        mock_create_registered_model.assert_called_once_with("Model 1")


def test_register_model_with_unexpected_exception_in_create_registered_model():
    with mock.patch.object(
        MlflowClient, "create_registered_model", side_effect=Exception("Dunno")
    ) as create_registered_model_mock:
        with pytest.raises(Exception, match="Dunno"):
            register_model("s3:/some/path/to/model", "Model 1")
        create_registered_model_mock.assert_called_once_with("Model 1")


def test_register_model_with_tags():
    tags = {"a": "1"}

    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, model_input):
            return model_input

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model("model", python_model=TestModel())

    register_model(f"runs:/{run.info.run_id}/model", "Model 1", tags=tags)
    mv = MlflowClient().get_model_version("Model 1", "1")
    assert mv.tags == tags
