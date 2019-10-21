import mock
import pytest

from mlflow import register_model
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_ALREADY_EXISTS
from mlflow.tracking import MlflowClient


def test_register_model_with_runs_uri():
    create_model_patch = mock.patch.object(MlflowClient, "create_registered_model",
                                           return_value=RegisteredModel("Model 1"))
    get_uri_patch = mock.patch(
        "mlflow.store.artifact.runs_artifact_repo.RunsArtifactRepository.get_underlying_uri",
        return_value="s3:/path/to/source")
    create_version_patch = mock.patch.object(
        MlflowClient, "create_model_version",
        return_value=ModelVersion(RegisteredModel("Model 1"), 1))
    with get_uri_patch, create_model_patch, create_version_patch:
        register_model("runs:/run12345/path/to/model", "Model 1")
        MlflowClient.create_registered_model.assert_called_once_with("Model 1")
        MlflowClient.create_model_version.assert_called_once_with("Model 1", "s3:/path/to/source",
                                                                  "run12345")


def test_register_model_with_non_runs_uri():
    create_model_patch = mock.patch.object(MlflowClient, "create_registered_model",
                                           return_value=RegisteredModel("Model 1"))
    create_version_patch = mock.patch.object(
        MlflowClient, "create_model_version",
        return_value=ModelVersion(RegisteredModel("Model 1"), 1))
    with create_model_patch, create_version_patch:
        register_model("s3:/some/path/to/model", "Model 1")
        MlflowClient.create_registered_model.assert_called_once_with("Model 1")
        MlflowClient.create_model_version.assert_called_once_with("Model 1", run_id=None,
                                                                  source="s3:/some/path/to/model")


def test_register_model_with_existing_registered_model():
    create_model_patch = mock.patch.object(MlflowClient, "create_registered_model",
                                           side_effect=MlflowException("Some Message",
                                                                       RESOURCE_ALREADY_EXISTS))
    create_version_patch = mock.patch.object(
        MlflowClient, "create_model_version",
        return_value=ModelVersion(RegisteredModel("Model 1"), 1))
    with create_model_patch, create_version_patch:
        register_model("s3:/some/path/to/model", "Model 1")
        MlflowClient.create_registered_model.assert_called_once_with("Model 1")
        MlflowClient.create_model_version.assert_called_once_with("Model 1", run_id=None,
                                                                  source="s3:/some/path/to/model")


def test_register_model_with_unexpected_mlflow_exception_in_create_registered_model():
    create_model_patch = mock.patch.object(MlflowClient, "create_registered_model",
                                           side_effect=MlflowException("Dunno", INTERNAL_ERROR))
    with create_model_patch, pytest.raises(MlflowException):
        register_model("s3:/some/path/to/model", "Model 1")
        MlflowClient.create_registered_model.assert_called_once_with("Model 1")


def test_register_model_with_unexpected_exception_in_create_registered_model():
    create_model_patch = mock.patch.object(MlflowClient, "create_registered_model",
                                           side_effect=Exception("Dunno"))
    with create_model_patch, pytest.raises(Exception):
        register_model("s3:/some/path/to/model", "Model 1")
        MlflowClient.create_registered_model.assert_called_once_with("Model 1")
