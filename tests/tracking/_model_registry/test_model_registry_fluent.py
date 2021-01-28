from unittest import mock
import pytest

from mlflow import register_model, set_registry_uri, get_registry_uri
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    ErrorCode,
    INTERNAL_ERROR,
    RESOURCE_ALREADY_EXISTS,
    FEATURE_DISABLED,
)
from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import TempDir
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS


def test_register_model_raises_exception_with_unsupported_registry_store():
    """
    This test case ensures that the `register_model` operation fails with an informative error
    message when the registry store URI refers to a store that does not support Model Registry
    features (e.g., FileStore).
    """
    with TempDir() as tmp:
        old_registry_uri = get_registry_uri()
        try:
            set_registry_uri(tmp.path())
            with pytest.raises(MlflowException) as exc:
                register_model(model_uri="runs:/1234/some_model", name="testmodel")
                assert exc.value.error_code == ErrorCode.Name(FEATURE_DISABLED)
        finally:
            set_registry_uri(old_registry_uri)


def test_register_model_with_runs_uri():
    create_model_patch = mock.patch.object(
        MlflowClient, "create_registered_model", return_value=RegisteredModel("Model 1")
    )
    get_uri_patch = mock.patch(
        "mlflow.store.artifact.runs_artifact_repo.RunsArtifactRepository.get_underlying_uri",
        return_value="s3:/path/to/source",
    )
    create_version_patch = mock.patch.object(
        MlflowClient,
        "create_model_version",
        return_value=ModelVersion("Model 1", "1", creation_timestamp=123),
    )
    with get_uri_patch, create_model_patch, create_version_patch:
        register_model("runs:/run12345/path/to/model", "Model 1")
        MlflowClient.create_registered_model.assert_called_once_with("Model 1")
        MlflowClient.create_model_version.assert_called_once_with(
            "Model 1",
            "s3:/path/to/source",
            "run12345",
            await_creation_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        )


def test_register_model_with_non_runs_uri():
    create_model_patch = mock.patch.object(
        MlflowClient, "create_registered_model", return_value=RegisteredModel("Model 1")
    )
    create_version_patch = mock.patch.object(
        MlflowClient,
        "create_model_version",
        return_value=ModelVersion("Model 1", "1", creation_timestamp=123),
    )
    with create_model_patch, create_version_patch:
        register_model("s3:/some/path/to/model", "Model 1")
        MlflowClient.create_registered_model.assert_called_once_with("Model 1")
        MlflowClient.create_model_version.assert_called_once_with(
            "Model 1",
            run_id=None,
            source="s3:/some/path/to/model",
            await_creation_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        )


def test_register_model_with_existing_registered_model():
    create_model_patch = mock.patch.object(
        MlflowClient,
        "create_registered_model",
        side_effect=MlflowException("Some Message", RESOURCE_ALREADY_EXISTS),
    )
    create_version_patch = mock.patch.object(
        MlflowClient,
        "create_model_version",
        return_value=ModelVersion("Model 1", "1", creation_timestamp=123),
    )
    with create_model_patch, create_version_patch:
        register_model("s3:/some/path/to/model", "Model 1")
        MlflowClient.create_registered_model.assert_called_once_with("Model 1")
        MlflowClient.create_model_version.assert_called_once_with(
            "Model 1",
            run_id=None,
            source="s3:/some/path/to/model",
            await_creation_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        )


def test_register_model_with_unexpected_mlflow_exception_in_create_registered_model():
    create_model_patch = mock.patch.object(
        MlflowClient,
        "create_registered_model",
        side_effect=MlflowException("Dunno", INTERNAL_ERROR),
    )
    with create_model_patch, pytest.raises(MlflowException):
        register_model("s3:/some/path/to/model", "Model 1")
        MlflowClient.create_registered_model.assert_called_once_with("Model 1")


def test_register_model_with_unexpected_exception_in_create_registered_model():
    create_model_patch = mock.patch.object(
        MlflowClient, "create_registered_model", side_effect=Exception("Dunno")
    )
    with create_model_patch, pytest.raises(Exception):
        register_model("s3:/some/path/to/model", "Model 1")
        MlflowClient.create_registered_model.assert_called_once_with("Model 1")
