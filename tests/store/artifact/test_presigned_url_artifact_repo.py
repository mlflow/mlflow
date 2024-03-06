import os.path
from unittest import mock
from unittest.mock import MagicMock

import pytest
import yaml

import mlflow
from mlflow.models import ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_uc_registry_messages_pb2 import TemporaryCredentials, ArclightCredentials
from mlflow.store._unity_catalog.registry.rest_store import UcModelRegistryStore
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.presigned_url_artifact_repo import PresignedUrlArtifactRepository
from mlflow.types import Schema, ColSpec, DataType
from mlflow.utils._unity_catalog_utils import get_artifact_repo_from_storage_info

S3_REPOSITORY_MODULE = "mlflow.store.artifact.presigned_url_artifact_repo"
ARTIFACT_REPOSITORY = f"{S3_REPOSITORY_MODULE}.PresignedUrlArtifactRepository"


@pytest.fixture
def s3_artifact_root(mock_s3_bucket):
    return f"s3://{mock_s3_bucket}"


@pytest.fixture
def local_model_dir(tmp_path):
    fake_signature = ModelSignature(
        inputs=Schema([ColSpec(DataType.string)]), outputs=Schema([ColSpec(DataType.double)])
    )
    fake_mlmodel_contents = {
        "artifact_path": "some-artifact-path",
        "run_id": "abc123",
        "signature": fake_signature.to_dict(),
    }
    with open(tmp_path.joinpath(MLMODEL_FILE_NAME), "w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)
    return tmp_path


@pytest.fixture
def store():
    with mock.patch("databricks_cli.configure.provider.get_config"):
        yield UcModelRegistryStore(store_uri="databricks-uc", tracking_uri="databricks")


@pytest.fixture
def temp_creds(s3_artifact_root):
    with mock.patch("mlflow.utils._unity_catalog_utils.get_artifact_repo_from_storage_info"):
        yield PresignedUrlArtifactRepository(s3_artifact_root)


def test_return_correct_repo(s3_artifact_root):
    obj = get_artifact_repo_from_storage_info(s3_artifact_root,
                                              TemporaryCredentials(arclight_credentials=ArclightCredentials()))
    assert type(obj) is PresignedUrlArtifactRepository


@mock.patch("mlflow.utils.rest_utils.call_endpoint")
def test_create_model_version(call_endpoint, tmp_path):
    local_file = os.path.join(tmp_path, "hello.txt")
    with open(local_file, "w+") as f:
        f.write("Hello, World!")

    call_endpoint.return_value = MagicMock(GetPresignedUrlsResponse(
        aws_presigned_urls=AwsPresignedUrls(
            urls=[
                AwsPresignedUrl(
                    filename=local_file,
                    presigned_url="http://example.com")
            ]
        )
    ))

    with mock.patch(ARTIFACT_REPOSITORY) as repo:
        PresignedUrlArtifactRepository("s3://bucket").log_artifacts(local_dir=tmp_path)
        repo._get_write_credential_infos.assert_called_once()


def test_download_artifacts():
    mlflow.set_registry_uri("databricks-uc://arclight-prototype")

    # uri = mlflow.MlflowClient().get_model_version_download_uri("artjen.test.newtestmodel", "2")
    # ModelsArtifactRepository(uri).download_artifacts(artifact_path="")
    ModelsArtifactRepository("models:/artjen.test.newtestmodel/2").download_artifacts(artifact_path="")
