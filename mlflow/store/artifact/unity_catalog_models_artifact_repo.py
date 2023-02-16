import logging
import json

import mlflow.tracking
from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_uc_registry_messages_pb2 import GenerateTemporaryModelVersionCredentialsResponse
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import download_file_using_http_uri
from mlflow.utils.rest_utils import http_request, call_endpoint
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri
from mlflow.store.artifact.utils.models import (
    get_model_name_and_version,
    is_using_databricks_registry,
)

from mlflow.store._unity_catalog.registry.utils import get_artifact_repo_from_storage_info

_logger = logging.getLogger(__name__)
REGISTRY_GET_DOWNLOAD_URI_ENDPOINT = "/api/2.0/mlflow/unity-catalog/model-versions/get"
REGISTRY_GET_SCOPED_TOKEN_ENDPOINT = "/mlflow/unity-catalog/model-versions/generate-temporary-credentials"


class UnityCatalogModelsArtifactRepository(ArtifactRepository):
    """
    Performs storage operations on artifacts controlled by a Unity Catalog model registry

    Temporary scoped tokens for the appropriate cloud storage locations are fetched from the
    remote backend and used to download model artifacts.

    The artifact_uri is expected to be of the form
    - `models:/<model_name>/<model_version>`
    - `models:/<model_name>/<stage>`  (refers to the latest model version in the given stage)
    - `models:/<model_name>/latest`  (refers to the latest of all model versions)
    - `models://<profile>/<model_name>/<model_version or stage or 'latest'>`

    Note : This artifact repository is meant is to be instantiated by the ModelsArtifactRepository
    when the client is pointing to a Unity Catalog model registry.
    """

    def __init__(self, artifact_uri):
        if not is_using_databricks_registry(artifact_uri):
            raise MlflowException(
                message="A valid databricks profile is required to instantiate this repository",
                error_code=INVALID_PARAMETER_VALUE,
            )
        super().__init__(artifact_uri)
        from mlflow.tracking.client import MlflowClient

        self.databricks_profile_uri = (
                get_databricks_profile_uri_from_artifact_uri(artifact_uri) or mlflow.get_registry_uri()
        )
        self.client = MlflowClient(registry_uri=self.databricks_profile_uri)
        self.model_name, self.model_version = get_model_name_and_version(self.client, artifact_uri)

    def _get_blob_storage_path(self):
        return self.client.get_model_version_download_uri(self.name, self.model_version)

    def _get_scoped_token(self):
        req_body = {"name": self.model_name, "version": self.model_version}
        db_creds = get_databricks_host_creds(self.databricks_profile_uri)
        response_proto = GenerateTemporaryModelVersionCredentialsResponse()
        return call_endpoint(host_creds=db_creds, endpoint=REGISTRY_GET_SCOPED_TOKEN_ENDPOINT, method="POST",
                             json_body=req_body, response_proto=response_proto)

    def list_artifacts(self, path=None):
        raise MlflowException("This repository does not support listing artifacts.")

    def download_artifacts(self, artifact_path, dst_path=None):
        if artifact_path != "":
            raise MlflowException(f"Got non-empty artifact_path {artifact_path} when attempting to download UC model "
                                  f"version artifacts. Downloading specific artifacts for a model version in the "
                                  f"Unity Catalog is not upported. Pass the empty string ('') as the artifact_path "
                                  f"argument instead")
        scoped_token = self._get_scoped_token()
        blob_storage_path = self._get_blob_storage_path()
        repo = get_artifact_repo_from_storage_info(storage_location=blob_storage_path, scoped_token=scoped_token)
        repo.download_artifacts(artifact_path, dst_path)

    def log_artifact(self, local_file, artifact_path=None):
        raise MlflowException("This repository does not support logging artifacts.")

    def log_artifacts(self, local_dir, artifact_path=None):
        raise MlflowException("This repository does not support logging artifacts.")

    def delete_artifacts(self, artifact_path=None):
        raise NotImplementedError("This artifact repository does not support deleting artifacts")
