import json

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    GenerateTemporaryModelVersionCredentialsResponse,
    MODEL_VERSION_READ_WRITE,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import call_endpoint
from mlflow.utils.uri import (
    get_databricks_profile_uri_from_artifact_uri,
    get_db_info_from_uri,
    is_databricks_unity_catalog_uri,
    _DATABRICKS_UNITY_CATALOG_SCHEME,
)
from mlflow.store.artifact.utils.models import (
    get_model_name_and_version,
)

from mlflow.store._unity_catalog.registry.utils import get_artifact_repo_from_storage_info

REGISTRY_GET_SCOPED_TOKEN_ENDPOINT = (
    "/mlflow/unity-catalog/model-versions/generate-temporary-credentials"
)


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

    def __init__(self, artifact_uri, registry_uri):
        if not is_databricks_unity_catalog_uri(registry_uri):
            raise MlflowException(
                message="Attempted to instantiate an artifact repo to access models in the "
                f"Unity Catalog with non-Unity Catalog registry URI '{registry_uri}'. "
                f"Please specify a Unity Catalog registry URI of the "
                f"form '{_DATABRICKS_UNITY_CATALOG_SCHEME}[://profile]', e.g. by calling "
                f"mlflow.set_registry_uri('{_DATABRICKS_UNITY_CATALOG_SCHEME}') if using the "
                f"MLflow Python client",
                error_code=INVALID_PARAMETER_VALUE,
            )
        super().__init__(artifact_uri)
        from mlflow.tracking.client import MlflowClient

        registry_uri_from_artifact_uri = get_databricks_profile_uri_from_artifact_uri(
            artifact_uri, result_scheme=_DATABRICKS_UNITY_CATALOG_SCHEME
        )
        if registry_uri_from_artifact_uri is not None:
            registry_uri = registry_uri_from_artifact_uri
        _, key_prefix = get_db_info_from_uri(registry_uri)
        if key_prefix is not None:
            raise MlflowException(
                "Remote model registry access via model URIs of the form "
                "'models://<scope>@<prefix>/<model_name>/<version_or_stage>' is unsupported for "
                "models in the Unity Catalog. We recommend that you access the Unity Catalog "
                "from the current Databricks workspace instead."
            )
        self.registry_uri = registry_uri
        self.client = MlflowClient(registry_uri=self.registry_uri)
        self.model_name, self.model_version = get_model_name_and_version(self.client, artifact_uri)

    def _get_blob_storage_path(self):
        return self.client.get_model_version_download_uri(self.model_name, self.model_version)

    def _get_scoped_token(self):
        req_body = {
            "name": self.model_name,
            "version": self.model_version,
            "operation": MODEL_VERSION_READ_WRITE,
        }
        db_creds = get_databricks_host_creds(self.registry_uri)
        response_proto = GenerateTemporaryModelVersionCredentialsResponse()
        return call_endpoint(
            host_creds=db_creds,
            endpoint=REGISTRY_GET_SCOPED_TOKEN_ENDPOINT,
            method="POST",
            json_body=json.dumps(req_body),
            response_proto=response_proto,
        ).credentials

    def list_artifacts(self, path=None):
        raise MlflowException("This repository does not support listing artifacts.")

    def download_artifacts(self, artifact_path, dst_path=None):
        scoped_token = self._get_scoped_token()
        blob_storage_path = self._get_blob_storage_path()
        repo = get_artifact_repo_from_storage_info(
            storage_location=blob_storage_path, scoped_token=scoped_token
        )
        repo.download_artifacts(artifact_path, dst_path)

    def log_artifact(self, local_file, artifact_path=None):
        raise MlflowException("This repository does not support logging artifacts.")

    def log_artifacts(self, local_dir, artifact_path=None):
        raise MlflowException("This repository does not support logging artifacts.")

    def delete_artifacts(self, artifact_path=None):
        raise NotImplementedError("This artifact repository does not support deleting artifacts")
