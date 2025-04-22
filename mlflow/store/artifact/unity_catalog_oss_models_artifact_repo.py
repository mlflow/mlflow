import base64

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.protos.databricks_uc_registry_service_pb2 import UcModelRegistryService
from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    READ_MODEL_VERSION as MODEL_VERSION_OPERATION_READ_OSS,
)
from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    GenerateTemporaryModelVersionCredential as GenerateTemporaryModelVersionCredentialsOSS,
)
from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    TemporaryCredentials,
)
from mlflow.protos.unity_catalog_oss_service_pb2 import UnityCatalogService
from mlflow.store._unity_catalog.lineage.constants import _DATABRICKS_LINEAGE_ID_HEADER
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.utils.models import (
    get_model_name_and_version,
)
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils._unity_catalog_utils import (
    get_artifact_repo_from_storage_info,
    get_full_name_from_sc,
)
from mlflow.utils.oss_registry_utils import get_oss_host_creds
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    _UC_OSS_REST_API_PATH_PREFIX,
    call_endpoint,
    extract_api_info_for_service,
)
from mlflow.utils.uri import (
    _OSS_UNITY_CATALOG_SCHEME,
    get_databricks_profile_uri_from_artifact_uri,
    get_db_info_from_uri,
    is_oss_unity_catalog_uri,
)

_METHOD_TO_INFO = extract_api_info_for_service(UcModelRegistryService, _REST_API_PATH_PREFIX)
_METHOD_TO_INFO_OSS = extract_api_info_for_service(
    UnityCatalogService, _UC_OSS_REST_API_PATH_PREFIX
)

import urllib.parse

from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.utils.uri import is_file_uri


class UnityCatalogOSSModelsArtifactRepository(ArtifactRepository):
    """
    Performs storage operations on artifacts controlled by a Unity Catalog model registry

    Temporary scoped tokens for the appropriate cloud storage locations are fetched from the
    remote backend and used to download model artifacts.

    The artifact_uri is expected to be of the form `models:/<model_name>/<model_version>`

    Note : This artifact repository is meant is to be instantiated by the ModelsArtifactRepository
    when the client is pointing to a Unity Catalog model registry.
    """

    def __init__(self, artifact_uri, registry_uri):
        if not is_oss_unity_catalog_uri(registry_uri):
            raise MlflowException(
                message="Attempted to instantiate an artifact repo to access models in the "
                f"OSS Unity Catalog with non-Unity Catalog registry URI '{registry_uri}'. "
                f"Please specify a Unity Catalog registry URI of the "
                f"form '{_OSS_UNITY_CATALOG_SCHEME}[://profile]', e.g. by calling "
                f"mlflow.set_registry_uri('{_OSS_UNITY_CATALOG_SCHEME}') if using the "
                f"MLflow Python client",
                error_code=INVALID_PARAMETER_VALUE,
            )
        super().__init__(artifact_uri)
        from mlflow.tracking.client import MlflowClient

        registry_uri_from_artifact_uri = get_databricks_profile_uri_from_artifact_uri(
            artifact_uri, result_scheme=_OSS_UNITY_CATALOG_SCHEME
        )
        if registry_uri_from_artifact_uri is not None:
            registry_uri = registry_uri_from_artifact_uri

        _, key_prefix = get_db_info_from_uri(urllib.parse.urlparse(registry_uri).path)
        if key_prefix is not None:
            raise MlflowException(
                "Remote model registry access via model URIs of the form "
                "'models://<scope>@<prefix>/<model_name>/<version_or_stage>' is unsupported for "
                "models in the Unity Catalog. We recommend that you access the Unity Catalog "
                "from the current Databricks workspace instead."
            )
        self.registry_uri = registry_uri
        self.client = MlflowClient(registry_uri=self.registry_uri)
        try:
            spark = _get_active_spark_session()
        except Exception:
            pass
        model_name, self.model_version = get_model_name_and_version(self.client, artifact_uri)
        self.model_name = get_full_name_from_sc(model_name, spark)

    def _get_blob_storage_path(self):
        return self.client.get_model_version_download_uri(self.model_name, self.model_version)

    def _get_scoped_token(self, lineage_header_info=None):
        extra_headers = {}
        if lineage_header_info:
            header_json = message_to_json(lineage_header_info)
            header_base64 = base64.b64encode(header_json.encode())
            extra_headers[_DATABRICKS_LINEAGE_ID_HEADER] = header_base64
        oss_creds = get_oss_host_creds(
            self.registry_uri
        )  # Implement ENV variable the same way the databricks user/token is specified
        oss_endpoint, oss_method = _METHOD_TO_INFO_OSS[GenerateTemporaryModelVersionCredentialsOSS]
        [catalog_name, schema_name, model_name] = self.model_name.split(
            "."
        )  # self.model_name is actually the full name
        oss_req_body = message_to_json(
            GenerateTemporaryModelVersionCredentialsOSS(
                catalog_name=catalog_name,
                schema_name=schema_name,
                model_name=model_name,
                version=int(self.model_version),
                operation=MODEL_VERSION_OPERATION_READ_OSS,
            )
        )
        oss_response_proto = TemporaryCredentials()
        return call_endpoint(
            host_creds=oss_creds,
            endpoint=oss_endpoint,
            method=oss_method,
            json_body=oss_req_body,
            response_proto=oss_response_proto,
            extra_headers=extra_headers,
        )

    def _get_artifact_repo(self, lineage_header_info=None):
        """
        Get underlying ArtifactRepository instance for model version blob
        storage
        """
        blob_storage_path = self._get_blob_storage_path()
        if is_file_uri(blob_storage_path):
            return LocalArtifactRepository(artifact_uri=blob_storage_path)
        scoped_token = self._get_scoped_token(lineage_header_info=lineage_header_info)
        return get_artifact_repo_from_storage_info(
            storage_location=blob_storage_path,
            scoped_token=scoped_token,
            base_credential_refresh_def=self._get_scoped_token,
            is_oss=True,
        )

    def list_artifacts(self, path=None):
        return self._get_artifact_repo().list_artifacts(path=path)

    def download_artifacts(self, artifact_path, dst_path=None, lineage_header_info=None):
        return self._get_artifact_repo(lineage_header_info=lineage_header_info).download_artifacts(
            artifact_path, dst_path
        )

    def log_artifact(self, local_file, artifact_path=None):
        raise MlflowException("This repository does not support logging artifacts.")

    def log_artifacts(self, local_dir, artifact_path=None):
        raise MlflowException("This repository does not support logging artifacts.")

    def delete_artifacts(self, artifact_path=None):
        raise NotImplementedError("This artifact repository does not support deleting artifacts")
