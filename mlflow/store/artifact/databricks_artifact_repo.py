from azure.storage.blob import BlobClient

import os
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.string_utils import strip_suffix
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.utils.rest_utils import call_endpoint, extract_api_info_for_service
from mlflow.protos.databricks_artifacts_pb2 import DatabricksMlflowArtifactsService
from mlflow.protos.databricks_artifacts_pb2 import GetCredentialsForWrite, GetCredentialsForRead
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.protos.service_pb2 import MlflowService, ListArtifacts
from mlflow.utils.uri import extract_and_normalize_path

_PATH_PREFIX = "/api/2.0"


class DatabricksArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on Azure/AWS with access control.

    The artifact_uri is expected to be of the form dbfs:/databricks/mlflow-tracking/EXP_ID/RUN_ID/artifacts/..
    """

    def __init__(self, artifact_uri):
        super(DatabricksArtifactRepository, self).__init__(artifact_uri)
        self._SERVICE_AND_METHOD_TO_INFO = {
            service: extract_api_info_for_service(service, _PATH_PREFIX)
            for service in [MlflowService, DatabricksMlflowArtifactsService]
        }
        self.credential_type_to_cloud_service = {

        }

    def _extract_run_id(self, artifact_uri):
        artifact_path = extract_and_normalize_path(artifact_uri)
        return artifact_path.split('/')[3]

    def _call_endpoint(self, service, api, json_body):
        endpoint, method = self._SERVICE_AND_METHOD_TO_INFO[service][api]
        response_proto = api.Response()
        return call_endpoint(get_databricks_host_creds(), endpoint, method, json_body, response_proto)

    def _create_json_body(self, run_id, path=None):
        path = path or ""
        return {
            "run_id": run_id,
            "path": path
        }

    def _get_write_credentials(self, run_id, path=None):
        return self._call_endpoint(DatabricksMlflowArtifactsService, GetCredentialsForWrite,
                                   self._create_json_body(run_id, path))

    def _get_read_credentials(self, run_id, path=None):
        return self._call_endpoint(DatabricksMlflowArtifactsService, GetCredentialsForRead,
                                   self._create_json_body(run_id, path))

    def _azure_upload_file(self, credentials, local_file):
        signed_write_uri = credentials.signed_uri
        service = BlobClient.from_blob_url(blob_url=signed_write_uri, credential=None)
        try:
            with open(local_file, "rb") as data:
                service.upload_blob(data, overwrite=True)
        except Exception as err:
            raise MlflowException(err)

    def _azure_download_file(self, credentials, local_path):
        signed_read_uri = credentials.signed_uri
        service = BlobClient.from_blob_url(blob_url=signed_read_uri, credential=None)
        try:
            with open(local_path, "wb") as output_file:
                blob = service.download_blob()
                output_file.write(blob.readall())
        except Exception as err:
            raise MlflowException(err)

    def _aws_upload_file(self, credentials, local_file):
        pass

    def _aws_download_file(self, credentials, local_path):
        pass

    def log_artifact(self, local_file, artifact_path=None):
        run_id = self._extract_run_id(self.artifact_uri)
        write_credentials = self._get_write_credentials(run_id, artifact_path)
        if write_credentials.credentials.type == 1:
            self._azure_upload_file(write_credentials.credentials, local_file)
        else:
            raise MlflowException('Not implemented yet')

    def log_artifacts(self, local_dir, artifact_path=None):
        artifact_path = artifact_path or ''
        basename = os.path.basename(strip_suffix(local_dir, '/'))
        for (dirpath, _, filenames) in os.walk(local_dir):
            artifact_subdir = basename
            if dirpath != local_dir:
                rel_path = os.path.relpath(dirpath, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                artifact_subdir = os.path.join(artifact_subdir, rel_path)
            for name in filenames:
                local_file = os.path.join(dirpath, name)
                artifact_location = os.path.join(artifact_path, artifact_subdir)
                self.log_artifact(local_file, artifact_location)

    def list_artifacts(self, path=None):
        run_id = self._extract_run_id(self.artifact_uri)
        return self._call_endpoint(MlflowService, ListArtifacts, self._create_json_body(run_id, path))

    def _download_file(self, remote_file_path, local_path):
        run_id = self._extract_run_id(self.artifact_uri)
        read_credentials = self._get_read_credentials(run_id, remote_file_path)
        if read_credentials.credentials.type == 1:
            self._azure_upload_file(read_credentials.credentials, local_path)
        else:
            raise MlflowException('Not implemented yet')

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException('Not implemented yet')
