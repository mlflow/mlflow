from azure.storage.blob import BlobClient

import os
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.protos.databricks_artifacts_pb2 import DatabricksMlflowArtifactsService, GetCredentialsForWrite, \
    GetCredentialsForRead
from mlflow.protos.service_pb2 import MlflowService, ListArtifacts
from mlflow.utils.uri import extract_and_normalize_path, is_databricks_acled_artifacts_uri
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.file_utils import relative_path_to_artifact_path, yield_file_in_chunks
from mlflow.utils.rest_utils import call_endpoint, extract_api_info_for_service
from mlflow.utils.databricks_utils import get_databricks_host_creds

_PATH_PREFIX = "/api/2.0"
_AZURE_SINGLE_BLOCK_BLOB_MAX_SIZE = 256000000 - 1  # Can upload blob in single request if it is no more thn 256 MB
_AZURE_MAX_BLOCK_CHUNK_SIZE = 100000000  # Maximum size of each block allowed is 100 MB in stage blob


class DatabricksArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on Azure/AWS with access control.

    The artifact_uri is expected to be of the form
    dbfs:/databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/artifacts/
    """

    def __init__(self, artifact_uri):
        super(DatabricksArtifactRepository, self).__init__(artifact_uri)
        if not artifact_uri.startswith('dbfs:/'):
            raise MlflowException('DatabricksArtifactRepository URI must start with dbfs:/')
        if not is_databricks_acled_artifacts_uri(artifact_uri):
            raise MlflowException('Artifact URI incorrect. Expected path prefix to be '
                                  'databricks/mlflow-tracking/path/to/artifact/..')

        self.run_id = self._extract_run_id()
        self._SERVICE_AND_METHOD_TO_INFO = {
            service: extract_api_info_for_service(service, _PATH_PREFIX)
            for service in [MlflowService, DatabricksMlflowArtifactsService]
        }

    def _extract_run_id(self):
        artifact_path = extract_and_normalize_path(self.artifact_uri)
        return artifact_path.split('/')[3]

    def _call_endpoint(self, service, api, json_body):
        endpoint, method = self._SERVICE_AND_METHOD_TO_INFO[service][api]
        response_proto = api.Response()
        return call_endpoint(get_databricks_host_creds(),
                             endpoint, method, json_body, response_proto)

    def _create_json_body(self, run_id, path=None):
        path = path or ""
        return {
            "run_id": run_id,
            "path": path
        }

    def _get_write_credentials(self, run_id, path=None):
        json_body = message_to_json(GetCredentialsForWrite(run_id=run_id, path=path))
        return self._call_endpoint(DatabricksMlflowArtifactsService,
                                   GetCredentialsForWrite, json_body)

    def _get_read_credentials(self, run_id, path=None):
        json_body = message_to_json(GetCredentialsForRead(run_id=run_id, path=path))
        return self._call_endpoint(DatabricksMlflowArtifactsService,
                                   GetCredentialsForRead, json_body)

    def _azure_upload_file(self, credentials, local_file, artifact_path):
        try:
            if os.path.getsize(local_file) < _AZURE_SINGLE_BLOCK_BLOB_MAX_SIZE:
                signed_write_uri = credentials.signed_uri
                service = BlobClient.from_blob_url(blob_url=signed_write_uri, credential=None)
                with open(local_file, "rb") as data:
                    service.upload_blob(data, overwrite=True)
            else:
                uploading_block_list = list()
                for chunk in yield_file_in_chunks(local_file, _AZURE_MAX_BLOCK_CHUNK_SIZE):
                    signed_write_uri = self._get_write_credentials(self.run_id, artifact_path).credentials.signed_uri
                    service = BlobClient.from_blob_url(blob_url=signed_write_uri, credential=None)
                    block_id = base64.b64encode(uuid.uuid4().hex.encode())
                    service.stage_block(block_id, chunk)
                    uploading_block_list.append(block_id)
                signed_write_uri = self._get_write_credentials(self.run_id, artifact_path).credentials.signed_uri
                service = BlobClient.from_blob_url(blob_url=signed_write_uri, credential=None)
                service.commit_block_list(uploading_block_list)
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

    def _upload_to_cloud(self, cloud_credentials, local_file, artifact_path):
        if cloud_credentials.credentials.type == 1:
            self._azure_upload_file(cloud_credentials.credentials, local_file, artifact_path)
        else:
            raise MlflowException('Not implemented yet')

    def _download_from_cloud(self, cloud_credentials, local_path):
        if cloud_credentials.credentials.type == 1:
            self._azure_download_file(cloud_credentials.credentials, local_path)
        else:
            raise MlflowException('Not implemented yet')

    def log_artifact(self, local_file, artifact_path=None):
        basename = os.path.basename(local_file)
        artifact_path = artifact_path or ""
        artifact_path = os.path.join(artifact_path, basename)
        write_credentials = self._get_write_credentials(self.run_id, artifact_path)
        self._upload_to_cloud(write_credentials, local_file, artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        artifact_path = artifact_path or ''
        for (dirpath, _, filenames) in os.walk(local_dir):
            artifact_subdir = artifact_path
            if dirpath != local_dir:
                rel_path = os.path.relpath(dirpath, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                artifact_subdir = os.path.join(artifact_path, rel_path)
            for name in filenames:
                file_path = os.path.join(dirpath, name)
                self.log_artifact(file_path, artifact_subdir)

    def list_artifacts(self, path=None):
        json_body = message_to_json(ListArtifacts(run_id=self.run_id, path=path))
        artifact_list = self._call_endpoint(MlflowService, ListArtifacts, json_body).files
        # If `path` is a file, ListArtifacts returns a single list element with the
        # same name as `path`. The list_artifacts API expects us to return an empty list in this
        # case, so we do so here.
        if len(artifact_list) == 1 and artifact_list[0].path == path:
            return []
        return artifact_list

    def _download_file(self, remote_file_path, local_path):
        read_credentials = self._get_read_credentials(self.run_id, remote_file_path)
        self._download_from_cloud(read_credentials, local_path)

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException('Not implemented yet')
