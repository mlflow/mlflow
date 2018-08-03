import json
import os

from mlflow.entities.file_info import FileInfo
from mlflow.exceptions import IllegalArtifactPathError, MlflowException
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import build_path, get_relative_path, TempDir
from mlflow.utils.rest_utils import http_request, RESOURCE_DOES_NOT_EXIST
from mlflow.utils.string_utils import strip_prefix

LIST_API_ENDPOINT = '/api/2.0/dbfs/list'
GET_STATUS_ENDPOINT = '/api/2.0/dbfs/get-status'
DOWNLOAD_CHUNK_SIZE = 1024


def _dbfs_list_api(json, http_request_kwargs):
    """
    Pulled out to make it easier to mock.
    """
    return http_request(endpoint=LIST_API_ENDPOINT, method='GET',
                        json=json, **http_request_kwargs)


def _dbfs_download(output_path, endpoint, http_request_kwargs):
    """
    Pulled out to make it easier to mock.
    """
    with open(output_path, 'wb') as f:
        response = http_request(endpoint=endpoint, method='GET', stream=True,
                                **http_request_kwargs)
        try:
            for content in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                f.write(content)
        finally:
            response.close()


def _dbfs_is_dir(dbfs_path, http_request_kwargs):
    response = http_request(endpoint=GET_STATUS_ENDPOINT, method='GET',
                            req_body_json={'path': dbfs_path}, **http_request_kwargs)
    json_response = json.loads(response.text)
    try:
        return json_response['is_dir']
    except KeyError:
        raise Exception('DBFS path %s does not exist' % dbfs_path)


class DbfsArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on DBFS.

    This repository is used with URIs of the form ``dbfs:/<path>``. The repository can only be used
    together with the DatabricksStore.
    """

    def __init__(self, artifact_uri, http_request_kwargs):
        """
        :param http_request_kwargs arguments to add to rest_utils.http_request for all requests
            'hostname', 'headers', and 'verify' are required.
            Should include authentication information to Databricks.
        """
        cleaned_artifact_uri = artifact_uri.rstrip('/')
        super(DbfsArtifactRepository, self).__init__(cleaned_artifact_uri)
        if not cleaned_artifact_uri.startswith('dbfs:/'):
            raise MlflowException('DbfsArtifactRepository URI must start with dbfs:/')
        self.http_request_kwargs = http_request_kwargs

    def _get_dbfs_path(self, artifact_path):
        return '/%s/%s' % (strip_prefix(self.artifact_uri, 'dbfs:/'),
                           strip_prefix(artifact_path, '/'))

    def _get_dbfs_endpoint(self, artifact_path):
        return "/dbfs%s" % self._get_dbfs_path(artifact_path)

    def log_artifact(self, local_file, artifact_path=None):
        basename = os.path.basename(local_file)
        if artifact_path == '':
            raise IllegalArtifactPathError('artifact_path cannot be the empty string.')
        if artifact_path:
            http_endpoint = self._get_dbfs_endpoint(os.path.join(artifact_path, basename))
        else:
            http_endpoint = self._get_dbfs_endpoint(os.path.basename(local_file))
        with open(local_file, 'rb') as f:
            http_request(endpoint=http_endpoint, method='POST', data=f, **self.http_request_kwargs)

    def log_artifacts(self, local_dir, artifact_path=None):
        if artifact_path:
            root_http_endpoint = self._get_dbfs_endpoint(artifact_path)
        else:
            root_http_endpoint = self._get_dbfs_endpoint(os.path.basename(local_dir))
        for (dirpath, _, filenames) in os.walk(local_dir):
            dir_http_endpoint = root_http_endpoint
            if dirpath != local_dir:
                rel_path = get_relative_path(local_dir, dirpath)
                dir_http_endpoint = build_path(root_http_endpoint, rel_path)
            for name in filenames:
                endpoint = build_path(dir_http_endpoint, name)
                with open(build_path(dirpath, name), 'r') as f:
                    http_request(endpoint=endpoint, method='POST', data=f,
                                 **self.http_request_kwargs)

    def list_artifacts(self, path=None):
        if path:
            dbfs_list_json = {'path': self._get_dbfs_path(path)}
        else:
            dbfs_list_json = {'path': self._get_dbfs_path('')}
        response = _dbfs_list_api(dbfs_list_json, self.http_request_kwargs)
        json_response = json.loads(response.text)
        # /api/2.0/dbfs/list will not have the 'files' key in the response for empty directories.
        infos = []
        artifact_prefix = strip_prefix(self.artifact_uri, 'dbfs:')
        if json_response.get('error_code', None) == RESOURCE_DOES_NOT_EXIST:
            return []
        dbfs_files = json_response.get('files', [])
        for dbfs_file in dbfs_files:
            is_dir = dbfs_file['is_dir']
            artifact_size = None if is_dir else dbfs_file['file_size']
            stripped_path = strip_prefix(dbfs_file['path'], artifact_prefix + '/')
            infos.append(FileInfo(stripped_path, is_dir, artifact_size))
        return sorted(infos, key=lambda f: f.path)

    def download_artifacts(self, artifact_path):
        with TempDir(remove_on_exit=False) as tmp:
            return self._download_artifacts_into(artifact_path, tmp.path())

    def _download_artifacts_into(self, artifact_path, dest_dir):
        """Private version of download_artifacts that takes a destination directory."""
        basename = os.path.basename(artifact_path)
        local_path = build_path(dest_dir, basename)
        dbfs_path = self._get_dbfs_path(artifact_path)
        if _dbfs_is_dir(dbfs_path, self.http_request_kwargs):
            # Artifact_path is a directory, so make a directory for it and download everything
            if not os.path.exists(local_path):
                os.mkdir(local_path)
            for file_info in self.list_artifacts(artifact_path):
                self._download_artifacts_into(file_info.path, local_path)
        else:
            _dbfs_download(output_path=local_path, endpoint=self._get_dbfs_endpoint(artifact_path),
                           http_request_kwargs=self.http_request_kwargs)
        return local_path
