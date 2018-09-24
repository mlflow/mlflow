import json
import os

from mlflow.entities import FileInfo
from mlflow.exceptions import IllegalArtifactPathError, MlflowException
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import build_path, get_relative_path
from mlflow.utils.rest_utils import http_request, RESOURCE_DOES_NOT_EXIST
from mlflow.utils.string_utils import strip_prefix

LIST_API_ENDPOINT = '/api/2.0/dbfs/list'
GET_STATUS_ENDPOINT = '/api/2.0/dbfs/get-status'
DOWNLOAD_CHUNK_SIZE = 1024


class DbfsArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on DBFS.

    This repository is used with URIs of the form ``dbfs:/<path>``. The repository can only be used
    together with the RestStore.
    """

    def __init__(self, artifact_uri, get_host_creds):
        cleaned_artifact_uri = artifact_uri.rstrip('/')
        super(DbfsArtifactRepository, self).__init__(cleaned_artifact_uri)
        self.get_host_creds = get_host_creds
        if not cleaned_artifact_uri.startswith('dbfs:/'):
            raise MlflowException('DbfsArtifactRepository URI must start with dbfs:/')

    def _databricks_api_request(self, **kwargs):
        host_creds = self.get_host_creds()
        return http_request(host_creds, **kwargs)

    def _dbfs_list_api(self, json):
        return self._databricks_api_request(endpoint=LIST_API_ENDPOINT, method='GET', json=json)

    def _dbfs_download(self, output_path, endpoint):
        with open(output_path, 'wb') as f:
            response = self._databricks_api_request(endpoint=endpoint, method='GET', stream=True)
            try:
                for content in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    f.write(content)
            finally:
                response.close()

    def _dbfs_is_dir(self, dbfs_path):
        response = self._databricks_api_request(
            endpoint=GET_STATUS_ENDPOINT, method='GET', json={'path': dbfs_path})
        json_response = json.loads(response.text)
        try:
            return json_response['is_dir']
        except KeyError:
            raise MlflowException('DBFS path %s does not exist' % dbfs_path)

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
            response = self._databricks_api_request(
                endpoint=http_endpoint, method='POST', data=f, allow_redirects=False)
            if response.status_code == 409:
                raise MlflowException('File already exists at {} and can\'t be overwritten.'
                                      .format(http_endpoint))
            elif response.status_code != 200:
                raise MlflowException('log_artifact to "{}" returned a non-200 status code.'
                                      .format(http_endpoint))

    def log_artifacts(self, local_dir, artifact_path=None):
        if artifact_path:
            root_http_endpoint = self._get_dbfs_endpoint(artifact_path)
        else:
            root_http_endpoint = self._get_dbfs_endpoint('')
        for (dirpath, _, filenames) in os.walk(local_dir):
            dir_http_endpoint = root_http_endpoint
            if dirpath != local_dir:
                rel_path = get_relative_path(local_dir, dirpath)
                dir_http_endpoint = build_path(root_http_endpoint, rel_path)
            for name in filenames:
                endpoint = build_path(dir_http_endpoint, name)
                with open(build_path(dirpath, name), 'rb') as f:
                    response = self._databricks_api_request(
                        endpoint=endpoint, method='POST', data=f, allow_redirects=False)
                if response.status_code == 409:
                    raise MlflowException('File already exists at {} and can\'t be overwritten.'
                                          .format(endpoint))
                elif response.status_code != 200:
                    raise MlflowException('log_artifacts to "{}" returned a non-200 status code.'
                                          .format(endpoint))

    def list_artifacts(self, path=None):
        if path:
            dbfs_list_json = {'path': self._get_dbfs_path(path)}
        else:
            dbfs_list_json = {'path': self._get_dbfs_path('')}
        response = self._dbfs_list_api(dbfs_list_json)
        json_response = json.loads(response.text)
        # /api/2.0/dbfs/list will not have the 'files' key in the response for empty directories.
        infos = []
        artifact_prefix = strip_prefix(self.artifact_uri, 'dbfs:')
        if json_response.get('error_code', None) == RESOURCE_DOES_NOT_EXIST:
            return []
        dbfs_files = json_response.get('files', [])
        for dbfs_file in dbfs_files:
            stripped_path = strip_prefix(dbfs_file['path'], artifact_prefix + '/')
            # If `path` is a file, the DBFS list API returns a single list element with the
            # same name as `path`. The list_artifacts API expects us to return an empty list in this
            # case, so we do so here.
            if stripped_path == path:
                return []
            is_dir = dbfs_file['is_dir']
            artifact_size = None if is_dir else dbfs_file['file_size']
            infos.append(FileInfo(stripped_path, is_dir, artifact_size))
        return sorted(infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path, local_path):
        self._dbfs_download(output_path=local_path,
                            endpoint=self._get_dbfs_endpoint(remote_file_path))
