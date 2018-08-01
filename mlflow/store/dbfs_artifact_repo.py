import os

from databricks_cli.dbfs.api import FileInfo

from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import build_path, get_relative_path, TempDir
from mlflow.utils.rest_utils import http_request, databricks_api_request
from mlflow.utils.string_utils import strip_prefix

LIST_API_ENDPOINT = 'dbfs/list'
GET_STATUS_ENDPOINT = 'dbfs/get-status'


class DbfsArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on DBFS.

    This repository is used with URIs of the form ``dbfs:/<path>``. The repository can only be used
    together with the DatabricksStore.
    """

    def __init__(self, artifact_uri, http_request_kwargs):
        """
        :param http_request_kwargs arguments to add to rest_utils.http_request for all requests
            'hostname', 'headers', and 'secure_verify' are required.
            Should include authentication information to Databricks.
        """
        super(DbfsArtifactRepository, self).__init__(artifact_uri)
        assert artifact_uri.startswith('dbfs:/'), 'DbfsArtifactRepository URI must start with ' \
                                                  'dbfs:/'
        # TODO(andrew): should artifact_uri always end in a '/'? We assume it does for the
        # implementation now.
        self.http_request_kwargs = http_request_kwargs

    def _get_dbfs_path(self, artifact_path):
        return '/%s/%s' % (strip_prefix(self.artifact_uri, 'dbfs:/'),
                           strip_prefix(artifact_path, '/'))

    def _get_dbfs_endpoint(self, artifact_path):
        return "/dbfs/%s" % self._get_dbfs_path(artifact_path)

    def log_artifact(self, local_file, artifact_path=None):
        if artifact_path:
            http_endpoint = self._get_dbfs_endpoint(artifact_path)
        else:
            http_endpoint = self._get_dbfs_endpoint(os.path.basename(local_file))
        with open(local_file, 'r') as f:
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
        response = databricks_api_request(endpoint=LIST_API_ENDPOINT, method='GET',
                                          req_body_json=dbfs_list_json, **self.http_request_kwargs)
        # /api/2.0/dbfs/list will not have the 'files' key in the response for empty directories.
        infos = []
        artifact_prefix = strip_prefix(self.artifact_uri, 'dbfs:')
        dbfs_files = response.get('files', [])
        for dbfs_file in dbfs_files:
            is_dir = dbfs_file['is_dir']
            artifact_size = None if is_dir else dbfs_file['file_size']
            stripped_path = strip_prefix(dbfs_file['path'], artifact_prefix)
            infos.append(FileInfo(stripped_path, is_dir, artifact_size))
        return sorted(infos, key=lambda f: f.path)

    def _dbfs_is_dir(self, dbfs_path):
        response = databricks_api_request(endpoint=GET_STATUS_ENDPOINT, method='GET',
                                          req_body_json={'path': dbfs_path},
                                          **self.http_request_kwargs)
        try:
            return response['is_dir']
        except KeyError:
            raise Exception('DBFS path %s does not exist' % dbfs_path)


    def download_artifacts(self, artifact_path):
        with TempDir(remove_on_exit=False) as tmp:
            return self._download_artifacts_into(artifact_path, tmp.path())

    def _download_artifacts_into(self, artifact_path, dest_dir):
        """Private version of download_artifacts that takes a destination directory."""
        basename = os.path.basename(artifact_path)
        local_path = build_path(dest_dir, basename)
        dbfs_path = self._get_dbfs_path(artifact_path)
        if self._dbfs_is_dir(dbfs_path):
            # Artifact_path is a directory, so make a directory for it and download everything
            os.mkdir(local_path)
            for file_info in self.list_artifacts(artifact_path):
                self._download_artifacts_into(file_info.path, local_path)
        else:
            with open(local_path, 'wb') as f:
                response = http_request(endpoint=self._get_dbfs_endpoint(artifact_path),
                                        method='GET', stream=True, **self.http_request_kwargs)
                try:
                    for bytes in response.iter_content(chunk_size=1024):
                        f.write(bytes)
                finally:
                    response.close()
        return local_path
