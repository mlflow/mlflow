import os
from contextlib import contextmanager

from six.moves import urllib

from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.store.local_artifact_repo import verify_artifact_path
from mlflow.utils.validation import bad_path_message


class HdfsArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on HDFS.

    This repository is used with URIs of the form ``hdfs:/<path>``. The repository can only be used
    together with the RestStore.
    """

    def __init__(self, artifact_uri):

        self.host, self.port, self.path = _resolve_connection_params(artifact_uri,
                                                                     'localhost',
                                                                     8020)
        super(HdfsArtifactRepository, self).__init__(artifact_uri)

    def get_path_module(self):
        import posixpath
        return posixpath

    def log_artifact(self, local_file, artifact_path=None):
        """
            Log artifact in hdfs.
        :param local_file: source file path
        :param artifact_path: when specified will attempt to write under artifact_uri/artifact_path
        """
        hdfs_path = 'hdfs:' + _resolve_path(self.path, artifact_path)
        verify_artifact_path(hdfs_path)

        with hdfs_system(host=self.host, port=self.port) as hdfs:
            with hdfs.open(hdfs_path, 'wb') as output:
                output.write(open(local_file, "rb").read())

    def log_artifacts(self, local_dir, artifact_path=None):
        """
            Log artifacts in hdfs.
            Missing remote directories will be created if needed.
        :param local_dir: source dir path
        :param artifact_path: when specified will attempt to write under artifact_uri/artifact_path
        """
        hdfs_path = 'hdfs:' + _resolve_path(self.path, artifact_path)
        verify_artifact_path(hdfs_path)

        with hdfs_system(host=self.host, port=self.port) as hdfs:

            if not hdfs.exists(hdfs_path):
                hdfs.mkdir(hdfs_path)

            for subdir_path, _, files in os.walk(local_dir):

                hdfs_subdir_path = _resolve_sub_dir(hdfs_path, subdir_path, local_dir)

                if not hdfs.exists(hdfs_subdir_path):
                    hdfs.mkdir(hdfs_subdir_path)

                for each_file in files:
                    source = subdir_path + os.sep + each_file
                    destination = hdfs_subdir_path + os.sep + each_file
                    with hdfs.open(destination, 'wb') as output_stream:
                        output_stream.write(open(source, "rb").read())

    def list_artifacts(self, path):
        paths = []

        with hdfs_system(host=self.host, port=self.port) as hdfs:
            if hdfs.exists(path):
                files_info = []
                for subdir, _, files in hdfs.walk(path):
                    files_info.append(FileInfo(subdir,
                                               hdfs.isdir(subdir),
                                               hdfs.info(subdir).get("size")))
                    for each_file in files:
                        file_path = subdir + os.sep + each_file
                        files_info.append(FileInfo(file_path,
                                                   hdfs.isdir(file_path),
                                                   hdfs.info(file_path).get("size")))
                return sorted(files_info, key=lambda f: paths)
            return paths

    def _download_file(self, remote_file_path, local_path):

        _verify_remote_file_path(remote_file_path)

        with hdfs_system(host=self.host, port=self.port) as hdfs:

            root_dir = os.path.split(os.path.dirname(local_path))[1]

            for subdir, _dirs, files in hdfs.walk(local_path):
                subdir_local_path = remote_file_path + os.sep + _child_path(subdir, root_dir)

                if not self.get_path_module().exists(subdir_local_path):
                    os.makedirs(subdir_local_path)

                for each_file in files:
                    file_path = subdir + os.sep + each_file
                    local_file_path = subdir_local_path + os.sep + each_file
                    with open(local_file_path, 'wb') as f:
                        f.write(hdfs.open(file_path, 'rb').read())


@contextmanager
def hdfs_system(host, port):
    import pyarrow as pa

    driver = os.getenv('MLFLOW_HDFS_DRIVER')
    kerb_ticket = os.getenv('MLFLOW_KERBEROS_TICKET_CACHE')
    kerberos_user = os.getenv('MLFLOW_KERBEROS_USER')

    connected = pa.hdfs.connect(host=host,
                                port=port,
                                user=kerberos_user,
                                driver=driver,
                                kerb_ticket=kerb_ticket)
    yield connected
    connected.close()


def _resolve_connection_params(artifact_uri, default_host, default_port):
    parsed = urllib.parse.urlparse(artifact_uri)
    host = default_host
    port = default_port
    if parsed.hostname:
        host = parsed.hostname
    if parsed.port:
        port = parsed.port
    return host, port, parsed.path


def _child_path(path, root_dir):
    split_paths = path.split(root_dir)
    child_dir_path = ''
    for index in range(1, len(split_paths)):
        child_dirs = split_paths[index].split(os.sep)
        for each_dir in child_dirs:
            child_dir_path = child_dir_path + os.sep + each_dir
    return child_dir_path


def _resolve_path(path, artifact_path):
    if artifact_path:
        return path + os.sep + artifact_path
    return path


def _resolve_sub_dir(hdfs_base_path, sub_dir, local_dir):
    if sub_dir is local_dir:
        return hdfs_base_path

    import posixpath
    _, tail = posixpath.split(sub_dir)
    return hdfs_base_path + os.sep + tail


def _verify_remote_file_path(remote_file_path):
    if remote_file_path is None or remote_file_path == '':
        message = bad_path_message(remote_file_path)
        raise MlflowException("Invalid output path: '%s'. %s" % (remote_file_path, message))
