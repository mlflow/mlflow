import os

from six.moves import urllib

from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils.validation import path_not_unique, bad_path_message


class HdfsArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on HDFS.

    This repository is used with URIs of the form ``hdfs:/<path>``. The repository can only be used
    together with the RestStore.
    """

    def __init__(self, artifact_uri):
        parsed = urllib.parse.urlparse(artifact_uri)
        self.config = {
            'host': parsed.hostname,
            'port': 8020 if parsed.port is None else parsed.port
        }
        self.path = parsed.path

        if self.config['host'] is None:
            self.config['host'] = 'localhost'
        super(HdfsArtifactRepository, self).__init__(artifact_uri)
        if not artifact_uri.startswith('hdfs:/'):
            raise MlflowException('hdfsArtifactRepository URI must start with hdfs:/')

    def get_path_module(self):
        import posixpath
        return posixpath

    def _create_hdfs_conn(self):
        from hdfs3 import HDFileSystem
        hdfs = HDFileSystem(host=self.config["host"],
                            port=int(self.config["port"]))
        return hdfs

    def log_artifact(self, local_file, artifact_path=None):
        if artifact_path:
            dest_path = artifact_path
        else:
            dest_path = self.path
        try:
            hdfs = self._create_hdfs_conn()
            hdfs.put(local_file, dest_path)
        finally:
            if hdfs:
                hdfs.disconnect()

    def log_artifacts(self, local_dir, artifact_path=None):
        if artifact_path and path_not_unique(artifact_path):
            raise Exception("Invalid artifact path: '%s'. %s" % (artifact_path,
                                                                 bad_path_message(artifact_path)))
        try:
            hdfs = self._create_hdfs_conn()
            hdfs_model_path = self.path + os.sep + artifact_path + "/model/"
            if not (hdfs.isdir(hdfs_model_path)):
                hdfs.mkdir(artifact_path)
            rootdir_name = os.path.split(os.path.dirname(local_dir))[1]
            for subdir, dirs, files in os.walk(local_dir):
                hdfs_subdir_path = self.path + os.sep \
                                   + artifact_path + os.sep \
                                   + self.extract_child(subdir,
                                                        rootdir_name)
                if not (hdfs.isdir(hdfs_subdir_path)):
                    hdfs.mkdir(hdfs_subdir_path)
                for file in files:
                    filepath = subdir + os.sep + file
                    hdfs.put(filepath, hdfs_subdir_path + "/" + file)
        finally:
            if hdfs:
                hdfs.disconnect()

    def extract_child(self, path, rootdir_name):
        splitpaths = path.split(os.sep + rootdir_name + os.sep)
        index = 0
        child_dir_path = ''
        for index in range(1, len(splitpaths)):
            childdirs = splitpaths[index].split(os.sep)
            for each_dir in childdirs:
                child_dir_path = child_dir_path + os.sep + each_dir
        return child_dir_path

    def list_artifacts(self, path=None):
        paths = []
        hdfs_path = '/tmp/mlflow/'
        if path:
            hdfs_path = path
        hdfs = None
        try:
            hdfs = self._create_hdfs_conn()
            if (hdfs.exists(hdfs_path)):
                paths = hdfs.glob(hdfs_path + "/*")
                if (paths and len(paths) > 0):
                    infos = []
                    for f in paths:
                        infos.append(FileInfo(f, hdfs.isdir(f), hdfs.info(f).get("size")))
                    return sorted(infos, key=lambda f: paths)
            return paths
        finally:
            if hdfs:
                hdfs.disconnect()

    def _hdfs_download(self, output_path=None, remote_path=None):
        if (output_path is None or output_path == ''):
            raise Exception("Invalid output path: '%s'. %s" % (output_path,
                                                               bad_path_message(output_path)))
        hdfs = None
        try:
            hdfs = self._create_hdfs_conn()
            rootdir_name = os.path.split(os.path.dirname(remote_path))[1]
            for subdir, dirs, files in hdfs.walk(remote_path):
                subdir_local_path = output_path + os.sep + self.extract_child(subdir, rootdir_name)
                os.makedirs(subdir_local_path, exist_ok=True)
                for file in files:
                    filepath = subdir + os.sep + file
                    hdfs.get(filepath, subdir_local_path + "/" + file)
                    # mv directory
        finally:
            if hdfs:
                hdfs.disconnect()

    def _download_file(self, remote_file_path, local_path):
        self._hdfs_download(output_path=local_path,
                            remote_path=remote_file_path)
