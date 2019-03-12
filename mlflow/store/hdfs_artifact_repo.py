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
        import pyarrow as pa
        driver = 'libhdfs'
        if "MLFLOW_HDFS_DRIVER" in os.environ:
            driver = os.environ["MLFLOW_HDFS_DRIVER"]
        hdfs = pa.hdfs.connect(host=self.config["host"],
                               port=int(self.config["port"]), driver=driver)
        return hdfs

    def log_artifact(self, local_file, artifact_path=None):
        if artifact_path:
            dest_path = artifact_path
        else:
            dest_path = self.path
        hdfs = None
        try:
            hdfs = self._create_hdfs_conn()
            with hdfs.open(dest_path, 'wb') as hdf:
                hdf.write(open(local_file, "rb").read())
        finally:
            if hdfs:
                hdfs.close()

    def log_artifacts(self, local_dir, artifact_path=None):
        if artifact_path and path_not_unique(artifact_path):
            raise Exception("Invalid artifact path: '%s'. %s" % (artifact_path,
                                                                 bad_path_message(artifact_path)))
        hdfs = None
        try:
            hdfs = self._create_hdfs_conn()
            hdfs_model_path = self.path + os.sep + artifact_path + "/model/"
            if not (hdfs.exists(hdfs_model_path)):
                hdfs.mkdir(hdfs_model_path)
            rootdir_name = os.path.split(os.path.dirname(local_dir))[1]
            for subdir, _dirs, files in os.walk(local_dir):
                hdfs_subdir_path = self.path + os.sep \
                                   + artifact_path + os.sep \
                                   + self.extract_child(subdir,
                                                        rootdir_name)
                if not (hdfs.exists(hdfs_subdir_path)):
                    hdfs.mkdir(hdfs_subdir_path)
                for each_file in files:
                    filepath = subdir + os.sep + each_file
                    with hdfs.open(hdfs_subdir_path + os.sep + each_file, 'wb') as hdf:
                        hdf.write(open(filepath, "rb").read())
        finally:
            if hdfs:
                hdfs.close()

    def extract_child(self, path, rootdir_name):
        splitpaths = path.split(os.sep + rootdir_name + os.sep)
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
                infos = []
                for subdir, _dirs, files in hdfs.walk(hdfs_path):
                    infos.append(FileInfo(subdir, hdfs.isdir(subdir),
                                          hdfs.info(subdir).get("size")))
                    for each_file in files:
                        filepath = subdir + os.sep + each_file
                        infos.append(FileInfo(filepath, hdfs.isdir(filepath),
                                              hdfs.info(filepath).get("size")))
                return sorted(infos, key=lambda f: paths)
            return paths
        finally:
            if hdfs:
                hdfs.close()

    def _hdfs_download(self, output_path=None, remote_path=None):
        if (output_path is None or output_path == ''):
            raise Exception("Invalid output path: '%s'. %s" % (output_path,
                                                               bad_path_message(output_path)))
        hdfs = None
        try:
            hdfs = self._create_hdfs_conn()
            rootdir_name = os.path.split(os.path.dirname(remote_path))[1]
            for subdir, _dirs, files in hdfs.walk(remote_path):
                subdir_local_path = output_path + os.sep + self.extract_child(subdir, rootdir_name)
                if (not self.get_path_module().exists(subdir_local_path)):
                    os.makedirs(subdir_local_path)
                for each_file in files:
                    filepath = subdir + os.sep + each_file
                    local_file_path = subdir_local_path + os.sep + each_file
                    with open(local_file_path, 'wb') as f:
                        f.write(hdfs.open(filepath, 'rb').read())
        finally:
            if hdfs:
                hdfs.close()

    def _download_file(self, remote_file_path, local_path):
        self._hdfs_download(output_path=local_path,
                            remote_path=remote_file_path)
