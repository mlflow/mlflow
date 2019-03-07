import os

from hdfs3 import HDFileSystem
from six.moves import urllib

from mlflow.exceptions import MlflowException
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils.string_utils import strip_prefix
from mlflow.utils.validation import path_not_unique, bad_path_message



class HdfsArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on HDFS.

    This repository is used with URIs of the form ``hdfs:/<path>``. The repository can only be used
    together with the RestStore.
    """

    def __init__(self, artifact_uri):
        cleaned_artifact_uri = artifact_uri.rstrip('/')
        parsed = urllib.parse.urlparse(artifact_uri)
        self.config = {
            'host': parsed.hostname,
            'port': 8020 if parsed.port is None else parsed.port
        }
        self.path = parsed.path

        if self.config['host'] is None:
            self.config['host'] = 'localhost'
        super(HdfsArtifactRepository, self).__init__(cleaned_artifact_uri)
        if not cleaned_artifact_uri.startswith('hdfs:/'):
            raise MlflowException('hdfsArtifactRepository URI must start with hdfs:/')


    def _get_hdfs_path(self, artifact_path):
        return '/%s/%s' % (strip_prefix(self.artifact_uri, 'hdfs:/'),
                           strip_prefix(artifact_path, '/'))

    def _create_hdfs(self):
        hdfs = HDFileSystem(host=self.config["host"],
                            port=int(self.config["port"]))
        return hdfs

    def log_artifact(self, local_file, artifact_path=None):
        if artifact_path:
            dest_path=artifact_path
        else:
            dest_path=self.path
        try:
            hdfs=self._create_hdfs()
            hdfs.put(local_file, dest_path)
        finally:
            if (not hdfs is None):
                hdfs.disconnect()


    def log_artifacts(self, local_dir, artifact_path=None):
        if artifact_path and path_not_unique(artifact_path):
            raise Exception("Invalid artifact path: '%s'. %s" % (artifact_path,
                                                                 bad_path_message(artifact_path)))
        try:
            hdfs=self._create_hdfs()
            hdfs_model_path=self.path+os.sep +artifact_path+"/model/"
            if not (hdfs.isdir(hdfs_model_path)):
                hdfs.mkdir(artifact_path)
            rootdir_name = os.path.split(os.path.dirname(local_dir))[1]
            for subdir, dirs, files in os.walk(local_dir):
                hdfs_subdir_path = self.path+artifact_path + os.sep + self.extract_child(subdir, rootdir_name)
                if not (hdfs.isdir(hdfs_subdir_path)):
                    hdfs.mkdir(hdfs_subdir_path)
                for file in files:
                    filepath = subdir + os.sep + file
                    hdfs.put(filepath, hdfs_subdir_path + "/" + file)
        finally:
            if (not hdfs is None):
                hdfs.disconnect()

    def extract_child(self,path, rootdir_name):
        splitpaths = path.split(os.sep + rootdir_name + os.sep)
        index = 0
        child_dir_path = ''
        for index in range(1, len(splitpaths)):
            childdirs = splitpaths[index].split(os.sep)
            for each_dir in childdirs:
                child_dir_path = child_dir_path + os.sep + each_dir
        return child_dir_path

    def list_artifacts(self, path=None):
        if path:
            hdfs_path = self._get_hdfs_path(path)
        else:
            hdfs_path = self._get_hdfs_path('')
        hdfs=None
        try:
            hdfs = self._create_hdfs()
            if(hdfs.isdir(hdfs_path)):
                return hdfs.ls(hdfs_path)
        finally:
            if not hdfs is None:
                hdfs.disconnect()

    def _hdfs_download(self,output_path=None,remote_path=None):
        if(output_path is None or output_path==''):
            raise Exception("Invalid output path: '%s'. %s" % (output_path,
                                                                 bad_path_message(output_path)))
        hdfs = None
        try:
            hdfs = self._create_hdfs()
            rootdir_name = os.path.split(os.path.dirname(remote_path))[1]
            for subdir, dirs, files in hdfs.walk(remote_path):
                subdir_local_path = output_path + os.sep +self.extract_child(subdir, rootdir_name)
                os.makedirs(subdir_local_path, exist_ok=True)
                for file in files:
                    filepath = subdir + os.sep + file
                    hdfs.get(filepath, subdir_local_path + "/" + file)
                    # mv directory
        finally:
            if not hdfs is None:
                hdfs.disconnect()

    def _download_file(self, remote_path, local_path):
        self._hdfs_download(output_path=local_path,
                            remote_path=remote_path)
