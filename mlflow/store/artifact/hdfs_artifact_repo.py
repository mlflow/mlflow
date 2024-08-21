import os
import posixpath
import tempfile
import urllib.parse
from contextlib import contextmanager

import packaging.version

from mlflow.entities import FileInfo
from mlflow.environment_variables import (
    MLFLOW_KERBEROS_TICKET_CACHE,
    MLFLOW_KERBEROS_USER,
    MLFLOW_PYARROW_EXTRA_CONF,
)
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import mkdir, relative_path_to_artifact_path



class HdfsArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on HDFS.

    This repository is used with URIs of the form ``hdfs:/<path>``. The repository can only be used
    together with the RestStore.
    """

    def __init__(self, artifact_uri):
        self.scheme, self.host, self.port, self.path = _resolve_connection_params(artifact_uri)
        super().__init__(artifact_uri)

    def log_artifact(self, local_file, artifact_path=None):
        """
        Log artifact in hdfs.

        Args:
            local_file: Source file path.
            artifact_path: When specified will attempt to write under artifact_uri/artifact_path.
        """
        hdfs_base_path = _resolve_base_path(self.path, artifact_path)

        with hdfs_system(scheme=self.scheme, host=self.host, port=self.port) as hdfs:
            _, file_name = os.path.split(local_file)
            destination = posixpath.join(hdfs_base_path, file_name)
            with open(local_file, "rb") as f:
                hdfs.upload(destination, f)

    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log artifacts in hdfs.
        Missing remote sub-directories will be created if needed.

        Args:
            local_dir: Source dir path.
            artifact_path: When specified will attempt to write under artifact_uri/artifact_path.
        """
        hdfs_base_path = _resolve_base_path(self.path, artifact_path)

        with hdfs_system(scheme=self.scheme, host=self.host, port=self.port) as hdfs:
            if not hdfs.exists(hdfs_base_path):
                hdfs.mkdir(hdfs_base_path)

            for subdir_path, _, files in os.walk(local_dir):
                relative_path = _relative_path_local(local_dir, subdir_path)

                hdfs_subdir_path = (
                    posixpath.join(hdfs_base_path, relative_path)
                    if relative_path
                    else hdfs_base_path
                )

                if not hdfs.exists(hdfs_subdir_path):
                    hdfs.mkdir(hdfs_subdir_path)

                for each_file in files:
                    source = os.path.join(subdir_path, each_file)
                    destination = posixpath.join(hdfs_subdir_path, each_file)
                    self.thread_pool.submit(
                        HdfsArtifactRepository._upload_dir_files_to_hdfs,
                        hdfs=hdfs, 
                        source=source, 
                        destination=destination,
                    ).result()
                    

    @staticmethod
    def _upload_dir_files_to_hdfs(hdfs, source, destination):
        with open(source, "rb") as f:
            hdfs.upload(destination, f)

    def list_artifacts(self, path=None):
        """
        Lists files and directories under artifacts directory for the current run_id.
        (self.path contains the base path - hdfs:/some/path/run_id/artifacts)

        Args:
            path: Relative source path. Possible subdirectory existing under
                hdfs:/some/path/run_id/artifacts

        Returns:
            List of FileInfos under given path
        """
        hdfs_base_path = _resolve_base_path(self.path, path)

        with hdfs_system(scheme=self.scheme, host=self.host, port=self.port) as hdfs:
            paths = []
            if hdfs.exists(hdfs_base_path):
                for file_detail in hdfs.ls(hdfs_base_path, detail=True):
                    file_name = file_detail.get("name")

                    # file_name is hdfs_base_path and not a child of that path
                    if file_name == hdfs_base_path:
                        continue

                    # Strip off anything that comes before the artifact root e.g. hdfs://name
                    offset = file_name.index(self.path)
                    rel_path = _relative_path_remote(self.path, file_name[offset:])
                    is_dir = file_detail.get("kind") == "directory"
                    size = file_detail.get("size")
                    paths.append(FileInfo(rel_path, is_dir=is_dir, file_size=size))
            return sorted(paths, key=lambda f: paths)

    def _walk_path(self, hdfs, hdfs_path):
        if hdfs.exists(hdfs_path):
            if hdfs.isdir(hdfs_path):
                for subdir, _, files in hdfs.walk(hdfs_path):
                    if subdir != hdfs_path:
                        yield subdir, hdfs.isdir(subdir), hdfs.info(subdir).get("size")
                    for f in files:
                        file_path = posixpath.join(subdir, f)
                        yield file_path, hdfs.isdir(file_path), hdfs.info(file_path).get("size")
            else:
                yield hdfs_path, False, hdfs.info(hdfs_path).get("size")

    def download_artifacts(self, artifact_path, dst_path=None):
        """
        Download an artifact file or directory to a local directory/file if applicable, and
        return a local path for it.
        The caller is responsible for managing the lifecycle of the downloaded artifacts.

        (self.path contains the base path - hdfs:/some/path/run_id/artifacts)

        Args:
            artifact_path: Relative source path to the desired artifacts file or directory.
            dst_path: Absolute path of the local filesystem destination directory to which
                to download the specified artifacts. This directory must already exist. If
                unspecified, the artifacts will be downloaded to a new, uniquely-named
                directory on the local filesystem.

        Returns:
            Absolute path of the local filesystem location containing the downloaded
            artifacts - file/directory.
        """

        hdfs_base_path = _resolve_base_path(self.path, artifact_path)
        if dst_path and os.path.exists(dst_path):
            local_dir = os.path.abspath(dst_path)
        else:
            local_dir = _tmp_dir(dst_path)

        with hdfs_system(scheme=self.scheme, host=self.host, port=self.port) as hdfs:
            if not hdfs.isdir(hdfs_base_path):
                local_path = os.path.join(local_dir, os.path.normpath(artifact_path))
                _download_hdfs_file(hdfs, hdfs_base_path, local_path)
                return local_path

            for path, is_dir, _ in self._walk_path(hdfs, hdfs_base_path):
                relative_path = _relative_path_remote(hdfs_base_path, path)
                local_path = os.path.join(local_dir, relative_path) if relative_path else local_dir

                if is_dir:
                    mkdir(local_path)
                else:
                    _download_hdfs_file(hdfs, path, local_path)
            return local_dir

    def _download_file(self, remote_file_path, local_path):
        raise MlflowException("This is not implemented. Should never be called.")

    def delete_artifacts(self, artifact_path=None):
        path = posixpath.join(self.path, artifact_path) if artifact_path else self.path
        with hdfs_system(scheme=self.scheme, host=self.host, port=self.port) as hdfs:
            hdfs.delete(path, recursive=True)


@contextmanager
def hdfs_system(scheme, host, port):
    """
    hdfs system context - Attempt to establish the connection to hdfs
    and yields HadoopFileSystem

    Args:
        scheme: scheme or use hdfs:// as default
        host: hostname or when relaying on the core-site.xml config use 'default'
        port: port or when relaying on the core-site.xml config use 0
    """
    import pyarrow

    kerb_ticket = MLFLOW_KERBEROS_TICKET_CACHE.get()
    kerberos_user = MLFLOW_KERBEROS_USER.get()
    extra_conf = _parse_extra_conf(MLFLOW_PYARROW_EXTRA_CONF.get())

    host = scheme + "://" + host if host else "default"

    if packaging.version.parse(pyarrow.__version__) < packaging.version.parse("2.0.0"):
        connected = pyarrow.fs.HadoopFileSystem(
            host=host,
            port=port or 0,
            user=kerberos_user,
            kerb_ticket=kerb_ticket,
            extra_conf=extra_conf,
        )
    else:
        connected = pyarrow.hdfs.connect(
            host=host,
            port=port or 0,
            user=kerberos_user,
            kerb_ticket=kerb_ticket,
            extra_conf=extra_conf,
        )
    yield connected
    connected.close()


def _resolve_connection_params(artifact_uri):
    parsed = urllib.parse.urlparse(artifact_uri)

    return parsed.scheme, parsed.hostname, parsed.port, parsed.path


def _resolve_base_path(path, artifact_path):
    if path == artifact_path:
        return path
    if artifact_path:
        return posixpath.join(path, artifact_path)
    return path


def _relative_path(base_dir, subdir_path, path_module):
    relative_path = path_module.relpath(subdir_path, base_dir)
    return relative_path if relative_path != "." else None


def _relative_path_local(base_dir, subdir_path):
    rel_path = _relative_path(base_dir, subdir_path, os.path)
    return relative_path_to_artifact_path(rel_path) if rel_path is not None else None


def _relative_path_remote(base_dir, subdir_path):
    return _relative_path(base_dir, subdir_path, posixpath)


def _tmp_dir(local_path):
    return os.path.abspath(tempfile.mkdtemp(dir=local_path))


def _download_hdfs_file(hdfs, remote_file_path, local_file_path):
    # Ensure all required directories exist. Without doing this nested files can't be downloaded.
    dirs = os.path.dirname(local_file_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    with open(local_file_path, "wb") as f:
        f.write(hdfs.open(remote_file_path, "rb").read())


def _parse_extra_conf(extra_conf):
    if extra_conf:

        def as_pair(config):
            key, val = config.split("=")
            return key, val

        list_of_key_val = [as_pair(conf) for conf in extra_conf.split(",")]
        return dict(list_of_key_val)
    return None
