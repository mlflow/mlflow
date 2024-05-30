import logging
import os
import posixpath
import urllib.parse
from contextlib import contextmanager

from mlflow.entities import FileInfo
from mlflow.environment_variables import (
    MLFLOW_KERBEROS_TICKET_CACHE,
    MLFLOW_KERBEROS_USER,
    MLFLOW_PYARROW_EXTRA_CONF,
)
from mlflow.store.artifact.artifact_repo import ArtifactRepository

_logger = logging.getLogger(__name__)


class HdfsArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on HDFS.

    This repository is used with URIs of the form ``hdfs:/<path>``. The repository can only be used
    together with the RestStore.
    """

    def __init__(self, artifact_uri):
        self.scheme, self.host, self.port, self.path = _resolve_connection_params(artifact_uri)
        self.path = posixpath.join(self.path, "")
        super().__init__(artifact_uri)

    def log_artifact(self, local_file, artifact_path=None):
        """
        Log artifact in hdfs.

        Args:
            local_file: Source file path.
            artifact_path: When specified will attempt to write under artifact_uri/artifact_path.
        """
        hdfs_base_path = posixpath.join(_resolve_base_path(self.path, artifact_path), "")

        with hdfs_system(scheme=self.scheme, host=self.host, port=self.port) as hdfs:
            hdfs.put(local_file, hdfs_base_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log artifacts in hdfs.
        Missing remote sub-directories will be created if needed.

        Args:
            local_dir: Source dir path.
            artifact_path: When specified will attempt to write under artifact_uri/artifact_path.
        """
        hdfs_base_path = posixpath.join(_resolve_base_path(self.path, artifact_path), "")

        with hdfs_system(scheme=self.scheme, host=self.host, port=self.port) as hdfs:
            hdfs.put(os.path.join(local_dir, ""), hdfs_base_path, recursive=True)

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
            try:
                if hdfs.isdir(hdfs_base_path):
                    details = hdfs.ls(hdfs_base_path, detail=True)
                else:
                    return []
            except OSError:
                _logger.exception("Error listing directory/file")
                return []

            paths = []
            for file_detail in details:
                file_name = file_detail.get("name")

                # Strip off anything that comes before the artifact root e.g. hdfs://name
                offset = file_name.index(self.path)
                rel_path = _relative_path_remote(self.path, file_name[offset:])
                is_dir = file_detail.get("type") == "directory"
                size = file_detail.get("size")
                paths.append(FileInfo(rel_path, is_dir=is_dir, file_size=size))

            return sorted(paths, key=lambda f: paths)

    def _download_file(self, remote_file_path, local_path):
        hdfs_base_path = _resolve_base_path(self.path, remote_file_path)
        with hdfs_system(scheme=self.scheme, host=self.host, port=self.port) as hdfs:
            hdfs.get_file(hdfs_base_path, local_path)

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
    from fsspec.implementations.arrow import HadoopFileSystem

    kerb_ticket = MLFLOW_KERBEROS_TICKET_CACHE.get()
    kerberos_user = MLFLOW_KERBEROS_USER.get()
    extra_conf = _parse_extra_conf(MLFLOW_PYARROW_EXTRA_CONF.get())

    host = scheme + "://" + host if host else "default"

    connected = HadoopFileSystem(
        host=host,
        port=port or 0,
        user=kerberos_user,
        kerb_ticket=kerb_ticket,
        extra_conf=extra_conf,
        auto_mkdir=True,
    )

    yield connected


def _resolve_connection_params(artifact_uri):
    parsed = urllib.parse.urlparse(artifact_uri)

    return parsed.scheme, parsed.hostname, parsed.port, parsed.path


def _resolve_base_path(path, artifact_path):
    if path == artifact_path:
        return path
    if artifact_path:
        return posixpath.join(path, artifact_path)
    return path


def _relative_path_remote(base_dir, subdir_path):
    relative_path = posixpath.relpath(subdir_path, base_dir)
    return relative_path if relative_path != "." else None


def _parse_extra_conf(extra_conf):
    if extra_conf:

        def as_pair(config):
            key, val = config.split("=")
            return key, val

        list_of_key_val = [as_pair(conf) for conf in extra_conf.split(",")]
        return dict(list_of_key_val)
    return None
