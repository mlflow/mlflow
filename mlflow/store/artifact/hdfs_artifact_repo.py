import os
import posixpath
import urllib.parse
from contextlib import contextmanager
from typing import Optional

try:
    from pyarrow.fs import FileSelector, FileType, HadoopFileSystem
except ImportError:
    pass

from mlflow.entities import FileInfo
from mlflow.environment_variables import (
    MLFLOW_KERBEROS_TICKET_CACHE,
    MLFLOW_KERBEROS_USER,
    MLFLOW_PYARROW_EXTRA_CONF,
)
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import relative_path_to_artifact_path


class HdfsArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on HDFS.

    This repository is used with URIs of the form ``hdfs:/<path>``. The repository can only be used
    together with the RestStore.
    """

    def __init__(self, artifact_uri: str, tracking_uri: Optional[str] = None) -> None:
        super().__init__(artifact_uri, tracking_uri)
        self.scheme, self.host, self.port, self.path = _resolve_connection_params(artifact_uri)

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
            destination_path = posixpath.join(hdfs_base_path, file_name)
            with open(local_file, "rb") as source:
                with hdfs.open_output_stream(destination_path) as destination:
                    destination.write(source.read())

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
            if not hdfs.get_file_info(hdfs_base_path).type == FileType.Directory:
                hdfs.create_dir(hdfs_base_path, recursive=True)

            for subdir_path, _, files in os.walk(local_dir):
                relative_path = _relative_path_local(local_dir, subdir_path)

                hdfs_subdir_path = (
                    posixpath.join(hdfs_base_path, relative_path)
                    if relative_path
                    else hdfs_base_path
                )

                if not hdfs.get_file_info(hdfs_subdir_path).type == FileType.Directory:
                    hdfs.create_dir(hdfs_subdir_path, recursive=True)

                for each_file in files:
                    source_path = os.path.join(subdir_path, each_file)
                    destination_path = posixpath.join(hdfs_subdir_path, each_file)
                    with open(source_path, "rb") as source:
                        with hdfs.open_output_stream(destination_path) as destination:
                            destination.write(source.read())

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
            base_info = hdfs.get_file_info(hdfs_base_path)
            if base_info.type == FileType.Directory:
                selector = FileSelector(hdfs_base_path)
            elif base_info.type == FileType.File:
                selector = [hdfs_base_path]
            else:
                return []

            for file_detail in hdfs.get_file_info(selector):
                file_name = file_detail.path

                # file_name is hdfs_base_path and not a child of that path
                if file_name == hdfs_base_path:
                    continue

                # Strip off anything that comes before the artifact root e.g. hdfs://name
                offset = file_name.index(self.path)
                rel_path = _relative_path_remote(self.path, file_name[offset:])
                is_dir = file_detail.type == FileType.Directory
                size = file_detail.size
                paths.append(FileInfo(rel_path, is_dir=is_dir, file_size=size))
            return sorted(paths, key=lambda f: paths)

    def _is_directory(self, artifact_path):
        hdfs_base_path = _resolve_base_path(self.path, artifact_path)
        with hdfs_system(scheme=self.scheme, host=self.host, port=self.port) as hdfs:
            return hdfs.get_file_info(hdfs_base_path).type == FileType.Directory

    def _download_file(self, remote_file_path, local_path):
        hdfs_base_path = _resolve_base_path(self.path, remote_file_path)
        with hdfs_system(scheme=self.scheme, host=self.host, port=self.port) as hdfs:
            with hdfs.open_input_stream(hdfs_base_path) as source:
                with open(local_path, "wb") as destination:
                    destination.write(source.read())

    def delete_artifacts(self, artifact_path=None):
        path = posixpath.join(self.path, artifact_path) if artifact_path else self.path
        with hdfs_system(scheme=self.scheme, host=self.host, port=self.port) as hdfs:
            file_info = hdfs.get_file_info(path)
            if file_info.type == FileType.File:
                hdfs.delete_file(path)
            elif file_info.type == FileType.Directory:
                hdfs.delete_dir_contents(path)


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
    kerb_ticket = MLFLOW_KERBEROS_TICKET_CACHE.get()
    kerberos_user = MLFLOW_KERBEROS_USER.get()
    extra_conf = _parse_extra_conf(MLFLOW_PYARROW_EXTRA_CONF.get())

    host = scheme + "://" + host if host else "default"

    yield HadoopFileSystem(
        host=host,
        port=port or 0,
        user=kerberos_user,
        kerb_ticket=kerb_ticket,
        extra_conf=extra_conf,
    )


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


def _parse_extra_conf(extra_conf):
    if extra_conf:

        def as_pair(config):
            key, val = config.split("=")
            return key, val

        list_of_key_val = [as_pair(conf) for conf in extra_conf.split(",")]
        return dict(list_of_key_val)
    return None
