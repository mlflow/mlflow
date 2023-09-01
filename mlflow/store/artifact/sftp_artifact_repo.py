import os
import posixpath
import sys
import threading
import urllib.parse
from contextlib import contextmanager

from mlflow.entities import FileInfo
from mlflow.store.artifact.artifact_repo import ArtifactRepository


# Based on: https://stackoverflow.com/a/58466685
def _put_r_for_windows(sftp, local_dir, remote_dir, preserve_mtime=False):
    for entry in os.listdir(local_dir):
        local_path = os.path.join(local_dir, entry)
        remote_path = posixpath.join(remote_dir, entry)
        if os.path.isdir(local_path):
            sftp.mkdir(remote_path)
            _put_r_for_windows(sftp, local_path, remote_path, preserve_mtime)
        else:
            sftp.put(local_path, remote_path, preserve_mtime=preserve_mtime)


class _SftpPool:
    def __init__(self, connections):
        self._connections = connections
        self._cond = threading.Condition()

    @contextmanager
    def get_sfp_connection(self):
        with self._cond:
            self._cond.wait_for(lambda: bool(self._connections))
            connection = self._connections.pop(-1)
        yield connection
        self._connections.append(connection)


class SFTPArtifactRepository(ArtifactRepository):
    """Stores artifacts as files in a remote directory, via sftp."""

    def __init__(self, artifact_uri):
        self.uri = artifact_uri
        parsed = urllib.parse.urlparse(artifact_uri)
        self.config = {
            "host": parsed.hostname,
            "port": parsed.port,
            "username": parsed.username,
            "password": parsed.password,
        }
        self.path = parsed.path or "/"

        import paramiko
        import pysftp

        if self.config["host"] is None:
            self.config["host"] = "localhost"

        ssh_config = paramiko.SSHConfig()
        user_config_file = os.path.expanduser("~/.ssh/config")
        if os.path.exists(user_config_file):
            with open(user_config_file) as f:
                ssh_config.parse(f)

        user_config = ssh_config.lookup(self.config["host"])

        if "hostname" in user_config:
            self.config["host"] = user_config["hostname"]

        if self.config.get("username", None) is None and "user" in user_config:
            self.config["username"] = user_config["user"]

        if self.config.get("port", None) is None:
            if "port" in user_config:
                self.config["port"] = int(user_config["port"])
            else:
                self.config["port"] = 22

        if "identityfile" in user_config:
            self.config["private_key"] = user_config["identityfile"][0]

        connections = [pysftp.Connection(**self.config) for _ in range(self.max_workers)]
        self.pool = _SftpPool(connections)

        super().__init__(artifact_uri)

    def log_artifact(self, local_file, artifact_path=None):
        artifact_dir = posixpath.join(self.path, artifact_path) if artifact_path else self.path
        with self.pool.get_sfp_connection() as sftp:
            sftp.makedirs(artifact_dir)
            sftp.put(local_file, posixpath.join(artifact_dir, os.path.basename(local_file)))

    def log_artifacts(self, local_dir, artifact_path=None):
        artifact_dir = posixpath.join(self.path, artifact_path) if artifact_path else self.path
        with self.pool.get_sfp_connection() as sftp:
            sftp.makedirs(artifact_dir)
            if sys.platform == "win32":
                _put_r_for_windows(sftp, local_dir, artifact_dir)
            else:
                sftp.put_r(local_dir, artifact_dir)

    def _is_directory(self, artifact_path):
        artifact_dir = self.path
        path = posixpath.join(artifact_dir, artifact_path) if artifact_path else artifact_dir
        with self.pool.get_sfp_connection() as sftp:
            return sftp.isdir(path)

    def list_artifacts(self, path=None):
        artifact_dir = self.path
        list_dir = posixpath.join(artifact_dir, path) if path else artifact_dir
        with self.pool.get_sfp_connection() as sftp:
            if not sftp.isdir(list_dir):
                return []
            artifact_files = sftp.listdir(list_dir)
            infos = []
            for file_name in artifact_files:
                file_path = file_name if path is None else posixpath.join(path, file_name)
                full_file_path = posixpath.join(list_dir, file_name)
                if sftp.isdir(full_file_path):
                    infos.append(FileInfo(file_path, True, None))
                else:
                    infos.append(FileInfo(file_path, False, sftp.stat(full_file_path).st_size))
            return infos

    def _download_file(self, remote_file_path, local_path):
        remote_full_path = posixpath.join(self.path, remote_file_path)
        with self.pool.get_sfp_connection() as sftp:
            sftp.get(remote_full_path, local_path)

    def delete_artifacts(self, artifact_path=None):
        artifact_dir = posixpath.join(self.path, artifact_path) if artifact_path else self.path
        with self.pool.get_sfp_connection() as sftp:
            self._delete_inner(artifact_dir, sftp)

    def _delete_inner(self, artifact_path, sftp):
        if sftp.isdir(artifact_path):
            with sftp.cd(artifact_path):
                for element in sftp.listdir():
                    self._delete_inner(element, sftp)
            sftp.rmdir(artifact_path)
        elif sftp.isfile(artifact_path):
            sftp.remove(artifact_path)
