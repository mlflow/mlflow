import os
import sys

import posixpath
import urllib.parse

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


class SFTPArtifactRepository(ArtifactRepository):
    """Stores artifacts as files in a remote directory, via sftp."""

    def __init__(self, artifact_uri, client=None):
        self.uri = artifact_uri
        parsed = urllib.parse.urlparse(artifact_uri)
        self.config = {
            "host": parsed.hostname,
            "port": parsed.port,
            "username": parsed.username,
            "password": parsed.password,
        }
        self.path = parsed.path

        if client:
            self.sftp = client
        else:
            import pysftp
            import paramiko

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

            self.sftp = pysftp.Connection(**self.config)

        super().__init__(artifact_uri)

    def log_artifact(self, local_file, artifact_path=None):
        artifact_dir = posixpath.join(self.path, artifact_path) if artifact_path else self.path
        self.sftp.makedirs(artifact_dir)
        self.sftp.put(local_file, posixpath.join(artifact_dir, os.path.basename(local_file)))

    def log_artifacts(self, local_dir, artifact_path=None):
        artifact_dir = posixpath.join(self.path, artifact_path) if artifact_path else self.path
        self.sftp.makedirs(artifact_dir)
        if sys.platform == "win32":
            _put_r_for_windows(self.sftp, local_dir, artifact_dir)
        else:
            self.sftp.put_r(local_dir, artifact_dir)

    def _is_directory(self, artifact_path):
        artifact_dir = self.path
        path = posixpath.join(artifact_dir, artifact_path) if artifact_path else artifact_dir
        return self.sftp.isdir(path)

    def list_artifacts(self, path=None):
        artifact_dir = self.path
        list_dir = posixpath.join(artifact_dir, path) if path else artifact_dir
        if not self.sftp.isdir(list_dir):
            return []
        artifact_files = self.sftp.listdir(list_dir)
        infos = []
        for file_name in artifact_files:
            file_path = file_name if path is None else posixpath.join(path, file_name)
            full_file_path = posixpath.join(list_dir, file_name)
            if self.sftp.isdir(full_file_path):
                infos.append(FileInfo(file_path, True, None))
            else:
                infos.append(FileInfo(file_path, False, self.sftp.stat(full_file_path).st_size))
        return infos

    def _download_file(self, remote_file_path, local_path):
        remote_full_path = posixpath.join(self.path, remote_file_path)
        self.sftp.get(remote_full_path, local_path)

    def delete_artifacts(self, artifact_path=None):
        if self.sftp.isdir(artifact_path):
            with self.sftp.cd(artifact_path):
                for element in self.sftp.listdir():
                    self.delete_artifacts(element)
            self.sftp.rmdir(artifact_path)
        elif self.sftp.isfile(artifact_path):
            self.sftp.remove(artifact_path)
