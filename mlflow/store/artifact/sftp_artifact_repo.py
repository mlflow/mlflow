import os
import stat
import posixpath
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


class SFTPArtifactRepository(ArtifactRepository):
    """Stores artifacts as files in a remote directory, via sftp."""

    def __init__(self, artifact_uri, client=None):
        self.uri = artifact_uri
        parsed = urllib.parse.urlparse(artifact_uri)
        self.config = {
            "hostname": parsed.hostname,
            "port": parsed.port,
            "username": parsed.username,
            "password": parsed.password,
        }
        self.path = parsed.path

        if client:
            self.sftp = client
        else:
            import paramiko

            if self.config["hostname"] is None:
                self.config["hostname"] = "localhost"

            ssh_config = paramiko.SSHConfig()
            user_config_file = os.path.expanduser("~/.ssh/config")
            if os.path.exists(user_config_file):
                with open(user_config_file) as f:
                    ssh_config.parse(f)

            user_config = ssh_config.lookup(self.config["hostname"])

            if "hostname" in user_config:
                self.config["hostname"] = user_config["hostname"]

            if self.config.get("username", None) is None and "user" in user_config:
                self.config["username"] = user_config["user"]

            if self.config.get("port", None) is None:
                if "port" in user_config:
                    self.config["port"] = int(user_config["port"])
                else:
                    self.config["port"] = 22

            if "identityfile" in user_config:
                self.config["key_filename"] = user_config["identityfile"][0]

            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.connect(**self.config)
            self.sftp = ssh_client.open_sftp()

        super().__init__(artifact_uri)

    def _isfile(self, remotepath):
        try:
            return stat.S_ISREG(self.sftp.stat(remotepath).st_mode)
        except IOError:
            return False

    def _isdir(self, remotepath):
        try:
            return stat.S_ISDIR(self.sftp.stat(remotepath).st_mode)
        except IOError:
            return False

    def _mkdir(self, remotedir):
        self.sftp.mkdir(remotedir)

    def _makedirs(self, remotedir):
        if self._isdir(remotedir):
            pass

        elif self._isfile(remotedir):
            raise OSError(
                "a file with the same name as the remotedir, " "'%s', already exists." % remotedir
            )
        else:
            head, tail = os.path.split(remotedir)
            if head and not self._isdir(head):
                self._makedirs(head)
            if tail:
                self._mkdir(remotedir)

    def _put(self, localpath, remotepath):
        self.sftp.put(localpath, remotepath)

    # https://stackoverflow.com/a/58466685
    def _put_r_portable(self, localdir, remotedir):
        for entry in os.listdir(localdir):
            remotepath = posixpath.join(remotedir, entry)
            localpath = os.path.join(localdir, entry)
            if not os.path.isfile(localpath):
                try:
                    self._mkdir(remotepath)
                except OSError:
                    pass
                self._put_r_portable(localpath, remotepath)
            else:
                self._put(localpath, remotepath)

    def _listdir(self, remotepath="."):
        return self.sftp.listdir(remotepath)

    @contextmanager
    def _cd(self, remotepath=None):
        original_cwd = self.sftp.getcwd()
        try:
            if remotepath is not None:
                self.sftp.chdir(remotepath)
            yield
        finally:
            self.sftp.chdir(original_cwd)

    def log_artifact(self, local_file, artifact_path=None):
        artifact_dir = posixpath.join(self.path, artifact_path) if artifact_path else self.path
        self._makedirs(artifact_dir)
        self._put(local_file, posixpath.join(artifact_dir, os.path.basename(local_file)))

    def log_artifacts(self, local_dir, artifact_path=None):
        artifact_dir = posixpath.join(self.path, artifact_path) if artifact_path else self.path
        self._makedirs(artifact_dir)
        self._put_r_portable(local_dir, artifact_dir)

    def _is_directory(self, artifact_path):
        artifact_dir = self.path
        path = posixpath.join(artifact_dir, artifact_path) if artifact_path else artifact_dir
        return self._isdir(path)

    def list_artifacts(self, path=None):
        artifact_dir = self.path
        list_dir = posixpath.join(artifact_dir, path) if path else artifact_dir
        if not self._isdir(list_dir):
            return []
        artifact_files = self._listdir(list_dir)
        infos = []
        for file_name in artifact_files:
            file_path = file_name if path is None else posixpath.join(path, file_name)
            full_file_path = posixpath.join(list_dir, file_name)
            if self._isdir(full_file_path):
                infos.append(FileInfo(file_path, True, None))
            else:
                infos.append(FileInfo(file_path, False, self.sftp.stat(full_file_path).st_size))
        return sorted(infos, key=lambda f: (f.is_dir, f.path))

    def _download_file(self, remote_file_path, local_path):
        remote_full_path = posixpath.join(self.path, remote_file_path)
        self.sftp.get(remote_full_path, local_path)

    def delete_artifacts(self, artifact_path=None):
        if self._isdir(artifact_path):
            with self._cd(artifact_path):
                for element in self._listdir():
                    self.delete_artifacts(element)
            self.sftp.rmdir(artifact_path)
        elif self._isfile(artifact_path):
            self.sftp.remove(artifact_path)
