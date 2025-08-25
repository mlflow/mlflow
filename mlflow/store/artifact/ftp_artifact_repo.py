import ftplib
import os
import posixpath
import urllib.parse
from contextlib import contextmanager
from ftplib import FTP
from urllib.parse import unquote

from mlflow.entities.file_info import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import relative_path_to_artifact_path


class FTPArtifactRepository(ArtifactRepository):
    """Stores artifacts as files in a remote directory, via ftp."""

    def __init__(self, artifact_uri: str, tracking_uri: str | None = None) -> None:
        super().__init__(artifact_uri, tracking_uri)
        parsed = urllib.parse.urlparse(artifact_uri)
        self.config = {
            "host": parsed.hostname,
            "port": 21 if parsed.port is None else parsed.port,
            "username": parsed.username,
            "password": parsed.password,
        }
        self.path = parsed.path or "/"

        if self.config["host"] is None:
            self.config["host"] = "localhost"
        if self.config["password"] is None:
            self.config["password"] = ""
        else:
            self.config["password"] = unquote(parsed.password)

    @contextmanager
    def get_ftp_client(self):
        ftp = FTP()
        ftp.connect(self.config["host"], self.config["port"])
        ftp.login(self.config["username"], self.config["password"])
        yield ftp
        ftp.close()

    @staticmethod
    def _is_dir(ftp, full_file_path):
        try:
            ftp.cwd(full_file_path)
            return True
        except ftplib.error_perm:
            return False

    @staticmethod
    def _mkdir(ftp, artifact_dir):
        try:
            if not FTPArtifactRepository._is_dir(ftp, artifact_dir):
                ftp.mkd(artifact_dir)
        except ftplib.error_perm:
            head, _ = posixpath.split(artifact_dir)
            FTPArtifactRepository._mkdir(ftp, head)
            FTPArtifactRepository._mkdir(ftp, artifact_dir)

    @staticmethod
    def _size(ftp, full_file_path):
        ftp.voidcmd("TYPE I")
        size = ftp.size(full_file_path)
        ftp.voidcmd("TYPE A")
        return size

    def log_artifact(self, local_file, artifact_path=None):
        with self.get_ftp_client() as ftp:
            artifact_dir = posixpath.join(self.path, artifact_path) if artifact_path else self.path
            self._mkdir(ftp, artifact_dir)
            with open(local_file, "rb") as f:
                ftp.cwd(artifact_dir)
                ftp.storbinary("STOR " + os.path.basename(local_file), f)

    def log_artifacts(self, local_dir, artifact_path=None):
        dest_path = posixpath.join(self.path, artifact_path) if artifact_path else self.path

        local_dir = os.path.abspath(local_dir)
        for root, _, filenames in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                rel_upload_path = relative_path_to_artifact_path(rel_path)
                upload_path = posixpath.join(dest_path, rel_upload_path)
            if not filenames:
                with self.get_ftp_client() as ftp:
                    self._mkdir(ftp, upload_path)
            for f in filenames:
                if os.path.isfile(os.path.join(root, f)):
                    self.log_artifact(os.path.join(root, f), upload_path)

    def _is_directory(self, artifact_path):
        artifact_dir = self.path
        list_dir = posixpath.join(artifact_dir, artifact_path) if artifact_path else artifact_dir
        with self.get_ftp_client() as ftp:
            return self._is_dir(ftp, list_dir)

    def list_artifacts(self, path=None):
        with self.get_ftp_client() as ftp:
            artifact_dir = self.path
            list_dir = posixpath.join(artifact_dir, path) if path else artifact_dir
            if not self._is_dir(ftp, list_dir):
                return []
            artifact_files = ftp.nlst(list_dir)
            # Make sure artifact_files is a list of file names because ftp.nlst
            # may return absolute paths.
            artifact_files = [os.path.basename(f) for f in artifact_files]
            artifact_files = list(filter(lambda x: x != "." and x != "..", artifact_files))
            infos = []
            for file_name in artifact_files:
                file_path = file_name if path is None else posixpath.join(path, file_name)
                full_file_path = posixpath.join(list_dir, file_name)
                if self._is_dir(ftp, full_file_path):
                    infos.append(FileInfo(file_path, True, None))
                else:
                    size = self._size(ftp, full_file_path)
                    infos.append(FileInfo(file_path, False, size))
        return infos

    def _download_file(self, remote_file_path, local_path):
        remote_full_path = (
            posixpath.join(self.path, remote_file_path) if remote_file_path else self.path
        )
        with self.get_ftp_client() as ftp:
            with open(local_path, "wb") as f:
                ftp.retrbinary("RETR " + remote_full_path, f.write)

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException("Not implemented yet")
