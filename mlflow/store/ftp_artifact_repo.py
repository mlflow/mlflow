import os
import ftplib
from ftplib import FTP
from contextlib import contextmanager

from six.moves import urllib

from mlflow.entities.file_info import FileInfo
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import TempDir, build_path, get_relative_path


class FTPArtifactRepository(ArtifactRepository):
    """Stores artifacts as files in a remote directory, via ftp."""

    def __init__(self, artifact_uri):
        self.uri = artifact_uri
        parsed = urllib.parse.urlparse(artifact_uri)
        self.config = {
            'host': parsed.hostname,
            'port': 21 if parsed.port is None else parsed.port,
            'username': parsed.username,
            'password': parsed.password
        }
        self.path = parsed.path

        if self.config['host'] is None:
            self.config['host'] = 'localhost'

        super(FTPArtifactRepository, self).__init__(artifact_uri)

    @contextmanager
    def get_ftp_client(self):
        ftp = FTP()
        ftp.connect(self.config['host'], self.config['port'])
        ftp.login(self.config['username'], self.config['password'])
        yield ftp
        ftp.close()

    def _is_dir(self, full_file_path):
        with self.get_ftp_client() as ftp:
            try:
                ftp.cwd(full_file_path)
                result = True
            except ftplib.error_perm:
                result = False
        return result

    def _mkdir(self, artifact_dir):
        with self.get_ftp_client() as ftp:
            try:
                ftp.mkd(artifact_dir)
            except ftplib.error_perm:
                head, _ = os.path.split(artifact_dir)
                self._mkdir(head)
                self._mkdir(artifact_dir)

    def _size(self, full_file_path):
        with self.get_ftp_client() as ftp:
            ftp.voidcmd('TYPE I')
            size = ftp.size(full_file_path)
            ftp.voidcmd('TYPE A')
        return size

    def download_files(self, path, destination):
        with self.get_ftp_client() as ftp:
            ftp.cwd(path)
            if not os.path.isdir(destination):
                os.makedirs(destination)

            filelist = ftp.nlst()

            for ftp_file in filelist:
                if self._is_dir(build_path(path, ftp_file)):
                    self.download_files(build_path(path, ftp_file),
                                        build_path(destination, ftp_file))
                else:
                    with open(os.path.join(destination, ftp_file), "wb") as f:
                        ftp.retrbinary("RETR "+ftp_file, f)

    def log_artifact(self, local_file, artifact_path=None):
        with self.get_ftp_client() as ftp:
            artifact_dir = os.path.join(self.path, artifact_path) \
                if artifact_path else self.path
            self._mkdir(artifact_dir)
            with open(local_file, 'rb') as f:
                ftp.cwd(artifact_dir)
                ftp.storbinary('STOR ' + os.path.basename(local_file), f)

    def log_artifacts(self, local_dir, artifact_path=None):
        dest_path = os.path.join(self.path, artifact_path) \
            if artifact_path else self.path

        dest_path = build_path(dest_path, os.path.split(local_dir)[1])
        dest_path_re = os.path.split(local_dir)[1]
        if artifact_path:
            dest_path_re = build_path(artifact_path, os.path.split(local_dir)[1])

        local_dir = os.path.abspath(local_dir)
        for (root, _, filenames) in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = get_relative_path(local_dir, root)
                upload_path = build_path(dest_path_re, rel_path)
            if not filenames:
                self._mkdir(build_path(self.path, upload_path))
            for f in filenames:
                if os.path.isfile(build_path(root, f)):
                    self.log_artifact(build_path(root, f), upload_path)

    def list_artifacts(self, path=None):
        with self.get_ftp_client() as ftp:
            artifact_dir = self.path
            list_dir = os.path.join(artifact_dir, path) if path else artifact_dir
            artifact_files = ftp.nlst(list_dir)
            infos = []
            for file_name in artifact_files:
                file_path = file_name if path is None else os.path.join(path, file_name)
                full_file_path = os.path.join(list_dir, file_name)
                if self._is_dir(full_file_path):
                    infos.append(FileInfo(file_path, True, None))
                else:
                    size = self._size(full_file_path)
                    infos.append(FileInfo(file_path, False, size))
        return infos

    def download_artifacts(self, artifact_path=None):
        with self.get_ftp_client() as ftp:
            full_path = os.path.join(self.path, artifact_path) \
                if artifact_path else self.path
            return_path = None
            with TempDir(remove_on_exit=False) as tmp:
                tmp_path = tmp.path()
                if self._is_dir(full_path):
                    self.download_files(full_path, tmp_path)
                    return_path = tmp_path
                else:
                    local_file = os.path.join(tmp_path, os.path.basename(full_path))
                    with open(local_file, 'wb') as f:
                        ftp.retrbinary('RETR ' + full_path, f)
                    return_path = local_file
        return return_path

    def _download_file(self, remote_file_path, local_path):
        remote_full_path = os.path.join(self.path, remote_file_path) \
                if remote_file_path else self.path
        with self.get_ftp_client() as ftp:
            local_file = os.path.join(local_path, os.path.basename(remote_full_path))
            with open(local_file, 'wb') as f:
                ftp.retrbinary('RETR ' + remote_full_path, f)
