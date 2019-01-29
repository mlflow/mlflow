import os

from mlflow.entities import FileInfo
from mlflow.store.artifact_repo import ArtifactRepository
from six.moves import urllib


class SFTPArtifactRepository(ArtifactRepository):
    """Stores artifacts as files in a remote directory, via sftp."""

    def __init__(self, artifact_uri, client=None):
        self.uri = artifact_uri
        parsed = urllib.parse.urlparse(artifact_uri)
        self.config = {
            'host': parsed.hostname,
            'port': 22 if parsed.port is None else parsed.port,
            'username': parsed.username,
            'password': parsed.password
        }
        self.path = parsed.path

        if client:
            self.sftp = client
        else:
            import pysftp
            import paramiko

            if self.config['host'] is None:
                self.config['host'] = 'localhost'

            ssh_config = paramiko.SSHConfig()
            user_config_file = os.path.expanduser("~/.ssh/config")
            if os.path.exists(user_config_file):
                with open(user_config_file) as f:
                    ssh_config.parse(f)

            user_config = ssh_config.lookup(self.config['host'])

            if 'hostname' in user_config:
                self.config['host'] = user_config['hostname']

            if self.config['username'] is None and 'username' in user_config:
                self.config['username'] = user_config['username']

            if 'identityfile' in user_config:
                self.config['private_key'] = user_config['identityfile'][0]

            self.sftp = pysftp.Connection(**self.config)

        super(SFTPArtifactRepository, self).__init__(artifact_uri)

    def get_path_module(self):
        return os.path

    def log_artifact(self, local_file, artifact_path=None):
        artifact_dir = self.get_path_module().join(self.path, artifact_path) \
            if artifact_path else self.path
        self.sftp.makedirs(artifact_dir)
        self.sftp.put(local_file,
                      self.get_path_module().join(
                          artifact_dir, self.get_path_module().basename(local_file)))

    def log_artifacts(self, local_dir, artifact_path=None):
        artifact_dir = self.get_path_module().join(self.path, artifact_path) \
            if artifact_path else self.path
        self.sftp.makedirs(artifact_dir)
        self.sftp.put_r(local_dir, artifact_dir)

    def list_artifacts(self, path=None):
        artifact_dir = self.path
        list_dir = self.get_path_module().join(artifact_dir, path) if path else artifact_dir
        if not self.sftp.isdir(list_dir):
            return []
        artifact_files = self.sftp.listdir(list_dir)
        infos = []
        for file_name in artifact_files:
            file_path = file_name if path is None else self.get_path_module().join(path, file_name)
            full_file_path = self.get_path_module().join(list_dir, file_name)
            if self.sftp.isdir(full_file_path):
                infos.append(FileInfo(file_path, True, None))
            else:
                infos.append(FileInfo(file_path, False, self.sftp.stat(full_file_path).st_size))
        return infos

    def _download_file(self, remote_file_path, local_path):
        remote_full_path = self.get_path_module().join(self.path, remote_file_path)
        self.sftp.get(remote_full_path, local_path)
