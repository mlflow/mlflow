import os
from abc import abstractmethod, ABCMeta

from mlflow.store.rest_store import RestStore
from mlflow.exceptions import MlflowException
from mlflow.utils.file_utils import build_path, TempDir


class ArtifactRepository:
    """
    Defines how to upload (log) and download potentially large artifacts from different
    storage backends.
    """

    __metaclass__ = ABCMeta

    def __init__(self, artifact_uri):
        self.artifact_uri = artifact_uri

    @abstractmethod
    def log_artifact(self, local_file, artifact_path=None):
        """
        Logs a local file as an artifact, optionally taking an ``artifact_path`` to place it in
        within the run's artifacts. Run artifacts can be organized into directories, so you can
        place the artifact in a directory this way.
        :param local_file: Path to artifact to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifact
        """
        pass

    @abstractmethod
    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Logs the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.
        :param local_dir: Directory of local artifacts to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifacts
        """
        pass

    @abstractmethod
    def list_artifacts(self, path):
        """
        Return all the artifacts for this run_uuid directly under path. If path is a file, returns
        an empty list. Will error if path is neither a file nor directory.

        :param path: Relative source path that contain desired artifacts
        :return: List of artifacts as FileInfo listed directly under path.
        """
        pass

    def download_artifacts(self, artifact_path):
        """
        Download an artifact file or directory to a local directory if applicable, and return a
        local path for it.
        The caller is responsible for managing the lifecycle of the downloaded artifacts.
        :param path: Relative source path to the desired artifact
        :return: Full path desired artifact.
        """
        # TODO: Probably need to add a more efficient method to stream just a single artifact
        # without downloading it, or to get a pre-signed URL for cloud storage.

        def download_artifacts_into(artifact_path, dest_dir):
            basename = os.path.basename(artifact_path)
            local_path = build_path(dest_dir, basename)
            listing = self.list_artifacts(artifact_path)
            if len(listing) > 0:
                # Artifact_path is a directory, so make a directory for it and download everything
                if not os.path.exists(local_path):
                    os.mkdir(local_path)
                for file_info in listing:
                    download_artifacts_into(artifact_path=file_info.path, dest_dir=local_path)
            else:
                self._download_file(remote_file_path=artifact_path, local_path=local_path)
            return local_path

        with TempDir(remove_on_exit=False) as tmp:
            return download_artifacts_into(artifact_path, tmp.path())

    @abstractmethod
    def _download_file(self, remote_file_path, local_path):
        """
        Downloads the file at the specified relative remote path and saves
        it at the specified local path.

        :param remote_file_path: Source path to the remote file, relative to the root
                                 directory of the artifact repository.
        :param local_path: The path to which to save the downloaded file.
        """
        pass

    @staticmethod
    def from_artifact_uri(artifact_uri, store):
        """
        Given an artifact URI for an Experiment Run (e.g., /local/file/path or s3://my/bucket),
        returns an ArtifactReposistory instance capable of logging and downloading artifacts
        on behalf of this URI.
        :param store: An instance of AbstractStore which the artifacts are registered in.
        """
        if artifact_uri.startswith("s3:/"):
            # Import these locally to avoid creating a circular import loop
            from mlflow.store.s3_artifact_repo import S3ArtifactRepository
            return S3ArtifactRepository(artifact_uri)
        elif artifact_uri.startswith("gs:/"):
            from mlflow.store.gcs_artifact_repo import GCSArtifactRepository
            return GCSArtifactRepository(artifact_uri)
        elif artifact_uri.startswith("wasbs:/"):
            from mlflow.store.azure_blob_artifact_repo import AzureBlobArtifactRepository
            return AzureBlobArtifactRepository(artifact_uri)
        elif artifact_uri.startswith("sftp:/"):
            from mlflow.store.sftp_artifact_repo import SFTPArtifactRepository
            return SFTPArtifactRepository(artifact_uri)
        elif artifact_uri.startswith("dbfs:/"):
            from mlflow.store.dbfs_artifact_repo import DbfsArtifactRepository
            if not isinstance(store, RestStore):
                raise MlflowException('`store` must be an instance of RestStore.')
            return DbfsArtifactRepository(artifact_uri, store.get_host_creds)
        else:
            from mlflow.store.local_artifact_repo import LocalArtifactRepository
            return LocalArtifactRepository(artifact_uri)
