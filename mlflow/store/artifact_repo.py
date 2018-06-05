
from abc import abstractmethod, ABCMeta
import shutil

from distutils import dir_util
from mlflow.utils.file_utils import (mkdir, exists, list_all, get_relative_path, 
                                     get_file_info, build_path)


class ArtifactRepository:
    """
    Defines how to upload (log) and download potentially large artifacts from different
    storage backends.
    """

    __metaclass__ = ABCMeta

    def __init__(self, artifact_uri):
        self.artifact_uri = artifact_uri

    @abstractmethod
    def log_artifact(self, local_file, artifact_path):
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
    def log_artifacts(self, local_dir, artifact_path):
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
        Return all the artifacts for this run_uuid directly under path.
        :param path: relative source path that contain desired artifacts
        :return: List of artifacts as FileInfo listed directly under path.
        """
        pass

    @abstractmethod
    def download_artifact(self, artifact_path):
        """
        :param path: relative source path to the desired artifact
        :return: Full path desired artifact.
        """
        pass

    @staticmethod
    def from_artifact_uri(artfact_uri):
        """Given an artifact URI for an Experiment Run (e.g., /local/file/path or s3://my/bucket),
        returns an ArtifactReposistory instance capable of logging and downloading artifacts
        on behalf of this URI.
        """
        return LocalFileRepository(artfact_uri)


class LocalFileRepository(ArtifactRepository):
    """Stores files in a local directory."""

    def log_artifact(self, local_file, artifact_path):
        artifact_dir = build_path(self.artifact_uri, artifact_path) \
            if artifact_path else self.artifact_uri
        if not exists(artifact_dir):
            mkdir(artifact_dir)
        shutil.copy(local_file, artifact_dir)

    def log_artifacts(self, local_dir, artifact_path):
        artifact_dir = build_path(self.artifact_uri, artifact_path) \
            if artifact_path else self.artifact_uri
        if not exists(artifact_dir):
            mkdir(artifact_dir)
        dir_util.copy_tree(src=local_dir, dst=artifact_dir)

    def list_artifacts(self, path=None):
        artifact_dir = self.artifact_uri
        list_dir = build_path(artifact_dir, path) if path else artifact_dir
        artifact_files = list_all(list_dir, full_path=True)
        return [get_file_info(f, get_relative_path(artifact_dir, f)) for f in artifact_files]

    def download_artifact(self, artifact_path):
        """Since this is a local file store, we do not need to download the artifact."""
        return build_path(self.artifact_uri, artifact_path)
