import urllib.parse

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.uri import (
    add_databricks_profile_info_to_artifact_uri,
    get_databricks_profile_uri_from_artifact_uri,
)


class RunsArtifactRepository(ArtifactRepository):
    """
    Handles artifacts associated with a Run via URIs of the form
      `runs:/<run_id>/run-relative/path/to/artifact`.
    It is a light wrapper that resolves the artifact path to an absolute URI then instantiates
    and uses the artifact repository for that URI.

    The relative path part of ``artifact_uri`` is expected to be in posixpath format, so Windows
    users should take special care when constructing the URI.
    """

    def __init__(self, artifact_uri):
        from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

        super().__init__(artifact_uri)
        uri = RunsArtifactRepository.get_underlying_uri(artifact_uri)
        self.repo = get_artifact_repository(uri)

    @staticmethod
    def is_runs_uri(uri):
        return urllib.parse.urlparse(uri).scheme == "runs"

    @staticmethod
    def get_underlying_uri(runs_uri):
        from mlflow.tracking.artifact_utils import get_artifact_uri

        (run_id, artifact_path) = RunsArtifactRepository.parse_runs_uri(runs_uri)
        tracking_uri = get_databricks_profile_uri_from_artifact_uri(runs_uri)
        uri = get_artifact_uri(run_id, artifact_path, tracking_uri)
        assert not RunsArtifactRepository.is_runs_uri(uri)  # avoid an infinite loop
        return add_databricks_profile_info_to_artifact_uri(uri, tracking_uri)

    @staticmethod
    def parse_runs_uri(run_uri):
        parsed = urllib.parse.urlparse(run_uri)
        if parsed.scheme != "runs":
            raise MlflowException(
                "Not a proper runs:/ URI: %s. " % run_uri
                + "Runs URIs must be of the form 'runs:/<run_id>/run-relative/path/to/artifact'"
            )

        path = parsed.path
        if not path.startswith("/") or len(path) <= 1:
            raise MlflowException(
                "Not a proper runs:/ URI: %s. " % run_uri
                + "Runs URIs must be of the form 'runs:/<run_id>/run-relative/path/to/artifact'"
            )
        path = path[1:]

        path_parts = path.split("/")
        run_id = path_parts[0]
        if run_id == "":
            raise MlflowException(
                "Not a proper runs:/ URI: %s. " % run_uri
                + "Runs URIs must be of the form 'runs:/<run_id>/run-relative/path/to/artifact'"
            )

        artifact_path = "/".join(path_parts[1:]) if len(path_parts) > 1 else None
        artifact_path = artifact_path if artifact_path != "" else None

        return run_id, artifact_path

    def log_artifact(self, local_file, artifact_path=None):
        """
        Log a local file as an artifact, optionally taking an ``artifact_path`` to place it in
        within the run's artifacts. Run artifacts can be organized into directories, so you can
        place the artifact in a directory this way.

        :param local_file: Path to artifact to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifact
        """
        self.repo.log_artifact(local_file, artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.

        :param local_dir: Directory of local artifacts to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifacts
        """
        self.repo.log_artifacts(local_dir, artifact_path)

    def _is_directory(self, artifact_path):
        return self.repo._is_directory(artifact_path)

    def list_artifacts(self, path):
        """
        Return all the artifacts for this run_id directly under path. If path is a file, returns
        an empty list. Will error if path is neither a file nor directory.

        :param path: Relative source path that contain desired artifacts

        :return: List of artifacts as FileInfo listed directly under path.
        """
        return self.repo.list_artifacts(path)

    def download_artifacts(self, artifact_path, dst_path=None):
        """
        Download an artifact file or directory to a local directory if applicable, and return a
        local path for it.
        The caller is responsible for managing the lifecycle of the downloaded artifacts.

        :param artifact_path: Relative source path to the desired artifacts.
        :param dst_path: Absolute path of the local filesystem destination directory to which to
                         download the specified artifacts. This directory must already exist.
                         If unspecified, the artifacts will either be downloaded to a new
                         uniquely-named directory on the local filesystem or will be returned
                         directly in the case of the LocalArtifactRepository.

        :return: Absolute path of the local filesystem location containing the desired artifacts.
        """
        return self.repo.download_artifacts(artifact_path, dst_path)

    def _download_file(self, remote_file_path, local_path):
        """
        Download the file at the specified relative remote path and saves
        it at the specified local path.

        :param remote_file_path: Source path to the remote file, relative to the root
                                 directory of the artifact repository.
        :param local_path: The path to which to save the downloaded file.
        """
        self.repo._download_file(remote_file_path, local_path)

    def delete_artifacts(self, artifact_path=None):
        self.repo.delete_artifacts(artifact_path)
