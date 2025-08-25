import logging
import os
import urllib.parse
from typing import Iterator

import mlflow
from mlflow.entities.file_info import FileInfo
from mlflow.entities.logged_model import LoggedModel
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import create_tmp_dir
from mlflow.utils.uri import (
    add_databricks_profile_info_to_artifact_uri,
    get_databricks_profile_uri_from_artifact_uri,
)

_logger = logging.getLogger(__name__)


class RunsArtifactRepository(ArtifactRepository):
    """
    Handles artifacts associated with a Run via URIs of the form
      `runs:/<run_id>/run-relative/path/to/artifact`.
    It is a light wrapper that resolves the artifact path to an absolute URI then instantiates
    and uses the artifact repository for that URI.

    The relative path part of ``artifact_uri`` is expected to be in posixpath format, so Windows
    users should take special care when constructing the URI.
    """

    def __init__(self, artifact_uri: str, tracking_uri: str | None = None) -> None:
        from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

        super().__init__(artifact_uri, tracking_uri)
        uri = RunsArtifactRepository.get_underlying_uri(artifact_uri, tracking_uri)
        self.repo = get_artifact_repository(uri, tracking_uri=self.tracking_uri)

    @staticmethod
    def is_runs_uri(uri):
        return urllib.parse.urlparse(uri).scheme == "runs"

    @staticmethod
    def get_underlying_uri(runs_uri: str, tracking_uri: str | None = None) -> str:
        from mlflow.tracking.artifact_utils import get_artifact_uri

        (run_id, artifact_path) = RunsArtifactRepository.parse_runs_uri(runs_uri)
        databricks_profile_uri = get_databricks_profile_uri_from_artifact_uri(runs_uri)
        uri = get_artifact_uri(
            run_id=run_id,
            artifact_path=artifact_path,
            tracking_uri=databricks_profile_uri or tracking_uri,
        )
        assert not RunsArtifactRepository.is_runs_uri(uri)  # avoid an infinite loop
        return add_databricks_profile_info_to_artifact_uri(
            artifact_uri=uri, databricks_profile_uri=databricks_profile_uri or tracking_uri
        )

    @staticmethod
    def parse_runs_uri(run_uri):
        parsed = urllib.parse.urlparse(run_uri)
        if parsed.scheme != "runs":
            raise MlflowException(
                f"Not a proper runs:/ URI: {run_uri}. "
                + "Runs URIs must be of the form 'runs:/<run_id>/run-relative/path/to/artifact'"
            )

        path = parsed.path
        if not path.startswith("/") or len(path) <= 1:
            raise MlflowException(
                f"Not a proper runs:/ URI: {run_uri}. "
                + "Runs URIs must be of the form 'runs:/<run_id>/run-relative/path/to/artifact'"
            )
        path = path[1:]

        path_parts = path.split("/")
        run_id = path_parts[0]
        if run_id == "":
            raise MlflowException(
                f"Not a proper runs:/ URI: {run_uri}. "
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

        Args:
            local_file: Path to artifact to log.
            artifact_path: Directory within the run's artifact directory in which to log the
                artifact.
        """
        self.repo.log_artifact(local_file, artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.

        Args:
            local_dir: Directory of local artifacts to log.
            artifact_path: Directory within the run's artifact directory in which to log the
                artifacts.
        """
        self.repo.log_artifacts(local_dir, artifact_path)

    def _is_directory(self, artifact_path):
        return self.repo._is_directory(artifact_path)

    def list_artifacts(self, path: str | None = None) -> list[FileInfo]:
        """
        Return all the artifacts for this run_id directly under path. If path is a file, returns
        an empty list. Will error if path is neither a file nor directory. When the run has an
        associated model, the artifacts of the model are also listed.

        Args:
            path: Relative source path that contain desired artifacts

        Returns:
            List of artifacts as FileInfo listed directly under path.
        """
        return self._list_run_artifacts(path) + self._list_model_artifacts(path)

    def _list_run_artifacts(self, path: str | None = None) -> list[FileInfo]:
        return self.repo.list_artifacts(path)

    def _get_logged_model_artifact_repo(self, run_id: str, name: str) -> ArtifactRepository | None:
        """
        Get the artifact repository for a logged model with the given name and run ID.
        Returns None if no such model exists.
        """
        from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

        client = mlflow.tracking.MlflowClient(self.tracking_uri)
        experiment_id = client.get_run(run_id).info.experiment_id

        def iter_models() -> Iterator[LoggedModel]:
            page_token: str | None = None
            while True:
                page = client.search_logged_models(
                    experiment_ids=[experiment_id],
                    # TODO: Filter by 'source_run_id' once Databricks backend supports it
                    filter_string=f"name = '{name}'",
                    page_token=page_token,
                )
                yield from page
                if not page.token:
                    break
                page_token = page.token

        if matched := next((m for m in iter_models() if m.source_run_id == run_id), None):
            return get_artifact_repository(
                matched.artifact_location, tracking_uri=self.tracking_uri
            )

        return None

    def _list_model_artifacts(self, path: str | None = None) -> list[FileInfo]:
        """
        A run can have an associated model. If so, this method lists the artifacts of the model.
        """
        full_path = f"{self.artifact_uri}/{path}" if path else self.artifact_uri
        run_id, rel_path = RunsArtifactRepository.parse_runs_uri(full_path)
        if not rel_path:
            # At least one part of the path must be present (e.g. "runs:/<run_id>/<name>")
            return []
        [model_name, *rest] = rel_path.split("/", 1)
        rel_path = rest[0] if rest else ""
        if repo := self._get_logged_model_artifact_repo(run_id=run_id, name=model_name):
            artifacts = repo.list_artifacts(path=rel_path)
            return [
                FileInfo(path=f"{model_name}/{a.path}", is_dir=a.is_dir, file_size=a.file_size)
                for a in artifacts
            ]

        return []

    def download_artifacts(self, artifact_path: str, dst_path: str | None = None) -> str:
        """
        Download an artifact file or directory to a local directory if applicable, and return a
        local path for it. When the run has an associated model, the artifacts of the model are also
        downloaded to the specified destination directory. The caller is responsible for managing
        the lifecycle of the downloaded artifacts.

        Args:
            artifact_path: Relative source path to the desired artifacts.
            dst_path: Absolute path of the local filesystem destination directory to which to
                download the specified artifacts. This directory must already exist.
                If unspecified, the artifacts will either be downloaded to a new
                uniquely-named directory on the local filesystem or will be returned
                directly in the case of the LocalArtifactRepository.

        Returns:
            Absolute path of the local filesystem location containing the desired artifacts.
        """
        dst_path = dst_path or create_tmp_dir()
        run_out_path: str | None = None
        try:
            # This fails when the run has no artifacts, so we catch the exception
            run_out_path = self.repo.download_artifacts(artifact_path, dst_path)
        except Exception:
            _logger.debug(
                f"Failed to download artifacts from {self.artifact_uri}/{artifact_path}.",
                exc_info=True,
            )

        # If there are artifacts with the same name in the run and model, the model artifacts
        # will overwrite the run artifacts.
        model_out_path: str | None = None
        try:
            model_out_path = self._download_model_artifacts(artifact_path, dst_path=dst_path)
        except Exception:
            _logger.debug(
                f"Failed to download model artifacts from {self.artifact_uri}/{artifact_path}.",
                exc_info=True,
            )
        path = run_out_path or model_out_path
        if path is None:
            raise MlflowException(
                f"Failed to download artifacts from path {artifact_path!r}, "
                "please ensure that the path is correct.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return path

    def _download_model_artifacts(self, artifact_path: str, dst_path: str) -> str | None:
        """
        A run can have an associated model. If so, this method downloads the artifacts of the model.
        """
        full_path = f"{self.artifact_uri}/{artifact_path}" if artifact_path else self.artifact_uri
        run_id, rel_path = RunsArtifactRepository.parse_runs_uri(full_path)
        if not rel_path:
            # At least one part of the path must be present (e.g. "runs:/<run_id>/<name>")
            return None
        [model_name, *rest] = rel_path.split("/", 1)
        rel_path = rest[0] if rest else ""
        if repo := self._get_logged_model_artifact_repo(run_id=run_id, name=model_name):
            dst = os.path.join(dst_path, model_name)
            os.makedirs(dst, exist_ok=True)
            return repo.download_artifacts(artifact_path=rel_path, dst_path=dst)

        return None

    def _download_file(self, remote_file_path, local_path):
        """
        Download the file at the specified relative remote path and saves
        it at the specified local path.

        Args:
            remote_file_path: Source path to the remote file, relative to the root
                directory of the artifact repository.
            local_path: The path to which to save the downloaded file.

        """
        self.repo._download_file(remote_file_path, local_path)

    def delete_artifacts(self, artifact_path=None):
        self.repo.delete_artifacts(artifact_path)
