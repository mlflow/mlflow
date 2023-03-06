import urllib.parse

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.databricks_models_artifact_repo import DatabricksModelsArtifactRepository
from mlflow.store.artifact.utils.models import (
    get_model_name_and_version,
    is_using_databricks_registry,
)
from mlflow.utils.uri import (
    add_databricks_profile_info_to_artifact_uri,
    get_databricks_profile_uri_from_artifact_uri,
    is_databricks_unity_catalog_uri,
)
from mlflow.store.artifact.unity_catalog_models_artifact_repo import (
    UnityCatalogModelsArtifactRepository,
)


class ModelsArtifactRepository(ArtifactRepository):
    """
    Handles artifacts associated with a model version in the model registry via URIs of the form:
      - `models:/<model_name>/<model_version>`
      - `models:/<model_name>/<stage>`  (refers to the latest model version in the given stage)
      - `models:/<model_name>/latest` (refers to the latest of all model versions)
    It is a light wrapper that resolves the artifact path to an absolute URI then instantiates
    and uses the artifact repository for that URI.
    """

    def __init__(self, artifact_uri):
        from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

        super().__init__(artifact_uri)
        registry_uri = mlflow.get_registry_uri()
        if is_databricks_unity_catalog_uri(uri=registry_uri):
            self.repo = UnityCatalogModelsArtifactRepository(
                artifact_uri=artifact_uri, registry_uri=registry_uri
            )
        elif is_using_databricks_registry(artifact_uri):
            # Use the DatabricksModelsArtifactRepository if a databricks profile is being used.
            self.repo = DatabricksModelsArtifactRepository(artifact_uri)
        else:
            uri = ModelsArtifactRepository.get_underlying_uri(artifact_uri)
            self.repo = get_artifact_repository(uri)
            # TODO: it may be nice to fall back to the source URI explicitly here if for some reason
            #  we don't get a download URI here, or fail during the download itself.

    @staticmethod
    def is_models_uri(uri):
        return urllib.parse.urlparse(uri).scheme == "models"

    @staticmethod
    def split_models_uri(uri):
        """
        Split 'models:/<name>/<version>/path/to/model' into
        ('models:/<name>/<version>', 'path/to/model').
        """
        path = urllib.parse.urlparse(uri).path
        if path.count("/") >= 3 and not path.endswith("/"):
            splits = path.split("/", 3)
            model_name_and_version = splits[:3]
            artifact_path = splits[-1]
            return "models:" + "/".join(model_name_and_version), artifact_path
        return uri, ""

    @staticmethod
    def get_underlying_uri(uri):
        # Note: to support a registry URI that is different from the tracking URI here,
        # we'll need to add setting of registry URIs via environment variables.

        from mlflow import MlflowClient

        databricks_profile_uri = (
            get_databricks_profile_uri_from_artifact_uri(uri) or mlflow.get_registry_uri()
        )
        client = MlflowClient(registry_uri=databricks_profile_uri)
        (name, version) = get_model_name_and_version(client, uri)
        download_uri = client.get_model_version_download_uri(name, version)
        return add_databricks_profile_info_to_artifact_uri(download_uri, databricks_profile_uri)

    def log_artifact(self, local_file, artifact_path=None):
        """
        Log a local file as an artifact, optionally taking an ``artifact_path`` to place it in
        within the run's artifacts. Run artifacts can be organized into directories, so you can
        place the artifact in a directory this way.

        :param local_file: Path to artifact to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifact
        """
        raise ValueError(
            "log_artifact is not supported for models:/ URIs. Use register_model instead."
        )

    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.

        :param local_dir: Directory of local artifacts to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifacts
        """
        raise ValueError(
            "log_artifacts is not supported for models:/ URIs. Use register_model instead."
        )

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
        raise MlflowException("Not implemented yet")
