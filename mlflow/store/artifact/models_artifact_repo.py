import logging
import os
import urllib.parse
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.databricks_models_artifact_repo import DatabricksModelsArtifactRepository
from mlflow.store.artifact.unity_catalog_models_artifact_repo import (
    UnityCatalogModelsArtifactRepository,
)
from mlflow.store.artifact.unity_catalog_oss_models_artifact_repo import (
    UnityCatalogOSSModelsArtifactRepository,
)
from mlflow.store.artifact.utils.models import (
    _parse_model_uri,
    get_model_name_and_version,
    is_using_databricks_registry,
)
from mlflow.utils.uri import (
    add_databricks_profile_info_to_artifact_uri,
    get_databricks_profile_uri_from_artifact_uri,
    is_databricks_unity_catalog_uri,
    is_models_uri,
    is_oss_unity_catalog_uri,
)

REGISTERED_MODEL_META_FILE_NAME = "registered_model_meta"

_logger = logging.getLogger(__name__)


class ModelsArtifactRepository(ArtifactRepository):
    """
    Handles artifacts associated with a model version in the model registry via URIs of the form:
      - `models:/<model_name>/<model_version>`
      - `models:/<model_name>/<stage>`  (refers to the latest model version in the given stage)
      - `models:/<model_name>/latest` (refers to the latest of all model versions)
    It is a light wrapper that resolves the artifact path to an absolute URI then instantiates
    and uses the artifact repository for that URI.
    """

    def __init__(self, artifact_uri: str, tracking_uri: str | None = None) -> None:
        from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

        super().__init__(artifact_uri, tracking_uri)
        registry_uri = mlflow.get_registry_uri()
        self.is_logged_model_uri = self._is_logged_model_uri(artifact_uri)
        if is_databricks_unity_catalog_uri(uri=registry_uri) and not self.is_logged_model_uri:
            self.repo = UnityCatalogModelsArtifactRepository(
                artifact_uri=artifact_uri, registry_uri=registry_uri
            )
            self.model_name = self.repo.model_name
            self.model_version = self.repo.model_version
        elif is_oss_unity_catalog_uri(uri=registry_uri) and not self.is_logged_model_uri:
            self.repo = UnityCatalogOSSModelsArtifactRepository(
                artifact_uri=artifact_uri, registry_uri=registry_uri
            )
            self.model_name = self.repo.model_name
            self.model_version = self.repo.model_version
        elif is_using_databricks_registry(artifact_uri) and not self.is_logged_model_uri:
            # Use the DatabricksModelsArtifactRepository if a databricks profile is being used.
            self.repo = DatabricksModelsArtifactRepository(artifact_uri)
            self.model_name = self.repo.model_name
            self.model_version = self.repo.model_version
        else:
            (
                self.model_name,
                self.model_version,
                underlying_uri,
            ) = ModelsArtifactRepository._get_model_uri_infos(artifact_uri)
            self.repo = get_artifact_repository(underlying_uri)
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
        Split 'models://<scope>:<prefix>@databricks/<name>/<version>/path/to/model' into
        ('models://<scope>:<prefix>@databricks/<name>/<version>', 'path/to/model').
        Split 'models:/<name>@alias/path/to/model' into
        ('models:/<name>@alias', 'path/to/model').
        """
        uri = uri.rstrip("/")
        parsed_url = urllib.parse.urlparse(uri)
        path = parsed_url.path
        netloc = parsed_url.netloc
        if path.count("/") >= 2 and not path.endswith("/"):
            splits = path.split("/", 3)
            cut_index = 2 if "@" in splits[1] else 3
            model_name_and_version = splits[:cut_index]
            artifact_path = "/".join(splits[cut_index:])
            base_part = f"models://{netloc}" if netloc else "models:"
            return base_part + "/".join(model_name_and_version), artifact_path
        return uri, ""

    @staticmethod
    def _is_logged_model_uri(uri: str | Path) -> bool:
        """
        Returns True if the URI is a logged model URI (e.g. 'models:/<model_id>'), False otherwise.
        """
        uri = str(uri)
        return is_models_uri(uri) and _parse_model_uri(uri).model_id is not None

    @staticmethod
    def _get_model_uri_infos(uri):
        # Note: to support a registry URI that is different from the tracking URI here,
        # we'll need to add setting of registry URIs via environment variables.

        from mlflow import MlflowClient

        databricks_profile_uri = (
            get_databricks_profile_uri_from_artifact_uri(uri) or mlflow.get_registry_uri()
        )
        client = MlflowClient(registry_uri=databricks_profile_uri)
        name_and_version_or_id = get_model_name_and_version(client, uri)
        if len(name_and_version_or_id) == 1:
            name = None
            version = None
            model_id = name_and_version_or_id[0]
            download_uri = client.get_logged_model(model_id).artifact_location
        else:
            name, version = name_and_version_or_id
            download_uri = client.get_model_version_download_uri(name, version)

        return (
            name,
            version,
            add_databricks_profile_info_to_artifact_uri(download_uri, databricks_profile_uri),
        )

    @staticmethod
    def get_underlying_uri(uri):
        _, _, underlying_uri = ModelsArtifactRepository._get_model_uri_infos(uri)

        return underlying_uri

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
        if self.is_logged_model_uri:
            return self.repo.log_artifact(local_file, artifact_path)
        raise ValueError(
            "log_artifact is not supported for models:/<name>/<version> URIs. "
            "Use register_model instead."
        )

    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.

        Args:
            local_dir: Directory of local artifacts to log.
            artifact_path: Directory within the run's artifact directory in which to log the
                artifacts.
        """
        if self.is_logged_model_uri:
            return self.repo.log_artifacts(local_dir, artifact_path)
        raise ValueError(
            "log_artifacts is not supported for models:/<name>/<version> URIs. "
            "Use register_model instead."
        )

    def list_artifacts(self, path):
        """
        Return all the artifacts for this run_id directly under path. If path is a file, returns
        an empty list. Will error if path is neither a file nor directory.

        Args:
            path: Relative source path that contain desired artifacts.

        Returns:
            List of artifacts as FileInfo listed directly under path.
        """
        return self.repo.list_artifacts(path)

    def _add_registered_model_meta_file(self, model_path):
        from mlflow.utils.yaml_utils import write_yaml

        write_yaml(
            model_path,
            REGISTERED_MODEL_META_FILE_NAME,
            {
                "model_name": self.model_name,
                "model_version": self.model_version,
            },
            overwrite=True,
            ensure_yaml_extension=False,
        )

    def download_artifacts(self, artifact_path, dst_path=None, lineage_header_info=None):
        """
        Download an artifact file or directory to a local directory if applicable, and return a
        local path for it.
        For registered models, when the artifact is downloaded, the model name and version
        are saved in the "registered_model_meta" file on the caller's side.
        The caller is responsible for managing the lifecycle of the downloaded artifacts.

        Args:
            artifact_path: Relative source path to the desired artifacts.
            dst_path: Absolute path of the local filesystem destination directory to which to
                download the specified artifacts. This directory must already exist.
                If unspecified, the artifacts will either be downloaded to a new
                uniquely-named directory on the local filesystem or will be returned
                directly in the case of the LocalArtifactRepository.
            lineage_header_info: Linear header information.

        Returns:
            Absolute path of the local filesystem location containing the desired artifacts.
        """

        from mlflow.models.model import MLMODEL_FILE_NAME

        # Pass lineage header info if model is registered in UC
        if isinstance(self.repo, UnityCatalogModelsArtifactRepository):
            model_path = self.repo.download_artifacts(
                artifact_path, dst_path, lineage_header_info=lineage_header_info
            )
        else:
            model_path = self.repo.download_artifacts(artifact_path, dst_path)
        # NB: only add the registered model metadata iff the artifact path is at the root model
        # directory. For individual files or subdirectories within the model directory, do not
        # create the metadata file.
        if os.path.isdir(model_path) and MLMODEL_FILE_NAME in os.listdir(model_path):
            self._add_registered_model_meta_file(model_path)

        return model_path

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
        raise MlflowException("Not implemented yet")
