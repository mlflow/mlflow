import warnings

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.azure_blob_artifact_repo import AzureBlobArtifactRepository
from mlflow.store.artifact.azure_data_lake_artifact_repo import AzureDataLakeArtifactRepository
from mlflow.store.artifact.dbfs_artifact_repo import dbfs_artifact_repo_factory
from mlflow.store.artifact.ftp_artifact_repo import FTPArtifactRepository
from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
from mlflow.store.artifact.hdfs_artifact_repo import HdfsArtifactRepository
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.artifact.mlflow_artifacts_repo import MlflowArtifactsRepository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.r2_artifact_repo import R2ArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.store.artifact.sftp_artifact_repo import SFTPArtifactRepository
from mlflow.store.artifact.uc_volume_artifact_repo import uc_volume_artifact_repo_factory
from mlflow.utils.plugins import get_entry_points
from mlflow.utils.uri import get_uri_scheme, is_uc_volumes_uri


class ArtifactRepositoryRegistry:
    """Scheme-based registry for artifact repository implementations

    This class allows the registration of a function or class to provide an implementation for a
    given scheme of `artifact_uri` through the `register` method. Implementations declared though
    the entrypoints `mlflow.artifact_repository` group can be automatically registered through the
    `register_entrypoints` method.

    When instantiating an artifact repository through the `get_artifact_repository` method, the
    scheme of the artifact URI provided will be used to select which implementation to instantiate,
    which will be called with same arguments passed to the `get_artifact_repository` method.
    """

    def __init__(self):
        self._registry = {}

    def register(self, scheme, repository):
        """Register artifact repositories provided by other packages"""
        self._registry[scheme] = repository

    def register_entrypoints(self):
        # Register artifact repositories provided by other packages
        for entrypoint in get_entry_points("mlflow.artifact_repository"):
            try:
                self.register(entrypoint.name, entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    'Failure attempting to register artifact repository for scheme "{}": {}'.format(
                        entrypoint.name, str(exc)
                    ),
                    stacklevel=2,
                )

    def get_artifact_repository(
        self, artifact_uri: str, tracking_uri: str | None = None
    ) -> ArtifactRepository:
        """
        Get an artifact repository from the registry based on the scheme of artifact_uri

        Args:
            artifact_uri: The artifact store URI. This URI is used to select which artifact
                repository implementation to instantiate and is passed to the constructor of the
                implementation.
            tracking_uri: The tracking URI. This URI is passed to the constructor of the
                implementation.

        Returns:
            An instance of `mlflow.store.ArtifactRepository` that fulfills the artifact URI
            requirements.
        """
        scheme = get_uri_scheme(artifact_uri)
        repository = self._registry.get(scheme)
        if repository is None:
            raise MlflowException(
                f"Could not find a registered artifact repository for: {artifact_uri}. "
                f"Currently registered schemes are: {list(self._registry.keys())}"
            )
        return repository(artifact_uri, tracking_uri=tracking_uri)

    def get_registered_artifact_repositories(self):
        """
        Get all registered artifact repositories.

        Returns:
            A dictionary mapping string artifact URI schemes to artifact repositories.
        """
        return self._registry


def _dbfs_artifact_repo_factory(
    artifact_uri: str, tracking_uri: str | None = None
) -> ArtifactRepository:
    return (
        uc_volume_artifact_repo_factory(artifact_uri, tracking_uri)
        if is_uc_volumes_uri(artifact_uri)
        else dbfs_artifact_repo_factory(artifact_uri, tracking_uri)
    )


_artifact_repository_registry = ArtifactRepositoryRegistry()

_artifact_repository_registry.register("", LocalArtifactRepository)
_artifact_repository_registry.register("file", LocalArtifactRepository)
_artifact_repository_registry.register("s3", S3ArtifactRepository)
_artifact_repository_registry.register("r2", R2ArtifactRepository)
_artifact_repository_registry.register("gs", GCSArtifactRepository)
_artifact_repository_registry.register("wasbs", AzureBlobArtifactRepository)
_artifact_repository_registry.register("ftp", FTPArtifactRepository)
_artifact_repository_registry.register("sftp", SFTPArtifactRepository)
_artifact_repository_registry.register("dbfs", _dbfs_artifact_repo_factory)
_artifact_repository_registry.register("hdfs", HdfsArtifactRepository)
_artifact_repository_registry.register("viewfs", HdfsArtifactRepository)
_artifact_repository_registry.register("runs", RunsArtifactRepository)
_artifact_repository_registry.register("models", ModelsArtifactRepository)
for scheme in ["http", "https"]:
    _artifact_repository_registry.register(scheme, HttpArtifactRepository)
_artifact_repository_registry.register("mlflow-artifacts", MlflowArtifactsRepository)
_artifact_repository_registry.register("abfss", AzureDataLakeArtifactRepository)

_artifact_repository_registry.register_entrypoints()


def get_artifact_repository(
    artifact_uri: str, tracking_uri: str | None = None
) -> ArtifactRepository:
    """
    Get an artifact repository from the registry based on the scheme of artifact_uri

    Args:
        artifact_uri: The artifact store URI. This URI is used to select which artifact
            repository implementation to instantiate and is passed to the constructor of the
            implementation.
        tracking_uri: The tracking URI. This URI is passed to the constructor of the
            implementation.

    Returns:
        An instance of `mlflow.store.ArtifactRepository` that fulfills the artifact URI
        requirements.
    """
    return _artifact_repository_registry.get_artifact_repository(artifact_uri, tracking_uri)


def get_registered_artifact_repositories() -> dict[str, ArtifactRepository]:
    """
    Get all registered artifact repositories.

    Returns:
        A dictionary mapping string artifact URI schemes to artifact repositories.
    """
    return _artifact_repository_registry.get_registered_artifact_repositories()
