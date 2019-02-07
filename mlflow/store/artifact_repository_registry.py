import entrypoints
from six.moves import urllib

from mlflow.exceptions import MlflowException
from mlflow.store.gcs_artifact_repo import GCSArtifactRepository
from mlflow.store.azure_blob_artifact_repo import AzureBlobArtifactRepository
from mlflow.store.ftp_artifact_repo import FTPArtifactRepository
from mlflow.store.sftp_artifact_repo import SFTPArtifactRepository
from mlflow.store.dbfs_artifact_repo import DbfsArtifactRepository
from mlflow.store.s3_artifact_repo import S3ArtifactRepository
from mlflow.store.local_artifact_repo import LocalArtifactRepository
from mlflow.store.rest_store import RestStore

class ArtifactRepositoryRegistry:

    def __init__(self):
        self.registry = {}

    def register(self, scheme, repository):
        self.registry[scheme] = repository

    def register_entrypoints(self):
        # Register artifact repositories provided by other packages
        for entrypoint in entrypoints.get_group_all("mlflow.artifact_repository"):
            self.register(entrypoint.name, entrypoint.load())

    def get_artifact_repository(self, artifact_uri, store=None):
        scheme = urllib.parse.urlparse(artifact_uri).scheme
        repository = self.registry.get(scheme)
        if scheme == "dbfs" and repository is not None:
            if not isinstance(store, RestStore):
                raise MlflowException('`store` must be an instance of RestStore.')
            return repository(artifact_uri, store.get_host_creds)
        elif repository is not None:
            return repository(artifact_uri)
        else:
            raise Exception("Artifact URI must be....")

artifact_repository_registry = ArtifactRepositoryRegistry()
artifact_repository_registry.register('', LocalArtifactRepository)
artifact_repository_registry.register('s3', S3ArtifactRepository)
artifact_repository_registry.register('gs', GCSArtifactRepository)
artifact_repository_registry.register('wasbs', AzureBlobArtifactRepository)
artifact_repository_registry.register('ftp', FTPArtifactRepository)
artifact_repository_registry.register('sftp', SFTPArtifactRepository)
artifact_repository_registry.register('dbfs', DbfsArtifactRepository)

artifact_repository_registry.register_entrypoints()