"""
The ``mlflow.store`` module provides an API for saving artifacts into different *artifact store* to
persist run artifacts like models, plots, images, etc.

Currently there are implementations available:

- :py:mod:`mlflow.store.artifact_repo.ArtifactRepository`
- :py:mod:`mlflow.store.artifact_repository_registry.ArtifactRepositoryRegistry`
- :py:mod:`mlflow.store.azure_blob_artifact_repo.AzureBlobArtifactRepository`
- :py:mod:`mlflow.store.dbfs_artifact_repo.DbfsArtifactRepository`
- :py:mod:`mlflow.store.ftp_artifact_repo.FTPArtifactRepository`
- :py:mod:`mlflow.store.gcs_artifact_repo.GCSArtifactRepository`
- :py:mod:`mlflow.store.hdfs_artifact_repo.HdfsArtifactRepository`
- :py:mod:`mlflow.store.local_artifact_repo.LocalArtifactRepository`
- :py:mod:`mlflow.store.s3_artifact_repo.S3ArtifactRepository`
- :py:mod:`mlflow.store.sftp_artifact_repo.SFTPArtifactRepository`

"""

# Path to default location for backend when using local FileStore or ArtifactStore.
# Also used as default location for artifacts, when not provided, in non local file based backends
# (eg MySQL)
DEFAULT_LOCAL_FILE_AND_ARTIFACT_PATH = "./mlruns"
