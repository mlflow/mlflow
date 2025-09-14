import mlflow.utils.databricks_utils
from mlflow.environment_variables import MLFLOW_ENABLE_UC_VOLUME_FUSE_ARTIFACT_REPO
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.databricks_sdk_artifact_repo import DatabricksSdkArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.utils.uri import (
    get_databricks_profile_uri_from_artifact_uri,
    is_databricks_model_registry_artifacts_uri,
    is_valid_uc_volumes_uri,
    remove_databricks_profile_info_from_artifact_uri,
    strip_scheme,
)


class UCVolumesArtifactRepository(DatabricksSdkArtifactRepository):
    """
    Stores artifacts on UC Volumes using the Files REST API.
    """

    def __init__(self, artifact_uri: str, tracking_uri: str | None = None) -> None:
        if not is_valid_uc_volumes_uri(artifact_uri):
            raise MlflowException(
                message=(
                    f"UC volumes URI must be of the form "
                    f"dbfs:/Volumes/<catalog>/<schema>/<volume>/<path>: {artifact_uri}"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        uri = remove_databricks_profile_info_from_artifact_uri(artifact_uri)
        super().__init__("/" + strip_scheme(uri).strip("/"), tracking_uri)


def uc_volume_artifact_repo_factory(artifact_uri: str, tracking_uri: str | None = None):
    """
    Returns an ArtifactRepository subclass for storing artifacts on Volumes.

    This factory method is used with URIs of the form ``dbfs:/Volumes/<path>``. Volume-backed
    artifact storage can only be used together with the RestStore.

    Args:
        artifact_uri: Volume root artifact URI.
        tracking_uri: The tracking URI.

    Returns:
        Subclass of ArtifactRepository capable of storing artifacts on DBFS.
    """
    if not is_valid_uc_volumes_uri(artifact_uri):
        raise MlflowException(
            message=(
                f"UC volumes URI must be of the form "
                f"dbfs:/Volumes/<catalog>/<schema>/<volume>/<path>: {artifact_uri}"
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    artifact_uri = artifact_uri.rstrip("/")
    db_profile_uri = get_databricks_profile_uri_from_artifact_uri(artifact_uri)
    if (
        mlflow.utils.databricks_utils.is_uc_volume_fuse_available()
        and MLFLOW_ENABLE_UC_VOLUME_FUSE_ARTIFACT_REPO.get()
        and not is_databricks_model_registry_artifacts_uri(artifact_uri)
        and (db_profile_uri is None or db_profile_uri == "databricks")
    ):
        # If the UC Volume FUSE mount is available, write artifacts directly to
        # /Volumes/... using local filesystem APIs.
        # Note: it is possible for a named Databricks profile to point to the current workspace,
        # but we're going to avoid doing a complex check and assume users will use `databricks`
        # to mean the current workspace. Using `UCVolumesArtifactRepository` to access
        # the current workspace's Volumes should still work; it just may be slower.
        uri_without_profile = remove_databricks_profile_info_from_artifact_uri(artifact_uri)
        path = strip_scheme(uri_without_profile).lstrip("/")
        return LocalArtifactRepository(f"file:///{path}", tracking_uri=tracking_uri)
    return UCVolumesArtifactRepository(artifact_uri, tracking_uri=tracking_uri)
