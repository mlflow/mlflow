"""
Utilities for dealing with artifacts in the context of a Run.
"""
import os
import pathlib
import posixpath
import tempfile
import urllib.parse
import uuid

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.dbfs_artifact_repo import DbfsRestArtifactRepository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.os import is_windows
from mlflow.utils.uri import add_databricks_profile_info_to_artifact_uri, append_to_uri_path


def get_artifact_uri(run_id, artifact_path=None, tracking_uri=None):
    """Get the absolute URI of the specified artifact in the specified run. If `path` is not
    specified the artifact root URI of the specified run will be returned; calls to ``log_artifact``
    and ``log_artifacts`` write artifact(s) to subdirectories of the artifact root URI.

    Args:
        run_id: The ID of the run for which to obtain an absolute artifact URI.
        artifact_path: The run-relative artifact path. For example,
            ``path/to/artifact``. If unspecified, the artifact root URI for the
            specified run will be returned.
        tracking_uri: The tracking URI from which to get the run and its artifact location. If
            not given, the current default tracking URI is used.

    Returns:
        An *absolute* URI referring to the specified artifact or the specified run's artifact
        root. For example, if an artifact path is provided and the specified run uses an
        S3-backed  store, this may be a uri of the form
        ``s3://<bucket_name>/path/to/artifact/root/path/to/artifact``. If an artifact path
        is not provided and the specified run uses an S3-backed store, this may be a URI of
        the form ``s3://<bucket_name>/path/to/artifact/root``.

    """
    if not run_id:
        raise MlflowException(
            message="A run_id must be specified in order to obtain an artifact uri!",
            error_code=INVALID_PARAMETER_VALUE,
        )

    store = _get_store(tracking_uri)
    run = store.get_run(run_id)
    # Maybe move this method to RunsArtifactRepository so the circular dependency is clearer.
    assert urllib.parse.urlparse(run.info.artifact_uri).scheme != "runs"  # avoid an infinite loop
    if artifact_path is None:
        return run.info.artifact_uri
    else:
        return append_to_uri_path(run.info.artifact_uri, artifact_path)


# TODO: This would be much simpler if artifact_repo.download_artifacts could take the absolute path
# or no path.
def _get_root_uri_and_artifact_path(artifact_uri):
    """Parse the artifact_uri to get the root_uri and artifact_path.

    Args:
        artifact_uri: The *absolute* URI of the artifact.
    """
    if os.path.exists(artifact_uri):
        if not is_windows():
            # If we're dealing with local files, just reference the direct pathing.
            # non-nt-based file systems can directly reference path information, while nt-based
            # systems need to url-encode special characters in directory listings to be able to
            # resolve them (i.e., spaces converted to %20 within a file name or path listing)
            root_uri = os.path.dirname(artifact_uri)
            artifact_path = os.path.basename(artifact_uri)
            return root_uri, artifact_path
        else:  # if we're dealing with nt-based systems, we need to utilize pathname2url to encode.
            artifact_uri = path_to_local_file_uri(artifact_uri)

    parsed_uri = urllib.parse.urlparse(str(artifact_uri))
    prefix = ""
    if parsed_uri.scheme and not parsed_uri.path.startswith("/"):
        # relative path is a special case, urllib does not reconstruct it properly
        prefix = parsed_uri.scheme + ":"
        parsed_uri = parsed_uri._replace(scheme="")

    # For models:/ URIs, it doesn't make sense to initialize a ModelsArtifactRepository with only
    # the model name portion of the URI, then call download_artifacts with the version info.
    if ModelsArtifactRepository.is_models_uri(artifact_uri):
        root_uri, artifact_path = ModelsArtifactRepository.split_models_uri(artifact_uri)
    else:
        artifact_path = posixpath.basename(parsed_uri.path)
        parsed_uri = parsed_uri._replace(path=posixpath.dirname(parsed_uri.path))
        root_uri = prefix + urllib.parse.urlunparse(parsed_uri)
    return root_uri, artifact_path


def _download_artifact_from_uri(artifact_uri, output_path=None):
    """
    Args:
        artifact_uri: The *absolute* URI of the artifact to download.
        output_path: The local filesystem path to which to download the artifact. If unspecified,
            a local output path will be created.
    """
    root_uri, artifact_path = _get_root_uri_and_artifact_path(artifact_uri)
    return get_artifact_repository(artifact_uri=root_uri).download_artifacts(
        artifact_path=artifact_path, dst_path=output_path
    )


def _upload_artifact_to_uri(local_path, artifact_uri):
    """Uploads a local artifact (file) to a specified URI.

    Args:
        local_path: The local path of the file to upload.
        artifact_uri: The *absolute* URI of the path to upload the artifact to.

    """
    root_uri, artifact_path = _get_root_uri_and_artifact_path(artifact_uri)
    get_artifact_repository(artifact_uri=root_uri).log_artifact(local_path, artifact_path)


def _upload_artifacts_to_databricks(
    source, run_id, source_host_uri=None, target_databricks_profile_uri=None
):
    """Copy the artifacts from ``source`` to the destination Databricks workspace (DBFS) given by
    ``databricks_profile_uri`` or the current tracking URI.

    Args:
        source: Source location for the artifacts to copy.
        run_id: Run ID to associate the artifacts with.
        source_host_uri: Specifies the source artifact's host URI (e.g. Databricks tracking URI)
            if applicable. If not given, defaults to the current tracking URI.
        target_databricks_profile_uri: Specifies the destination Databricks host. If not given,
            defaults to the current tracking URI.

    Returns:
        The DBFS location in the target Databricks workspace the model files have been
        uploaded to.
    """

    with tempfile.TemporaryDirectory() as local_dir:
        source_with_profile = add_databricks_profile_info_to_artifact_uri(source, source_host_uri)
        _download_artifact_from_uri(source_with_profile, local_dir)
        dest_root = "dbfs:/databricks/mlflow/tmp-external-source/"
        dest_root_with_profile = add_databricks_profile_info_to_artifact_uri(
            dest_root, target_databricks_profile_uri
        )
        dest_repo = DbfsRestArtifactRepository(dest_root_with_profile)
        dest_artifact_path = run_id if run_id else uuid.uuid4().hex
        # Allow uploading from the same run id multiple times by randomizing a suffix
        if len(dest_repo.list_artifacts(dest_artifact_path)) > 0:
            dest_artifact_path = dest_artifact_path + "-" + uuid.uuid4().hex[0:4]
        dest_repo.log_artifacts(local_dir, artifact_path=dest_artifact_path)
        dirname = pathlib.PurePath(source).name  # innermost directory name
        return posixpath.join(dest_root, dest_artifact_path, dirname)  # new source
