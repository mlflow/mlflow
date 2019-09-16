"""
Utilities for dealing with artifacts in the context of a Run.
"""
import posixpath
import os

from six.moves import urllib

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact_repository_registry import get_artifact_repository, \
    get_artifact_repository_type, ArtifactRepositoryType
from mlflow.tracking.utils import _get_store
from mlflow.store.db_artifact_repo import ROOT_PATH_BASE


def get_artifact_uri(run_id, artifact_path=None):
    """
    Get the absolute URI of the specified artifact in the specified run. If `path` is not specified,
    the artifact root URI of the specified run will be returned; calls to ``log_artifact``
    and ``log_artifacts`` write artifact(s) to subdirectories of the artifact root URI.

    :param run_id: The ID of the run for which to obtain an absolute artifact URI.
    :param artifact_path: The run-relative artifact path. For example,
                          ``path/to/artifact``. If unspecified, the artifact root URI for the
                          specified run will be returned.
    :return: An *absolute* URI referring to the specified artifact or the specified run's artifact
             root. For example, if an artifact path is provided and the specified run uses an
             S3-backed store, this may be a uri of the form
             ``s3://<bucket_name>/path/to/artifact/root/path/to/artifact``. If an artifact path
             is not provided and the specified run uses an S3-backed store, this may be a URI of
             the form ``s3://<bucket_name>/path/to/artifact/root``.
    """
    if not run_id:
        raise MlflowException(
            message="A run_id must be specified in order to obtain an artifact uri!",
            error_code=INVALID_PARAMETER_VALUE)

    store = _get_store()
    run = store.get_run(run_id)
    # Maybe move this method to RunsArtifactRepository so the circular dependency is clearer.
    assert urllib.parse.urlparse(run.info.artifact_uri).scheme != "runs"  # avoid an infinite loop
    if artifact_path is None:
        return run.info.artifact_uri
    else:
        return posixpath.join(run.info.artifact_uri, artifact_path)


# TODO: This method does not require a Run and its internals should be moved to
#  data.download_uri (requires confirming that Projects will not break with this change).
# Also this would be much simpler if artifact_repo.download_artifacts could take the absolute path
# or no path.
def _download_artifact_from_uri(artifact_uri, output_path=None):
    """
    :param artifact_uri: The *absolute* URI of the artifact to download.
    :param output_path: The local filesystem path to which to download the artifact. If unspecified,
                        a local output path will be created.
    """
    artifact_repo_type = get_artifact_repository_type(artifact_uri)

    if artifact_repo_type == ArtifactRepositoryType.FileSystem:
        parsed_uri = urllib.parse.urlparse(artifact_uri)
        prefix = ""
        if parsed_uri.scheme and not parsed_uri.path.startswith("/"):
            # relative path is a special case, urllib does not reconstruct it properly
            prefix = parsed_uri.scheme + ":"
            parsed_uri = parsed_uri._replace(scheme="")
        artifact_path = posixpath.basename(parsed_uri.path)
        parsed_uri = parsed_uri._replace(path=posixpath.dirname(parsed_uri.path))
        root_uri = prefix + urllib.parse.urlunparse(parsed_uri)
        return get_artifact_repository(artifact_uri=root_uri).download_artifacts(
            artifact_path=artifact_path, dst_path=output_path)
    else:
        repo_uri, relative_path = extract_repo_uri_and_relative_artifact_path(artifact_uri)
        return get_artifact_repository(artifact_uri=repo_uri).download_artifacts(
            artifact_path=relative_path, dst_path=output_path)


def extract_repo_uri_and_relative_artifact_path(artifact_uri):
    """
    Parse the specified artifact URI to extract the repository uri and the
    relative artifact path.
    The repo_uri is of the form DB_URI/runID/ROOT_PATH_BASE where DB_URI:
    <dialect>+<driver>://<username>:<password>@<host>:<port>/<database>?<query>.
    """

    split_uri = artifact_uri.split(ROOT_PATH_BASE + os.sep, 1)
    repo_uri = split_uri[0] + ROOT_PATH_BASE
    relative_path = split_uri[1]
    return repo_uri, relative_path
