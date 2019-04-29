"""
Utilities for dealing with artifacts in the context of a Run.
"""

from six.moves import urllib

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact_repository_registry import get_artifact_repository
from mlflow.tracking.utils import _get_store


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
             S3-backed  store, this may be a uri of the form
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
        # Path separators may not be consistent across all artifact repositories. Therefore, when
        # joining the run's artifact root directory with the artifact's relative path, we use the
        # path module defined by the appropriate artifact repository
        artifact_path_module = \
            get_artifact_repository(run.info.artifact_uri).get_path_module()
        return artifact_path_module.join(run.info.artifact_uri, artifact_path)


# TODO: This method does not require a Run and its internals should be moved to
#  data.download_uri (requires confirming that Projects will not break with this change).
def _download_artifact_from_uri(artifact_uri, output_path=None):
    """
    :param artifact_uri: The *absolute* URI of the artifact to download.
    :param output_path: The local filesystem path to which to download the artifact. If unspecified,
                        a local output path will be created.
    """
    artifact_path_module = \
        get_artifact_repository(artifact_uri).get_path_module()
    artifact_src_dir = artifact_path_module.dirname(artifact_uri)
    artifact_src_relative_path = artifact_path_module.basename(artifact_uri)
    artifact_repo = get_artifact_repository(artifact_uri=artifact_src_dir)
    return artifact_repo.download_artifacts(artifact_path=artifact_src_relative_path,
                                            dst_path=output_path)


def _get_model_log_dir(model_name, run_id):
    if not run_id:
        raise Exception("Must specify a run_id to get logging directory for a model.")
    store = _get_store()
    run = store.get_run(run_id)
    artifact_repo = get_artifact_repository(run.info.artifact_uri)
    return artifact_repo.download_artifacts(model_name)
