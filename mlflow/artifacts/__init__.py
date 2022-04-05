"""
APIs for interacting with artifacts in MLflow
"""
import pathlib
from typing import Optional

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking import _get_store
from mlflow.tracking.artifact_utils import _download_artifact_from_uri, get_artifact_repository


def download_artifacts(
    artifact_uri: Optional[str] = None,
    run_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
    dst_path: Optional[str] = None,
) -> str:
    """
    Download an artifact file or directory to a local directory.

    :param artifact_uri: URI pointing to the artifacts, such as
                         ``"runs:/500cf58bee2b40a4a82861cc31a617b1/my_model.pkl"``,
                         ``"models:/my_model/Production"``, or ``"s3://my_bucket/my/file.txt"``.
                         Exactly one of ``artifact_uri`` or ``run_id`` must be specified.
    :param run_id: ID of the MLflow Run containing the artifacts. Exactly one of ``run_id`` or
                   ``artifact_uri`` must be specified.
    :param artifact_path: (For use with ``run_id``) If specified, a path relative to the MLflow
                          Run's root directory containing the artifacts to download.
    :param dst_path: Path of the local filesystem destination directory to which to download the
                     specified artifacts. If the directory does not exist, it is created. If
                     unspecified, the artifacts are downloaded to a new uniquely-named directory on
                     the local filesystem, unless the artifacts already exist on the local
                     filesystem, in which case their local path is returned directly.
    :return: The location of the artifact file or directory on the local filesystem.
    """
    if (run_id, artifact_uri).count(None) != 1:
        raise MlflowException(
            message="Exactly one of `run_id` or `artifact_uri` must be specified",
            error_code=INVALID_PARAMETER_VALUE,
        )
    elif artifact_uri is not None and artifact_path is not None:
        raise MlflowException(
            message="`artifact_path` cannot be specified if `artifact_uri` is specified",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if dst_path is not None:
        pathlib.Path(dst_path).mkdir(exist_ok=True, parents=True)

    if artifact_uri is not None:
        return _download_artifact_from_uri(artifact_uri, output_path=dst_path)

    artifact_path = artifact_path if artifact_path is not None else ""
    store = _get_store()
    artifact_uri = store.get_run(run_id).info.artifact_uri
    artifact_repo = get_artifact_repository(artifact_uri)
    artifact_location = artifact_repo.download_artifacts(artifact_path, dst_path=dst_path)
    return artifact_location
