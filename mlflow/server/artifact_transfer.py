from mlflow.environment_variables import MLFLOW_ARTIFACTS_PRESIGNED_ONLY
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED
from mlflow.version import VERSION

_PRESIGNED_ONLY_UPLOAD_MESSAGE = (
    "This server requires artifacts to be uploaded directly to storage. "
    f"Upgrade your MLflow client to match this server (MLflow {VERSION}) or a newer version "
    "with presigned-only artifact support, then retry the upload. "
    "If you are using the MLflow UI, reload the page to refresh the server settings."
)
_PRESIGNED_ONLY_DOWNLOAD_MESSAGE = (
    "This server requires artifacts to be downloaded directly from storage. "
    f"Upgrade your MLflow client to match this server (MLflow {VERSION}) or a newer version "
    "with presigned-only artifact support, then retry the download. "
    "If you are using the MLflow UI, reload the page to refresh the server settings."
)


def reject_proxied_artifact_upload() -> None:
    if MLFLOW_ARTIFACTS_PRESIGNED_ONLY.get():
        raise MlflowException(
            _PRESIGNED_ONLY_UPLOAD_MESSAGE,
            error_code=PERMISSION_DENIED,
            artifacts_presigned_only=True,
        )


def reject_proxied_artifact_download() -> None:
    if MLFLOW_ARTIFACTS_PRESIGNED_ONLY.get():
        raise MlflowException(
            _PRESIGNED_ONLY_DOWNLOAD_MESSAGE,
            error_code=PERMISSION_DENIED,
            artifacts_presigned_only=True,
        )
