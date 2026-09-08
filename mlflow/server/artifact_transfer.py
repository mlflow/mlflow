from mlflow.environment_variables import MLFLOW_ARTIFACTS_ONLY_PRESIGNED
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_CONFLICT

_PRESIGNED_ONLY_UPLOAD_MESSAGE = (
    "This server requires presigned artifact uploads and does not accept artifact bytes through "
    "the legacy proxy endpoint. Upgrade your MLflow client to a version that supports "
    "presigned-only artifact servers."
)
_PRESIGNED_ONLY_DOWNLOAD_MESSAGE = (
    "This server requires presigned artifact downloads and does not serve artifact bytes through "
    "the legacy proxy endpoint. Upgrade your MLflow client to a version that supports "
    "presigned-only artifact servers."
)


def reject_legacy_artifact_upload() -> None:
    if MLFLOW_ARTIFACTS_ONLY_PRESIGNED.get():
        raise MlflowException(_PRESIGNED_ONLY_UPLOAD_MESSAGE, error_code=RESOURCE_CONFLICT)


def reject_legacy_artifact_download() -> None:
    if MLFLOW_ARTIFACTS_ONLY_PRESIGNED.get():
        raise MlflowException(_PRESIGNED_ONLY_DOWNLOAD_MESSAGE, error_code=RESOURCE_CONFLICT)
