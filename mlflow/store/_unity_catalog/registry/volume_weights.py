"""
Registration of models whose weights are in a Unity Catalog volume. The model's own files hold
no weights; Databricks copies them from the volume into the model version, so the version can be
finalized only once that copy has finished.
"""

import logging
import os
import posixpath
import time
from typing import Any

from mlflow.environment_variables import MLFLOW_DATABRICKS_VOLUME_WEIGHTS_COPY_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import FEATURE_DISABLED, INTERNAL_ERROR, ErrorCode
from mlflow.utils.databricks_utils import (
    get_databricks_model_serving_staging_status,
    stage_model_for_databricks_model_serving,
)

_logger = logging.getLogger(__name__)

VOLUME_ARTIFACTS_PATH_METADATA_KEY = "volume_artifacts_path"

_POLL_INTERVAL_SECONDS = 10
# A copy that fails is started again this many times before registration gives up.
_MAX_COPY_RESTARTS = 2

_STATE_SUCCEEDED = "SUCCEEDED"
_STATE_FAILED = "FAILED"
_STATE_NOT_STAGED = "NOT_STAGED"


def get_volume_artifacts_path(model) -> str | None:
    """
    Returns the Unity Catalog volume path set as ``volume_artifacts_path`` in the model's
    metadata, or None when the model carries its weights itself.
    """
    path = (model.metadata or {}).get(VOLUME_ARTIFACTS_PATH_METADATA_KEY)
    if path is None:
        return None
    # /Volumes/<catalog>/<schema>/<volume> is the shortest path inside a volume.
    if (
        not isinstance(path, str)
        or not path.startswith("/Volumes/")
        or len(path.strip("/").split("/")) < 4
    ):
        raise MlflowException.invalid_parameter_value(
            f"The model's '{VOLUME_ARTIFACTS_PATH_METADATA_KEY}' metadata must be a Unity Catalog "
            f"volume path such as '/Volumes/<catalog>/<schema>/<volume>/<path>', but it is "
            f"{path!r}. Log the model again with a volume path."
        )
    return path


def volume_weights_in_model_dir(local_model_dir: str, volume_path: str) -> bool:
    """
    Returns whether the model's own files already hold the weights Databricks copies from the
    volume, as when registering a copy of a version whose weights were already copied.
    """
    # Databricks copies the volume directory to artifacts/<its last path segment>/.
    weights_dir = os.path.join(
        local_model_dir, "artifacts", posixpath.basename(volume_path.rstrip("/"))
    )
    if not os.path.isdir(weights_dir):
        return False
    with os.scandir(weights_dir) as entries:
        return any(entries)


def _error_code_value(error_code_name: str) -> int:
    # The server can return codes this client's ErrorCode enum doesn't define.
    try:
        return ErrorCode.Value(error_code_name)
    except ValueError:
        return INTERNAL_ERROR


def start_volume_weights_copy(name: str, version: str, host_creds) -> dict[str, Any]:
    """
    Asks Databricks to copy the weights into the PENDING_REGISTRATION model version and returns
    the copy's staging status.
    """
    try:
        stage_model_for_databricks_model_serving(name, version, host_creds=host_creds)
    except MlflowException as e:
        # A workspace that can't copy weights into a version that isn't finalized yet rejects
        # the request with NOT_FOUND.
        if e.error_code == "NOT_FOUND":
            raise MlflowException(
                f"This Databricks workspace can't copy the weights of model '{name}' version "
                f"{version} from its Unity Catalog volume yet. To register the model, log it "
                "again with its weights in the model's own files.",
                error_code=FEATURE_DISABLED,
            ) from e
        raise MlflowException(
            f"Databricks couldn't copy the weights of model '{name}' version {version} from its "
            f"Unity Catalog volume into the model version: {e.message} To register the model "
            "anyway, log it again with its weights in the model's own files.",
            error_code=_error_code_value(e.error_code),
        ) from e
    try:
        status = get_databricks_model_serving_staging_status(name, version, host_creds=host_creds)
    except MlflowException as e:
        if e.error_code != "FEATURE_DISABLED":
            raise
        status = None
    if not status:
        raise MlflowException(
            f"This workspace doesn't report when Databricks finishes copying the weights of model "
            f"'{name}' version {version} from its Unity Catalog volume, so the version can't be "
            "finalized safely. To register the model, log it again with its weights in the "
            "model's own files."
        )
    return status


def wait_for_volume_weights_copy(
    name: str, version: str, status: dict[str, Any], host_creds
) -> None:
    """
    Polls the copy until the model weights are in the version. A failed copy is started again
    up to ``_MAX_COPY_RESTARTS`` times.
    """
    timeout = MLFLOW_DATABRICKS_VOLUME_WEIGHTS_COPY_TIMEOUT.get()
    deadline = time.monotonic() + timeout
    restarts = 0
    _logger.info(
        "Waiting for Databricks to copy the weights of model '%s' version %s from its Unity "
        "Catalog volume.",
        name,
        version,
    )
    while True:
        # Only the weights must be in the version before it is finalized; the serving
        # environment, the other artifact in the status, is prepared independently.
        state = status.get("model_artifact", {}).get("state")
        if state == _STATE_SUCCEEDED:
            return
        remaining = deadline - time.monotonic()
        if state in (_STATE_FAILED, _STATE_NOT_STAGED):
            # A copy started after the deadline couldn't finish in time, so report the failure.
            if restarts == _MAX_COPY_RESTARTS or remaining <= 0:
                raise MlflowException(
                    f"Databricks couldn't copy the weights of model '{name}' version {version} "
                    "from its Unity Catalog volume into the model version. Check that the volume "
                    "path in the model's metadata exists and that you can read it, then register "
                    "the model again."
                )
            restarts += 1
            _logger.warning(
                "Copying the weights of model '%s' version %s failed; starting it again "
                "(attempt %d of %d).",
                name,
                version,
                restarts + 1,
                _MAX_COPY_RESTARTS + 1,
            )
            status = start_volume_weights_copy(name, version, host_creds)
            continue
        if remaining <= 0:
            raise MlflowException(
                f"Databricks didn't finish copying the weights of model '{name}' version "
                f"{version} from its Unity Catalog volume within {timeout} seconds. Register the "
                f"model again, or raise {MLFLOW_DATABRICKS_VOLUME_WEIGHTS_COPY_TIMEOUT.name} for "
                "larger models."
            )
        time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))
        status = get_databricks_model_serving_staging_status(name, version, host_creds=host_creds)
