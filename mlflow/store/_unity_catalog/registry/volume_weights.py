"""
Registration of models whose weights are in a Unity Catalog volume. The model's own files hold
no weights. Registration stages the PENDING_REGISTRATION version for Databricks Model Serving,
which copies the weights from the volume into the version as part of staging the model, and
builds and packs the model's serving environment. The version can be finalized once the model,
with its weights, is staged; the environment is staged independently.
"""

import logging
import time
from typing import Any

from mlflow.environment_variables import MLFLOW_DATABRICKS_VOLUME_WEIGHTS_STAGING_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import FEATURE_DISABLED, INTERNAL_ERROR, ErrorCode
from mlflow.utils.databricks_utils import (
    get_databricks_model_serving_staging_status,
    stage_model_for_databricks_model_serving,
)

_logger = logging.getLogger(__name__)

VOLUME_ARTIFACTS_PATH_METADATA_KEY = "volume_artifacts_path"

_POLL_INTERVAL_SECONDS = 10
# Staging that fails is started again this many times before registration gives up. Databricks
# restages only the failed model, so the environment isn't built again.
_MAX_STAGING_RESTARTS = 2

_STATE_SUCCEEDED = "SUCCEEDED"
_STATE_FAILED = "FAILED"
_STATE_NOT_STAGED = "NOT_STAGED"


def get_volume_artifacts_path(model) -> str | None:
    """
    Returns the Unity Catalog volume path set as ``volume_artifacts_path`` in the model's
    metadata, or None when the model carries its weights itself.
    """
    metadata = model.metadata
    if not isinstance(metadata, dict):
        return None
    path = metadata.get(VOLUME_ARTIFACTS_PATH_METADATA_KEY)
    if path is None:
        return None
    segments = path.rstrip("/").split("/") if isinstance(path, str) else []
    # /Volumes/<catalog>/<schema>/<volume> is the shortest path inside a volume.
    if (
        segments[:2] != ["", "Volumes"]
        or len(segments) < 5
        or any(segment in ("", ".", "..") for segment in segments[1:])
    ):
        raise MlflowException.invalid_parameter_value(
            f"The model's '{VOLUME_ARTIFACTS_PATH_METADATA_KEY}' metadata must be a Unity Catalog "
            f"volume path such as '/Volumes/<catalog>/<schema>/<volume>/<path>', but it is "
            f"{path!r}. Log the model again with a volume path."
        )
    return path


def get_staging_timeout() -> int:
    """
    Returns how many seconds to wait for staging. Read before staging starts, so a bad value
    fails before anything is staged or uploaded.
    """
    name = MLFLOW_DATABRICKS_VOLUME_WEIGHTS_STAGING_TIMEOUT.name
    try:
        timeout = MLFLOW_DATABRICKS_VOLUME_WEIGHTS_STAGING_TIMEOUT.get()
    except ValueError:
        timeout = None
    if timeout is None or timeout <= 0:
        raise MlflowException.invalid_parameter_value(
            f"{name} must be a positive whole number of seconds, but it is "
            f"{MLFLOW_DATABRICKS_VOLUME_WEIGHTS_STAGING_TIMEOUT.get_raw()!r}. Set it to a "
            "positive number of seconds, or unset it to wait the default 7200 seconds."
        )
    return timeout


def _error_code_value(error_code_name: str) -> int:
    # The server can return codes this client's ErrorCode enum doesn't define.
    try:
        return ErrorCode.Value(error_code_name)
    except ValueError:
        return INTERNAL_ERROR


def start_model_serving_staging(name: str, version: str, host_creds) -> dict[str, Any]:
    """
    Stages the PENDING_REGISTRATION model version for Databricks Model Serving, which copies the
    weights from the volume into the version and prepares its serving environment, and returns
    the staging status.
    """
    try:
        stage_model_for_databricks_model_serving(name, version, host_creds=host_creds)
    except MlflowException as e:
        # A workspace that can't stage a version that isn't finalized yet rejects the request
        # with NOT_FOUND.
        if e.error_code == "NOT_FOUND":
            raise MlflowException(
                f"This Databricks workspace can't stage model '{name}' version {version} for "
                "Model Serving before it's finalized, so it can't copy the weights from the "
                "model's Unity Catalog volume. To register the model, log it again with its "
                "weights in the model's own files.",
                error_code=FEATURE_DISABLED,
            ) from e
        raise MlflowException(
            f"Databricks couldn't stage model '{name}' version {version} for Model Serving, "
            f"which copies its weights from its Unity Catalog volume into the model version: "
            f"{e.message} To register the model anyway, log it again with its weights in the "
            "model's own files.",
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
            f"This workspace doesn't report when Databricks finishes staging model '{name}' "
            f"version {version} for Model Serving, which copies its weights from its Unity "
            "Catalog volume, so the version can't be finalized safely. To register the model, "
            "log it again with its weights in the model's own files."
        )
    return status


def wait_for_model_artifact_staging(
    name: str, version: str, status: dict[str, Any], host_creds, timeout: int
) -> None:
    """
    Polls the staging status until the model, with the weights copied into the version, is
    staged, for at most ``timeout`` seconds. Staging that fails is started again up to
    ``_MAX_STAGING_RESTARTS`` times while time remains.
    """
    deadline = time.monotonic() + timeout
    restarts = 0
    _logger.info(
        "Waiting for Databricks to stage model '%s' version %s for Model Serving, which copies "
        "its weights from its Unity Catalog volume.",
        name,
        version,
    )
    while True:
        # Only the weights must be in the version before it is finalized; the serving
        # environment, the other artifact in the status, is staged independently.
        state = (status.get("model_artifact") or {}).get("state")
        if state == _STATE_SUCCEEDED:
            return
        remaining = deadline - time.monotonic()
        if state in (_STATE_FAILED, _STATE_NOT_STAGED):
            # Staging started again after the deadline couldn't finish in time, so report the
            # failure instead.
            if restarts == _MAX_STAGING_RESTARTS or remaining <= 0:
                raise MlflowException(
                    f"Databricks couldn't copy the weights of model '{name}' version {version} "
                    "from its Unity Catalog volume into the model version while staging it for "
                    "Model Serving. Check that the volume path in the model's metadata exists "
                    "and that you can read it, then register the model again."
                )
            restarts += 1
            _logger.warning(
                "Staging model '%s' version %s for Model Serving failed; starting it again "
                "(attempt %d of %d).",
                name,
                version,
                restarts + 1,
                _MAX_STAGING_RESTARTS + 1,
            )
            # The status read right after restarting can still show the failed attempt, so the
            # restart is judged by the next poll.
            start_model_serving_staging(name, version, host_creds)
        elif remaining <= 0:
            raise MlflowException(
                f"Databricks didn't finish staging model '{name}' version {version} for Model "
                f"Serving, which copies its weights from its Unity Catalog volume, within "
                f"{timeout} seconds. Register the model again, or raise "
                f"{MLFLOW_DATABRICKS_VOLUME_WEIGHTS_STAGING_TIMEOUT.name} for larger models."
            )
        time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))
        status = get_databricks_model_serving_staging_status(name, version, host_creds=host_creds)
