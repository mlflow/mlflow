"""
Stages a Unity Catalog model version for Databricks Model Serving while it is registered, for
models whose weights are in a Unity Catalog volume (``volume_artifacts_path`` in their metadata).
The model's own files hold no weights. Staging the PENDING_REGISTRATION version copies the
weights from the volume into the version as part of staging the model, and builds and packs the
model's serving environment. The version can be finalized once the model, with its weights, is
staged; the environment is staged independently.

Registering with ``env_pack`` stages a finalized version through a different path and doesn't
use this module.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any

from mlflow.environment_variables import MLFLOW_DATABRICKS_VOLUME_WEIGHTS_STAGING_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, ErrorCode
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

# Databricks finds the weights of a version that isn't finalized yet through the logged model that
# the version's source names, in either of these forms.
_LOGGED_MODEL_SOURCE = re.compile(r"models:/m-[0-9a-f]+|.*/logged_models/m-[0-9a-f]+/artifacts/?")
# A registered model version by number or alias, as when a version is copied.
_MODEL_VERSION_SOURCE = re.compile(r"models:/(?:[^/?#]+/[0-9]+|[^/?#]+@[^/@?#\s]+)")


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


def is_model_version_copy(source) -> bool:
    """
    Returns whether ``source`` is a registered model version (``models:/<name>/<version>`` or
    ``models:/<name>@<alias>``), as when a version is copied. Its files hold the weights already,
    so they're uploaded like any model's files instead of being staged.
    """
    return _MODEL_VERSION_SOURCE.fullmatch(str(source)) is not None


def check_volume_staging_source(source, volume_path: str, model_dir, model_id=None) -> None:
    """
    Rejects, before the version is created, a ``source`` the version can't get its weights from.
    A source that names a logged model is staged. A copy of a registered model version must
    already contain the directory its weights were staged into,
    ``artifacts/<last segment of the volume path>``. ``model_id`` is the model's logged model ID
    from its MLmodel file, if it has one, for the error message.
    """
    if _LOGGED_MODEL_SOURCE.fullmatch(str(source)):
        return
    logged_model_uri = f"models:/{model_id}" if model_id else "models:/<model_id>"
    if is_model_version_copy(source):
        weights_dir = Path(model_dir, "artifacts", volume_path.rstrip("/").rsplit("/", 1)[-1])
        # The upload doesn't follow a symlinked directory, so its files wouldn't be copied.
        if weights_dir.is_dir() and not weights_dir.is_symlink() and any(weights_dir.iterdir()):
            return
        raise MlflowException.invalid_parameter_value(
            f"Model version '{source}' doesn't contain the weights from the Unity Catalog volume "
            f"in its metadata ('{volume_path}'), so a copy of it wouldn't have them either. "
            f"Register the model from its logged model ('{logged_model_uri}') instead, so that "
            "Databricks copies the weights from the volume, or log the model with its weights in "
            f"its own files and without '{VOLUME_ARTIFACTS_PATH_METADATA_KEY}' in its metadata."
        )
    advice = (
        f"Register it by its logged model URI '{logged_model_uri}'."
        if model_id
        else "Log the model with `mlflow.<flavor>.log_model()`, then register it by its "
        "'models:/<model_id>' URI."
    )
    raise MlflowException.invalid_parameter_value(
        "A model whose weights are in a Unity Catalog volume must be registered from its logged "
        "model ('models:/<model_id>') or copied from a registered model version "
        "('models:/<name>/<version>' or 'models:/<name>@<alias>'), but this model's source is "
        f"{str(source)!r}. {advice}"
    )


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
        # NOT_FOUND means Databricks didn't stage the version before it's finalized, for a reason
        # it doesn't report: the workspace doesn't support it, the model isn't eligible, or the
        # request failed temporarily.
        if e.error_code == "NOT_FOUND":
            raise MlflowException(
                f"Databricks couldn't stage model '{name}' version {version} for Model Serving "
                "before it's finalized, so the weights in the model's Unity Catalog volume weren't "
                "copied into the version. The workspace may not support this yet, the model may "
                "not be eligible, or the request may have failed temporarily. Try registering the "
                "model again. If it fails again, log the model with its weights in its own files "
                f"and without '{VOLUME_ARTIFACTS_PATH_METADATA_KEY}' in its metadata, and register "
                "that.",
                error_code=_error_code_value(e.error_code),
            ) from e
        raise MlflowException(
            f"Databricks couldn't stage model '{name}' version {version} for Model Serving, "
            f"which copies its weights from its Unity Catalog volume into the model version: "
            f"{e.message} To register the model anyway, log it again with its weights in the "
            f"model's own files and without '{VOLUME_ARTIFACTS_PATH_METADATA_KEY}' in its "
            "metadata.",
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
            "log it again with its weights in the model's own files and without "
            f"'{VOLUME_ARTIFACTS_PATH_METADATA_KEY}' in its metadata."
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
            if (status.get("environment_artifact") or {}).get("state") == _STATE_FAILED:
                _logger.warning(
                    "Databricks couldn't prepare the serving environment of model '%s' version %s "
                    "from its requirements, so a Model Serving endpoint for this version may be "
                    "slower to start or fail to start. Check the model's requirements, such as "
                    "the package versions it pins, then log and register the model again.",
                    name,
                    version,
                )
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
            # Restarting takes time, so the wait before that poll mustn't outlast the deadline.
            remaining = max(deadline - time.monotonic(), 0)
        elif remaining <= 0:
            raise MlflowException(
                f"Databricks didn't finish staging model '{name}' version {version} for Model "
                f"Serving, which copies its weights from its Unity Catalog volume, within "
                f"{timeout} seconds. Register the model again, or raise "
                f"{MLFLOW_DATABRICKS_VOLUME_WEIGHTS_STAGING_TIMEOUT.name} for larger models."
            )
        time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))
        status = get_databricks_model_serving_staging_status(name, version, host_creds=host_creds)
