import logging
import os
from typing import Dict, Tuple

_logger = logging.getLogger(__name__)

MLServerMLflowRuntime = "mlserver_mlflow.MLflowRuntime"
MLServerDefaultModelName = "mlflow-model"


def get_cmd(
    model_uri: str,
    port: int = None,
    host: str = None,
    timeout: int = None,
    nworkers: int = None,
    model_name: str = None,
    model_version: str = None,
) -> Tuple[str, Dict[str, str]]:
    cmd = f"mlserver start {model_uri}"

    cmd_env = os.environ.copy()

    if port:
        cmd_env["MLSERVER_HTTP_PORT"] = str(port)

    if host:
        cmd_env["MLSERVER_HOST"] = host

    if timeout:
        _logger.warning("Timeout is not yet supported in MLServer.")

    if nworkers:
        cmd_env["MLSERVER_PARALLEL_WORKERS"] = str(nworkers)

    # give precedence to user env var input
    cmd_env["MLSERVER_MODEL_NAME"] = (
        cmd_env.get("MLSERVER_MODEL_NAME") or model_name or MLServerDefaultModelName
    )
    if model_version and not cmd_env.get("MLSERVER_MODEL_VERSION"):
        cmd_env["MLSERVER_MODEL_VERSION"] = model_version

    cmd_env["MLSERVER_MODEL_IMPLEMENTATION"] = MLServerMLflowRuntime
    cmd_env["MLSERVER_MODEL_URI"] = model_uri

    return cmd, cmd_env
