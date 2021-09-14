import asyncio
import os

from typing import Tuple, Dict

from mlserver.server import MLServer
from mlserver.cli.serve import load_settings

MLServerMLflowRuntime = "mlserver_mlflow.MLflowRuntime"


def get_cmd(
    model_uri: str, port: int = None, host: str = None, nworkers: int = None
) -> Tuple[str, Dict[str, str]]:
    cmd = f"mlserver start {model_uri}"

    cmd_env = os.environ.copy()

    if port:
        cmd_env["MLSERVER_HTTP_PORT"] = str(port)

    if host:
        cmd_env["MLSERVER_HOST"] = host

    # TODO: What name should it have?
    #  cmd_env["MLSERVER_MODEL_NAME"] = model_details.name,

    if nworkers:
        cmd_env["MLSERVER_MODEL_PARALLEL_WORKERS"] = str(nworkers)

    cmd_env["MLSERVER_MODEL_IMPLEMENTATION"] = MLServerMLflowRuntime
    cmd_env["MLSERVER_MODEL_URI"] = model_uri

    return cmd, cmd_env
