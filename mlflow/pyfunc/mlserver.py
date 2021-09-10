import asyncio
import os

from typing import Tuple, Dict

from mlserver.server import MLServer
from mlserver.cli.serve import load_settings

MLServerMLflowRuntime = "mlserver_mlflow.MLflowRuntime"


def _predict(
    model_uri: str, input_path: str, output_path: str, content_type: str, json_format: str
):
    pass


def _serve(model_uri: str, port: int, host: str):
    settings, models = asyncio.run(load_settings(model_uri))

    settings.host = host
    settings.http_port = port

    server = MLServer(settings)
    asyncio.run(server.start(models))


def get_cmd(model_uri: str, port: int, host: str, nworkers: int) -> Tuple[str, Dict[str, str]]:
    cmd = f"mlserver start {model_uri}"

    cmd_env = os.environ.copy()
    cmd_env["MLSERVER_HTTP_PORT"] = str(port)
    cmd_env["MLSERVER_HOST"] = host

    # TODO: What name should it have?
    #  cmd_env["MLSERVER_MODEL_NAME"] = model_details.name,
    cmd_env["MLSERVER_MODEL_PARALLEL_WORKERS"] = str(nworkers)
    cmd_env["MLSERVER_MODEL_IMPLEMENTATION"] = MLServerMLflowRuntime
    cmd_env["MLSERVER_MODEL_URI"] = model_uri

    return cmd, cmd_env
