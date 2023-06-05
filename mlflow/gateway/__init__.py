import os
import pathlib
import signal
import sys
import time
from typing import Optional

from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import CONF_PATH_ENV_VAR, LOCALHOST
from mlflow.gateway.handlers import _load_gateway_config, RouteConfig
from mlflow.protos.databricks_pb2 import BAD_REQUEST

import subprocess

server_process: Optional[subprocess.Popen] = None
gateway_host: Optional[str] = None
gateway_port: Optional[int] = None


def start_server(config_path: str, host, port):
    """
    Starts the server with the given configuration, host and port.

    This function sets the global server_process, gateway_host, and gateway_port variables.
    It starts the server by creating a new subprocess that runs the server script.

    Args:
        config_path (str): Path to the configuration file.
        host (str): The hostname to use for the server.
        port (int): The port number to use for the server.

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the server subprocess fails to start.
    """
    global server_process
    global gateway_host
    global gateway_port
    os.environ[CONF_PATH_ENV_VAR] = config_path

    if server_process is not None:
        raise MlflowException(
            f"There is a currently running server instance at pid '{server_process.pid}'. Please "
            "terminate the running server instance prior to starting a new instance within this "
            "context.",
            error_code=BAD_REQUEST,
        )

    cmd_path = pathlib.Path(__file__).parent.joinpath("gateway_app.py")
    server_cmd = [sys.executable, str(cmd_path), "--host", host, "--port", str(port)]
    gateway_host = host
    gateway_port = port
    server_process = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _stop_server():
    """
    Stops the currently running server.

    This function kills the process of the running server and resets the global server_process variable to None.

    Args:
        None

    Returns:
        None
    """
    global server_process

    if not server_process:
        raise MlflowException(
            "There is no currently running gateway server", error_code=BAD_REQUEST
        )

    os.kill(server_process.pid, signal.SIGTERM)
    server_process = None


def update_server(config_path: str):
    """
    Updates the server with the new configuration.

    This function stops the currently running server and starts it again with the new configuration.
    It raises an exception if there is no currently running server.

    Args:
        config_path (str): Path to the new configuration file.

    Returns:
        None

    Raises:
        mlflow.exceptions.MlflowException: If there is no currently running server.
    """
    global server_process
    global gateway_host
    global gateway_port

    if server_process is None:
        raise MlflowException(
            "Unable to update server configuration. There is no currently running gateway server.",
            error_code=BAD_REQUEST,
        )

    _stop_server()

    time.sleep(2)

    start_server(config_path, gateway_host, gateway_port)
