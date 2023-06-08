import logging
import os
import pathlib
import sys

from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import CONF_PATH_ENV_VAR, GATEWAY_SERVER_STATE_FILE
from mlflow.gateway.utils import (
    wait_until_server_starts,
    is_pid_alive,
    write_server_state,
    read_server_state,
    kill_parent_and_child_processes,
)
from mlflow.protos.databricks_pb2 import BAD_REQUEST

import subprocess


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


def _start_server(config_path: str, host, port, server_state_file: str = GATEWAY_SERVER_STATE_FILE):
    """
    Starts the server with the given configuration, host and port.

    This function sets the global server_process, gateway_host, and gateway_port variables.
    It starts the server by creating a new subprocess that runs the server script.

    Args:
        config_path (str): Path to the configuration file.
        host (str): The hostname to use for the server.
        port (int): The port number to use for the server.
        server_state_file (str): The location on the local file system to store the server state

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the server subprocess fails to start.
    """
    server_state = read_server_state(server_state_file)

    if server_state and is_pid_alive(server_state["pid"]):
        raise MlflowException(
            f"There is a currently running server instance at pid '{server_state['pid']}'. Please "
            "terminate the running server instance prior to starting a new instance within this "
            "context.",
            error_code=BAD_REQUEST,
        )

    os.environ[CONF_PATH_ENV_VAR] = config_path

    cmd_path = pathlib.Path(__file__).parent.joinpath("gateway_app.py")
    server_cmd = [sys.executable, str(cmd_path), "--host", host, "--port", str(port)]

    server_process = subprocess.Popen(server_cmd)

    url = f"http://{host}:{port}/health"
    wait_until_server_starts(url=url, server_process=server_process, logger=_logger)

    # Write the server state to disk
    write_server_state(
        host=host, port=port, pid=server_process.pid, server_state_file=server_state_file
    )

    return server_process


def _stop_server(server_pid: int, server_state_file: str):
    """
    Stops the currently running server.

    This function kills the process of the running server and resets the global server_process
    variable to None.

    Args:
        server_pid (int): The process id of a server instance main process
        server_state_file (str): The location on the local file system to store the server state

    Returns:
        None
    """

    if is_pid_alive(server_pid):
        kill_parent_and_child_processes(server_pid, server_state_file)
        _logger.info("The Gateway server has been terminated.")
    else:
        raise MlflowException(
            "There is no currently running gateway server", error_code=BAD_REQUEST
        )


def _update_server(config_path: str, server_state_file: str = GATEWAY_SERVER_STATE_FILE):
    """
    Updates the server with the new configuration.

    This function stops the currently running server and starts it again with the new configuration.
    It raises an exception if there is no currently running server.

    Args:
        config_path (str): Path to the new configuration file.
        server_state_file (str): The location on the local file system to store the server state

    Returns:
        None

    Raises:
        mlflow.exceptions.MlflowException: If there is no currently running server.
    """
    server_state = read_server_state(server_state_file)

    if not server_state:
        raise MlflowException(
            f"There is no server_state_file present at {server_state_file}. Verify that the "
            "server has been started",
            error_code=BAD_REQUEST,
        )

    server_pid = server_state["pid"]

    if not is_pid_alive(server_pid):
        raise MlflowException(
            "Unable to update server configuration. There is no server currently running at "
            f"process id {server_pid}",
            error_code=BAD_REQUEST,
        )

    _stop_server(server_pid, server_state_file)

    return _start_server(config_path, server_state["host"], server_state["port"], server_state_file)
