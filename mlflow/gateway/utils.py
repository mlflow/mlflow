import errno
import logging
import os
from pathlib import Path
import psutil
import re
import requests
import time
import yaml

from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import (
    MAX_WAIT_TIME_SECONDS,
    BASE_WAIT_TIME_SECONDS,
    GATEWAY_SERVER_STATE_FILE,
)
from mlflow.protos.databricks_pb2 import BAD_REQUEST, INVALID_PARAMETER_VALUE


_logger = logging.getLogger(__name__)


def _parse_url_path_for_base_url(url_string):
    split_url = url_string.split("/")
    return "/".join(split_url[:-1])


def is_valid_endpoint_name(name: str) -> bool:
    """
    Check whether a string contains any URL reserved characters, spaces, or characters other
    than alphanumeric, underscore, and hyphen.

    Returns True if the string doesn't contain any of these characters.
    """
    return bool(re.fullmatch(r"^[\w-]+", name))


def check_configuration_route_name_collisions(config):
    if len(config) < 2:
        return
    names = [route["name"] for route in config]
    if len(names) != len(set(names)):
        raise MlflowException(
            "Duplicate names found in route configurations. Please remove the duplicate route "
            "name from the configuration to ensure that route endpoints are created properly.",
            error_code=INVALID_PARAMETER_VALUE,
        )


def check_server_status(url):
    """
    Check the server status.
    This function sends an HTTP GET request to the server and considers it to be
    running if it responds with a status code of 200.
    """
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def wait_until_server_starts(
    url,
    server_process,
    logger,
    max_wait_time=MAX_WAIT_TIME_SECONDS,
    base_wait_time=BASE_WAIT_TIME_SECONDS,
):
    """
    Waits until the server starts using an exponential backoff strategy.
    """
    wait_time = base_wait_time
    total_wait_time = 0

    while total_wait_time < max_wait_time:
        if check_server_status(url):
            logger.info("Gateway server is online and is ready to receive requests...")
            return server_process
        else:
            if server_process.poll() is not None:
                stderr_output = server_process.stderr.read().decode()
                logger.error("The gateway server has encountered an error and has terminated:")
                for stderr_line in stderr_output.split("\n"):
                    logger.error(stderr_line)
                raise Exception("The server process has terminated.")
            time.sleep(wait_time)
            total_wait_time += wait_time
            wait_time *= 2

    raise MlflowException("Server didn't start within the expected time.", error_code=BAD_REQUEST)


def is_pid_alive(pid):
    """
    Determine if a given process id is still running on the system
    """
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            # NB: ESRCH is the error code for a pid that doesn't exist
            return False
        elif err.errno == errno.EPERM:
            # NB: EPERM indicates an existing pid that has a locked permission state
            return True
        else:
            raise MlflowException(
                f"The server has entered an unknown state. Process id {pid} is in an "
                "indeterminate state.",
                error_code=BAD_REQUEST,
            )
    else:
        return True


def kill_parent_and_child_processes(parent_pid, server_state_file):
    """
    Gracefully terminate all processes associated with the server process if possible, else
    kill them if too much time has elapsed.
    """
    parent = psutil.Process(parent_pid)
    for child in parent.children(recursive=True):
        child.terminate()
    _, still_alive = psutil.wait_procs(parent.children(), timeout=3)
    for p in still_alive:
        p.kill()
    parent.terminate()
    try:
        parent.wait(timeout=5)
    except psutil.TimeoutExpired:
        parent.kill()
    # Remove the server state file when the server is terminated / killed so that we don't have
    # hanging references for shell-killed (manually terminated) processes
    _delete_server_state(server_state_file)


def write_server_state(host: str, port: int, pid: int, server_state_file: str):
    """
    Write the server state information to disk to ensure that subsequent calls to the server
    process (i.e., update) will access the appropriate parent process and preserve host and
    port information for the running server.
    """
    state_path = Path(server_state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    state_data = {"host": host, "port": port, "pid": pid}

    state_path.write_text(yaml.safe_dump(state_data))


def read_server_state(server_state_file: str):
    """
    Read the server state information from disk for server update commands to ensure that the
    server management stops the appropriate process id and restarts the server with the correct
    host and port information
    """
    state_path = Path(server_state_file)
    if state_path.exists():
        return yaml.safe_load(state_path.read_text())


def _delete_server_state(file: str = GATEWAY_SERVER_STATE_FILE):
    """
    Internal utility for deleting the server state file whether it exists or not (cleanup for
    errors in updating the server)
    """
    state_path = Path(file)

    state_path.unlink(missing_ok=True)
