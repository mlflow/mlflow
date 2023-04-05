from subprocess import Popen

import sys
import os
import logging
import socket
import time

import mlflow
from mlflow.server import BACKEND_STORE_URI_ENV_VAR, ARTIFACT_ROOT_ENV_VAR
from tests.helper_functions import LOCALHOST, get_safe_port

_logger = logging.getLogger(__name__)


def _await_server_up_or_die(port, timeout=20):
    """Waits until the local flask server is listening on the given port."""
    _logger.info(f"Awaiting server to be up on {LOCALHOST}:{port}")
    start_time = time.time()
    while time.time() - start_time < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            if sock.connect_ex((LOCALHOST, port)) == 0:
                _logger.info(f"Server is up on {LOCALHOST}:{port}!")
                break
        _logger.info("Server not yet up, waiting...")
        time.sleep(0.5)
    else:
        raise Exception(f"Failed to connect on {LOCALHOST}:{port} within {timeout} seconds")


# NB: We explicitly wait and timeout on server shutdown in order to ensure that pytest output
# reveals the cause in the event of a test hang due to the subprocess not exiting.
def _terminate_server(process, timeout=10):
    """Waits until the local flask server process is terminated."""
    _logger.info("Terminating server...")
    process.terminate()
    process.wait(timeout=timeout)


def _init_server(backend_uri, root_artifact_uri, extra_env=None):
    """
    Launch a new REST server using the tracking store specified by backend_uri and root artifact
    directory specified by root_artifact_uri.
    :returns A tuple (url, process) containing the string URL of the server and a handle to the
             server process (a multiprocessing.Process object).
    """
    mlflow.set_tracking_uri(None)
    server_port = get_safe_port()
    process = Popen(
        [
            sys.executable,
            "-c",
            f'from mlflow.server import app; app.run("{LOCALHOST}", {server_port})',
        ],
        env={
            **os.environ,
            BACKEND_STORE_URI_ENV_VAR: backend_uri,
            ARTIFACT_ROOT_ENV_VAR: root_artifact_uri,
            **(extra_env or {}),
        },
    )

    _await_server_up_or_die(server_port)
    url = f"http://{LOCALHOST}:{server_port}"
    _logger.info(f"Launching tracking server against backend URI {backend_uri}. Server URL: {url}")
    return url, process


def _send_rest_tracking_post_request(tracking_server_uri, api_path, json_payload):
    """
    Make a POST request to the specified MLflow Tracking API and retrieve the
    corresponding `requests.Response` object
    """
    import requests

    url = tracking_server_uri + api_path
    response = requests.post(url, json=json_payload)
    return response
