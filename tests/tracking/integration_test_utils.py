from subprocess import Popen

from unittest import mock
import os
from threading import Thread

import logging
import socket
import time
import tempfile

import mlflow
from mlflow.server import BACKEND_STORE_URI_ENV_VAR, ARTIFACT_ROOT_ENV_VAR
from mlflow.utils.file_utils import path_to_local_file_uri, local_file_uri_to_path
from tests.helper_functions import LOCALHOST, get_safe_port

_logger = logging.getLogger(__name__)


def _await_server_up_or_die(port, timeout=60):
    """Waits until the local flask server is listening on the given port."""
    _logger.info(f"Awaiting server to be up on {LOCALHOST}:{port}")
    start_time = time.time()
    connected = False
    while not connected and time.time() - start_time < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((LOCALHOST, port))
        if result == 0:
            connected = True
        else:
            _logger.info("Server not yet up, waiting...")
            time.sleep(0.5)
    if not connected:
        raise Exception("Failed to connect on %s:%s after %s seconds" % (LOCALHOST, port, timeout))
    _logger.info(f"Server is up on {LOCALHOST}:{port}!")


# NB: We explicitly wait and timeout on server shutdown in order to ensure that pytest output
# reveals the cause in the event of a test hang due to the subprocess not exiting.
def _await_server_down_or_die(process, timeout=60):
    """Waits until the local flask server process is terminated."""
    _logger.info("Awaiting termination of server process...")
    start_time = time.time()

    def wait():
        process.wait()

    Thread(target=wait).start()
    while process.returncode is None and time.time() - start_time < timeout:
        time.sleep(0.5)
    if process.returncode is None:
        raise Exception("Server failed to shutdown after %s seconds" % timeout)


def _init_server(backend_uri, root_artifact_uri, additional_env_variables = dict()):
    """
    Launch a new REST server using the tracking store specified by backend_uri and root artifact
    directory specified by root_artifact_uri.
    :returns A tuple (url, process) containing the string URL of the server and a handle to the
             server process (a multiprocessing.Process object).
    """
    mlflow.set_tracking_uri(None)
    server_port = get_safe_port()
    env = {
        BACKEND_STORE_URI_ENV_VAR: backend_uri,
        ARTIFACT_ROOT_ENV_VAR: path_to_local_file_uri(
            tempfile.mkdtemp(dir=local_file_uri_to_path(root_artifact_uri))
        ),
        **additional_env_variables
    }
    with mock.patch.dict(os.environ, env):
        cmd = [
            "python",
            "-c",
            'from mlflow.server import app; app.run("{hostname}", {port})'.format(
                hostname=LOCALHOST, port=server_port
            ),
        ]
        process = Popen(cmd)

    _await_server_up_or_die(server_port)
    url = "http://{hostname}:{port}".format(hostname=LOCALHOST, port=server_port)
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
