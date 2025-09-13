import contextlib
import logging
import os
import socket
import sys
import time
from subprocess import Popen
from threading import Thread
from typing import Any, Generator, Literal

import requests
import uvicorn
from fastapi import FastAPI

import mlflow
from mlflow.server import ARTIFACT_ROOT_ENV_VAR, BACKEND_STORE_URI_ENV_VAR

from tests.helper_functions import LOCALHOST, get_safe_port

_logger = logging.getLogger(__name__)


def _await_server_up_or_die(port: int, timeout: int = 30) -> None:
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


@contextlib.contextmanager
def _init_server(
    backend_uri: str,
    root_artifact_uri: str,
    extra_env: dict[str, Any] | None = None,
    app: str | None = None,
    server_type: Literal["flask", "fastapi"] = "fastapi",
) -> Generator[str, None, None]:
    """
    Launch a new REST server using the tracking store specified by backend_uri and root artifact
    directory specified by root_artifact_uri.

    Args:
        backend_uri: Backend store URI for the server
        root_artifact_uri: Root artifact URI for the server
        extra_env: Additional environment variables
        app: Application module path (defaults based on server_type if None)
        server_type: Server type to use - "fastapi" (default) or "flask"

    Yields:
        The string URL of the server.
    """
    mlflow.set_tracking_uri(None)
    server_port = get_safe_port()

    if server_type == "fastapi":
        # Use uvicorn for FastAPI
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            app or "mlflow.server.fastapi_app:app",
            "--host",
            LOCALHOST,
            "--port",
            str(server_port),
        ]
    else:
        # Default to Flask
        cmd = [
            sys.executable,
            "-m",
            "flask",
            "--app",
            app or "mlflow.server:app",
            "run",
            "--host",
            LOCALHOST,
            "--port",
            str(server_port),
        ]

    with Popen(
        cmd,
        env={
            **os.environ,
            BACKEND_STORE_URI_ENV_VAR: backend_uri,
            ARTIFACT_ROOT_ENV_VAR: root_artifact_uri,
            **(extra_env or {}),
        },
    ) as proc:
        try:
            _await_server_up_or_die(server_port)
            url = f"http://{LOCALHOST}:{server_port}"
            _logger.info(
                f"Launching tracking server on {url} with backend URI {backend_uri} and "
                f"artifact root {root_artifact_uri}"
            )
            yield url
        finally:
            proc.terminate()


def _send_rest_tracking_post_request(tracking_server_uri, api_path, json_payload, auth=None):
    """
    Make a POST request to the specified MLflow Tracking API and retrieve the
    corresponding `requests.Response` object
    """
    import requests

    url = tracking_server_uri + api_path
    return requests.post(url, json=json_payload, auth=auth)


class ServerThread(Thread):
    """Run a FastAPI/uvicorn app in a background thread, usable as a context manager."""

    def __init__(self, app: FastAPI, port: int):
        super().__init__(name="mlflow-tracking-server", daemon=True)
        self.host = "127.0.0.1"
        self.port = port
        self.url = f"http://{self.host}:{port}"
        self.health_url = f"{self.url}/health"
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="error")
        self.server = uvicorn.Server(config)

    def run(self) -> None:
        """Thread target: let Uvicorn manage its own event loop."""
        self.server.run()

    def shutdown(self) -> None:
        """Ask Uvicorn to exit; the serving loop checks this flag."""
        self.server.should_exit = True

    def __enter__(self) -> str:
        """Use as a context manager for tests or short-lived runs."""
        self.start()

        # Quick readiness wait (poll the health endpoint if available)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                r = requests.get(self.health_url, timeout=0.2)
                if r.ok:
                    break
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(0.1)
        return self.url

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        """Clean up resources when exiting context."""
        self.shutdown()
        # Give the server a moment to wind down
        self.join(timeout=5.0)
        return None
