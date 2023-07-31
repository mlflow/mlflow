import asyncio
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Union, Dict
from unittest import mock
import uvicorn

import requests
import yaml

from mlflow.gateway import app
from mlflow.gateway.utils import kill_child_processes
from tests.helper_functions import get_safe_port


class Gateway:
    def __init__(self, config_path: Union[str, Path], *args, **kwargs):
        self.port = get_safe_port()
        self.host = "localhost"
        self.url = f"http://{self.host}:{self.port}"
        self.workers = 2
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "gateway",
                "start",
                "--config-path",
                config_path,
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--workers",
                str(self.workers),
            ],
            *args,
            **kwargs,
        )
        self.wait_until_ready()

    def wait_until_ready(self) -> None:
        s = time.time()
        while time.time() - s < 10:
            try:
                if self.get("health").ok:
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)

        raise Exception("Gateway failed to start")

    def wait_reload(self) -> None:
        """
        Should be called after we update a gateway config file in tests to ensure
        that the gateway service has reloaded the config.
        """
        time.sleep(self.workers)

    def request(self, method: str, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return requests.request(method, f"{self.url}/{path}", *args, **kwargs)

    def get(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.request("GET", path, *args, **kwargs)

    def assert_health(self):
        assert self.get("health").ok

    def post(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.request("POST", path, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        kill_child_processes(self.process.pid)
        self.process.terminate()
        self.process.wait()


def save_yaml(path, conf):
    path.write_text(yaml.safe_dump(conf))


class MockAsyncResponse:
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def raise_for_status(self) -> None:
        pass

    async def json(self) -> Dict[str, Any]:
        return self.data

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        pass


class MockHttpClient(mock.Mock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return


def mock_http_client(mock_response: MockAsyncResponse):
    mock_http_client = MockHttpClient()
    mock_http_client.post = mock.Mock(return_value=mock_response)
    return mock_http_client


class UvicornGateway:
    # This test utility class is used to validate the internal functionality of the
    # AI Gateway within-process so that the provider endpoints can be mocked,
    # allowing a nearly end-to-end validation of the entire AI Gateway stack.
    # NB: this implementation should only be used for integration testing. Unit tests that
    # require validation of the AI Gateway server should use the `Gateway` implementation in
    # this module which executes the uvicorn server through gunicorn as a process manager.
    def __init__(self, config_path: Union[str, Path], *args, **kwargs):
        self.port = get_safe_port()
        self.host = "localhost"
        self.url = f"http://{self.host}:{self.port}"
        self.config_path = config_path
        self.server = None
        self.loop = None
        self.thread = None
        self.stop_event = threading.Event()

    def start_server(self):
        uvicorn_app = app.create_app_from_path(self.config_path)

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        config = uvicorn.Config(
            app=uvicorn_app,
            host=self.host,
            port=self.port,
            lifespan="on",
            loop="auto",
            log_level="info",
        )
        self.server = uvicorn.Server(config)

        def run():
            self.loop.run_until_complete(self.server.serve())

        self.thread = threading.Thread(target=run)
        self.thread.start()

    def request(self, method: str, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return requests.request(method, f"{self.url}/{path}", *args, **kwargs)

    def get(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.request("GET", path, *args, **kwargs)

    def assert_health(self):
        assert self.get("health").ok

    def post(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.request("POST", path, *args, **kwargs)

    def stop(self):
        if self.server is not None:
            self.server.should_exit = True  # Instruct the uvicorn server to stop
            self.stop_event.wait()  # Wait for the server to actually stop
            self.thread.join()  # block until thread termination
            self.server = None
            self.loop = None
            self.thread = None

    def __enter__(self):
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop the server and the thread
        if self.server is not None:
            self.server.should_exit = True
        self.thread.join()
