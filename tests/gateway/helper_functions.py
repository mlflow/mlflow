from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Union

import requests
import yaml

import mlflow.gateway.utils
from mlflow.gateway.utils import kill_child_processes
from tests.helper_functions import get_safe_port


def reset_gateway_uri():
    # Reset the state of the global gateway_uri during teardown of the Gateway instance
    mlflow.gateway.utils._gateway_uri = None


class Gateway:
    def __init__(self, config_path: Union[str, Path], *args, **kwargs):
        self.port = get_safe_port()
        self.host = "localhost"
        self.url = f"http://{self.host}:{self.port}"
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
                "2",
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
        reset_gateway_uri()


def store_conf(path, conf):
    path.write_text(yaml.safe_dump(conf))


def wait():
    """
    A sleep statement for testing purposes only to ensure that the file watch and app reload
    has enough time to resolve to updated endpoints.
    """
    time.sleep(2)
