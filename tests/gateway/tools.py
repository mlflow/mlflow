from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Union, Dict

import requests
import yaml

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
