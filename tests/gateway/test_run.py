import subprocess
import time
import requests
import psutil
import sys
from pathlib import Path

from tests.helper_functions import get_safe_port


class Gateway:
    def __init__(self, config_path: str, *args, **kwargs):
        self.port = get_safe_port()
        self.host = "localhost"
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
                if self.request("health").ok:
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(0.5)

        raise Exception("Gateway failed to start")

    def request(self, path: str) -> requests.Response:
        return requests.get(f"http://{self.host}:{self.port}/{path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        parent = psutil.Process(self.process.pid)
        for child in parent.children(recursive=True):
            child.kill()
        self.process.terminate()
        self.process.wait()


def test_run_app(tmp_path: Path):
    config = tmp_path.joinpath("config.yml")
    config.write_text(
        """
routes:
    - a
    - b
"""
    )
    with Gateway(config) as gateway:
        for char in "ab":
            response = gateway.request(char)
            assert response.status_code == 200
            assert response.json() == {"message": char}

        # Append a new route
        with config.open("a") as f:
            f.write("    - c\n")
        time.sleep(2.5)
        for char in "abc":
            response = gateway.request(char)
            assert response.status_code == 200
            assert response.json() == {"message": char}

        # Invalid config
        config.write_text(
            """
routes:
    - a
    -
"""
        )
        time.sleep(2.5)
        for char in "abc":
            response = gateway.request(char)
            assert response.status_code == 200
            assert response.json() == {"message": char}

        # Valid config
        config.write_text(
            """
routes:
    - x
    - y
"""
        )
        time.sleep(2.5)
        for char in "xy":
            response = gateway.request(char)
            assert response.status_code == 200
            assert response.json() == {"message": char}

        for char in "abc":
            response = gateway.request(char)
            assert response.status_code == 404

        # Delete config
        config.unlink()
        time.sleep(2.5)
        for char in "xy":
            response = gateway.request(char)
            assert response.status_code == 200
            assert response.json() == {"message": char}
        time.sleep(2.5)

        # Recreate config
        config.write_text(
            """
routes:
    - z
"""
        )
        time.sleep(2.5)
        char = "z"
        response = gateway.request(char)
        assert response.status_code == 200
        assert response.json() == {"message": char}
