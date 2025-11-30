"""
Runs MLflow server, gateway, and UI in development mode.
"""

import os
import socket
import subprocess
import sys
import time


def random_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    gateway_port = random_port()
    gateway_host = "localhost"
    with (
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "gateway",
                "start",
                "--config-path",
                "examples/gateway/openai/config.yaml",
                "--host",
                gateway_host,
                "--port",
                str(gateway_port),
            ]
        ) as gateway,
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "--dev",
            ],
            env={
                **os.environ,
                "MLFLOW_DEPLOYMENTS_TARGET": f"http://{gateway_host}:{gateway_port}",
            },
        ) as server,
        subprocess.Popen(
            [
                "yarn",
                "start",
            ],
            cwd="mlflow/server/js",
        ) as ui,
    ):
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                gateway.terminate()
                server.terminate()
                ui.terminate()
                break


if __name__:
    main()
