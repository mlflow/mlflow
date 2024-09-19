import logging
import os
import subprocess
import sys
from typing import Generator

from watchfiles import watch

from mlflow.environment_variables import MLFLOW_GATEWAY_CONFIG
from mlflow.gateway import app
from mlflow.gateway.config import _load_route_config
from mlflow.gateway.utils import kill_child_processes

_logger = logging.getLogger(__name__)


def monitor_config(config_path: str) -> Generator[None, None, None]:
    with open(config_path) as f:
        prev_config = f.read()

    for changes in watch(os.path.dirname(config_path)):
        if not any((path == config_path) for _, path in changes):
            continue

        if not os.path.exists(config_path):
            _logger.warning(f"{config_path} deleted")
            continue

        with open(config_path) as f:
            config = f.read()
        if config == prev_config:
            continue

        try:
            _load_route_config(config_path)
        except Exception as e:
            _logger.warning("Invalid configuration: %s", e)
            continue
        else:
            prev_config = config

        yield


class Runner:
    def __init__(
        self,
        config_path: str,
        host: str,
        port: int,
        workers: int,
    ) -> None:
        self.config_path = config_path
        self.host = host
        self.port = port
        self.workers = workers
        self.process = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gunicorn",
                "--bind",
                f"{self.host}:{self.port}",
                "--workers",
                str(self.workers),
                "--worker-class",
                "uvicorn.workers.UvicornWorker",
                f"{app.__name__}:create_app_from_env()",
            ],
            env={
                **os.environ,
                MLFLOW_GATEWAY_CONFIG.name: self.config_path,
            },
        )

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def reload(self) -> None:
        kill_child_processes(self.process.pid)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def run_app(config_path: str, host: str, port: int, workers: int) -> None:
    config_path = os.path.abspath(os.path.normpath(os.path.expanduser(config_path)))
    with Runner(
        config_path=config_path,
        host=host,
        port=port,
        workers=workers,
    ) as runner:
        for _ in monitor_config(config_path):
            _logger.info("Configuration updated, reloading workers")
            runner.reload()
