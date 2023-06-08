import os
import signal
import subprocess
import time
import logging
import sys
from typing import Generator, List

import psutil
from . import app
from .config import _validate_config
from watchfiles import watch

_logger = logging.getLogger(__name__)


def _find_child_processes(parent_pid: int) -> List[int]:
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return []
    return [c.pid for c in parent.children(recursive=True)]


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
        prev_config = config

        try:
            _validate_config(config_path)
        except Exception as e:
            _logger.warning("Invalid configuration: %s", e)
            continue

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
                "--reload",
                "--log-level",
                "debug",
            ],
            env={
                **os.environ,
                app.MLFLOW_GATEWAY_CONFIG: self.config_path,
            },
        )

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def reload(self) -> None:
        for child in _find_child_processes(self.process.pid):
            os.kill(child, signal.SIGTERM)
            time.sleep(1)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def run(config_path: str, host: str, port: int, workers: int) -> None:
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
