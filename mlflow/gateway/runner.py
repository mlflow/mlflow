import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Union
from watchfiles import watch

from mlflow.gateway import app
from mlflow.gateway.handlers import _load_route_config
from mlflow.gateway.utils import kill_child_processes


_logger = logging.getLogger(__name__)


def _monitor_config(config_path: str):
    config_path = Path(config_path)
    prev_config = config_path.read_text()
    for changes in watch(
        config_path,
    ):
        if not any((path == str(config_path)) for _, path in changes):
            continue

        if not config_path.exists():
            _logger.warning("Configuration file does not exist")
            continue

        cfg = config_path.read_text()

        try:
            _load_route_config(config_path)
            load_successful = True
        except Exception as e:
            _logger.warning("Invalid configuration: %s", e)
            load_successful = False

        if cfg == prev_config:
            continue
        if load_successful:
            prev_config = cfg

        yield load_successful


def run_app(config_path: str, host: str, port: Union[int, str], workers: int = 4):
    """
    Execute uvicorn servers as subprocesses to the gunicorn process manager.

    While running, a polling file watcher will execute to detect changes in
    the route configuration yaml file. If changes are detected, the subprocesses (the uvicorn
    server instances) will restart, loading a new FastAPI app configuration for the updated
    routes in the replaced configuration yaml file.
    """
    config_path = os.path.abspath(os.path.normpath(os.path.expanduser(config_path)))
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gunicorn",
            "-b",
            f"{host}:{port}",
            "--workers",
            str(workers),
            "--worker-class",
            "uvicorn.workers.UvicornWorker",
            f"{app.__name__}:create_app_from_env()",
            "--reload",
            "--log-level",
            "debug",
        ],
        env={
            **os.environ,
            app.MLFLOW_GATEWAY_CONFIG: config_path,
        },
    ) as proc:
        for load_successful in _monitor_config(config_path):
            if load_successful:
                _logger.info("Configuration updated, reloading workers")
                kill_child_processes(proc.pid)
