import os
import signal
import subprocess
import time
import logging
import sys
from pathlib import Path

import psutil
from . import app
from .config import _validate_config
from watchfiles import watch

_logger = logging.getLogger(__name__)


def _find_child_processes(parent_pid):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return []
    return [c.pid for c in parent.children(recursive=True)]


def monitor_config(config_path: str):
    config_path = Path(config_path)
    prev_config = config_path.read_text()
    for changes in watch(
        # Watch the directory containing the config file to detect file recreation
        os.path.dirname(config_path),
    ):
        if not any((path == str(config_path)) for _, path in changes):
            continue

        if not os.path.exists(config_path):
            _logger.warning("Configuration file does not exist")
            continue

        # Have the file contents changed?
        cfg = config_path.read_text()
        if cfg == prev_config:
            continue
        prev_config = cfg

        try:
            _validate_config(config_path)
        except Exception as e:
            _logger.warning("Invalid configuration: %s", e)
            continue
        yield


def run_app(config_path: str, host: str, port: int, workers: int = 4):
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
        for _ in monitor_config(config_path):
            _logger.info("Configuration updated, reloading workers")
            for child in _find_child_processes(proc.pid):
                os.kill(child, signal.SIGTERM)
                time.sleep(1)
