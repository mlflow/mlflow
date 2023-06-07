import os
import psutil
import signal
import subprocess
import time
from watchfiles import watch
from app import CONFIG_ENV_VAR
from pydantic import BaseModel
import click
import yaml
import logging
import sys
from typing import List

logger = logging.getLogger(__name__)


class Config(BaseModel):
    routes: List[str]


def find_child_processes(parent_pid):
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return []
    return [c.pid for c in parent.children(recursive=True)]


def is_valid(config_path):
    try:
        with open(config_path) as f:
            Config(**yaml.safe_load(f))
            return True
    except Exception:
        return False


@click.command()
@click.option("--config-path", default="config.yml", help="Path to config file")
def start(config_path):
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gunicorn",
            "--workers",
            "4",
            "--worker-class",
            "uvicorn.workers.UvicornWorker",
            "app:create_app()",
            "--reload",
        ],
        env={
            **os.environ,
            CONFIG_ENV_VAR: config_path,
        },
    ) as proc:
        for _ in watch(config_path):
            logger.warning("Config file updated")
            if not is_valid(config_path):
                logger.warning("Invalid config")
                continue
            logger.warning("Valid config, restarting workers")
            for child in find_child_processes(proc.pid):
                os.kill(child, signal.SIGTERM)
                time.sleep(1)


if __name__ == "__main__":
    start()
