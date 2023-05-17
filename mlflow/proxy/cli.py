import click
import os
import subprocess
import sys
import pathlib

from mlflow.utils import cli_args


@click.group("proxy-server")
def commands():
    pass


@commands.command("start-proxy")
@click.option(
    "--config",
    "-c",
    default=None,
    help="The path to the configuration file that defines the serving endpoint definitions",
)
@cli_args.HOST
@cli_args.PORT
def start_proxy(config, host, port):
    os.environ["MLFLOW_PROXY_CONFIG_PATH"] = config

    cmd_path = pathlib.Path.cwd().joinpath("server.py")

    cmd = [sys.executable, str(cmd_path), "--host", host, "--port", str(port)]

    subprocess.run(cmd)
