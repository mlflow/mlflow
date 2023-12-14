import click

from mlflow.deployments.server.runner import run_app
from mlflow.environment_variables import MLFLOW_GATEWAY_CONFIG
from mlflow.gateway.config import _validate_config
from mlflow.gateway.utils import gateway_deprecated


def validate_config_path(_ctx, _param, value):
    try:
        _validate_config(value)
        return value
    except Exception as e:
        raise click.BadParameter(str(e))


@click.group("gateway", help="Manage the MLflow Gateway service")
def commands():
    pass


@commands.command("start", help="Start the MLflow Gateway service")
@click.option(
    "--config-path",
    envvar=MLFLOW_GATEWAY_CONFIG.name,
    callback=validate_config_path,
    required=True,
    help="The path to the gateway configuration file.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="The network address to listen on (default: 127.0.0.1).",
)
@click.option(
    "--port",
    default=5000,
    help="The port to listen on (default: 5000).",
)
@click.option(
    "--workers",
    default=2,
    help="The number of workers.",
)
@gateway_deprecated
def start(config_path: str, host: str, port: str, workers: int):
    run_app(config_path=config_path, host=host, port=port, workers=workers)
