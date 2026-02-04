import click

from mlflow.environment_variables import MLFLOW_GATEWAY_CONFIG
from mlflow.gateway.config import _validate_config
from mlflow.gateway.runner import run_app
from mlflow.telemetry.events import GatewayStartEvent
from mlflow.telemetry.track import record_usage_event
from mlflow.utils.os import is_windows


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
@record_usage_event(GatewayStartEvent)
def start(config_path: str, host: str, port: str, workers: int):
    if is_windows():
        raise click.ClickException("MLflow AI Gateway does not support Windows.")
    run_app(config_path=config_path, host=host, port=port, workers=workers)
