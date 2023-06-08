import click
from .run import run
from .config import _validate_config


def validate_config_path(_ctx, _param, value):
    try:
        _validate_config(value)
        return value
    except Exception as e:
        raise click.BadParameter(str(e))


@click.group("gateway", help="Manage MLflow Model Gateway")
def commands():
    pass


@commands.command("start", help="Start the MLflow Model Gateway service")
@click.option(
    "--config-path",
    callback=validate_config_path,
    required=True,
    help="The path to the gateway configuration file.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="The network address to listen on.",
)
@click.option(
    "--port",
    default=5000,
    show_default=True,
    help="The port to listen on.",
)
@click.option(
    "--workers",
    default=2,
    show_default=True,
    help="The number of workers.",
)
def start(config_path: str, host: str, port: str, workers: int):
    run(config_path=config_path, host=host, port=port, workers=workers)
