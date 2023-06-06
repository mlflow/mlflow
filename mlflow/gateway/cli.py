import click


@click.command("start-gateway", help="Start the MLflow Model Gateway service.")
@click.option(
    "--config-path",
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
def start_gateway(config_path: str, host: str, port: str):
    # TODO: Implement this command
    click.echo("Starting gateway...")
