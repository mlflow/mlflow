import click


@click.group("gateway", help="Manage MLflow Model Gateway")
def commands():
    pass


@commands.command("start", help="Start the MLflow Model Gateway service")
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
def start(config_path: str, host: str, port: str):
    # TODO: Implement this command
    click.echo("Starting gateway...")


@commands.command("update", help="Update the MLflow Model Gateway service")
def update():
    # TODO: Implement this command
    click.echo("Updating gateway...")
