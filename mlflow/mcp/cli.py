import click

from mlflow.mcp.server import create_mcp


@click.group("mcp", help="Command Line Interface for the MLflow MCP server")
def cli():
    """
    Command Line Interface for the MLflow MCP server.
    """


@cli.command(help="Run the MLflow MCP server")
def run():
    # TODO: Add telemetry here
    mcp = create_mcp()
    mcp.run(show_banner=False)
