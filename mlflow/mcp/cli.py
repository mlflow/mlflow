import click

from mlflow.mcp.server import run_server


@click.group("mcp", help="Command Line Interface for the MLflow MCP server")
def cli():
    """
    Command Line Interface for the MLflow MCP server.
    """


@cli.command(help="Run the MLflow MCP server")
def run():
    run_server()
