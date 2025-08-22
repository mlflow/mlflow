import click

from mlflow.mcp.server import create_mcp


@click.group("mcp", help="Command Line Interface for the MCP")
def cli():
    """
    Command Line Interface for the MCP
    """


@cli.command(help="Run the MCP server")
def run():
    # TODO: Add telemetry here
    mcp = create_mcp()
    mcp.run(show_banner=False)
