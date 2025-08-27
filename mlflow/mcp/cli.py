import click

from mlflow.mcp.server import run_server


@click.group(
    "mcp",
    help=(
        "Model Context Protocol (MCP) server for MLflow. "
        "MCP enables LLM applications to interact with MLflow traces programmatically."
    ),
)
def cli():
    """
    Model Context Protocol (MCP) server for MLflow.

    MCP enables LLM applications and coding assistants to interact with MLflow traces
    programmatically. Use this to expose MLflow trace data to AI tools.
    """


@cli.command(
    help=(
        "Run the MLflow MCP server. "
        "This starts a server that exposes MLflow trace operations to MCP-compatible clients "
        "like Claude Desktop or other AI assistants."
    )
)
def run():
    run_server()
