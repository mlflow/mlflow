import click

from mlflow.mcp.server import run_server
from mlflow.telemetry.events import McpRunEvent
from mlflow.telemetry.track import record_usage_event


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
@record_usage_event(McpRunEvent)
def run():
    run_server()
