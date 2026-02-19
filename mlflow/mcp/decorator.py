"""
Decorator for exposing MLflow CLI commands as MCP tools.

Usage:
    from mlflow.mcp.decorator import mlflow_mcp

    @commands.command("search")
    @mlflow_mcp(tool_name="search_traces")
    @click.option(...)
    def search_traces(...):
        ...

The decorator attaches MCP metadata to the Click command, which is then
used by the MCP server to register the tool with the specified name.
"""

from typing import Callable, TypeVar

import click

# Attribute name used to store MCP metadata on Click commands
MCP_METADATA_ATTR = "_mlflow_mcp_metadata"

F = TypeVar("F", bound=Callable)


def mlflow_mcp(tool_name: str) -> Callable[[F], F]:
    """
    Decorator to expose a Click command as an MCP tool with a curated name.

    Args:
        tool_name: The name to use for the MCP tool. This should be a clear,
            agent-friendly name that describes what the tool does.
            Convention: action_entity (e.g., "search_traces", "get_experiment")

    Example:
        @commands.command("search")
        @mlflow_mcp(tool_name="search_traces")
        def search(...):
            '''Search for traces in the specified experiment.'''
            ...

    The decorator stores metadata on the function that the MCP server reads
    when registering tools. Commands without this decorator are not exposed
    as MCP tools.
    """

    def decorator(fn: F) -> F:
        # Store MCP metadata on the function
        setattr(fn, MCP_METADATA_ATTR, {"tool_name": tool_name})
        return fn

    return decorator


def get_mcp_tool_name(cmd: click.Command) -> str | None:
    """
    Get the MCP tool name from a Click command, if it has been decorated.

    Args:
        cmd: The Click command to check.

    Returns:
        The MCP tool name if the command has been decorated with @mlflow_mcp,
        None otherwise.
    """
    if cmd.callback is None:
        return None

    metadata = getattr(cmd.callback, MCP_METADATA_ATTR, None)
    if metadata is None:
        return None

    return metadata.get("tool_name")
