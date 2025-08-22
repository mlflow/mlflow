import contextlib
import io
from typing import Any, Callable

import click
from click.types import BOOL, FLOAT, INT, STRING, UUID
from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

from mlflow.cli.traces import commands as traces_cli


def param_type_to_json_schema_type(pt: click.ParamType) -> str:
    """
    Converts a Click ParamType to a JSON schema type.
    """
    if pt is STRING:
        return "string"
    if pt is BOOL:
        return "boolean"
    if pt is INT:
        return "integer"
    if pt is FLOAT:
        return "number"
    if pt is UUID:
        return "string"
    return "string"


def get_input_schema(params: list[click.Parameter]) -> dict[str, Any]:
    """
    Converts click params to JSON schema
    """
    res: dict[str, Any] = {}
    for p in params:
        schema = {
            "type": param_type_to_json_schema_type(p.type),
            "default": p.default,
            "required": p.required,
        }
        if description := getattr(p, "help", None):
            schema["description"] = description
        if isinstance(p.type, click.Choice):
            schema["enum"] = [str(choice) for choice in p.type.choices]

        res[p.name] = schema

    return res


def wrapper(command: click.Command) -> Callable[..., str]:
    def wrapper(**kwargs: Any) -> str:
        string_io = io.StringIO()
        with contextlib.redirect_stdout(string_io), contextlib.redirect_stderr(string_io):
            command.callback(**kwargs)
        return string_io.getvalue().strip()

    return wrapper


def cmd_to_function_tool(cmd: click.Command) -> FunctionTool:
    """
    Converts a Click command to a FunctionTool.
    """
    return FunctionTool(
        fn=wrapper(cmd),
        name=cmd.callback.__name__,
        description=cmd.help,
        parameters={
            "name": cmd.name,
            "description": cmd.help,
            "inputSchema": get_input_schema(cmd.params),
        },
    )


def create_mcp() -> FastMCP:
    return FastMCP(
        name=traces_cli.name,
        instructions=traces_cli.help,
        tools=[cmd_to_function_tool(cmd) for cmd in traces_cli.commands.values()],
    )


if __name__ == "__main__":
    mcp = create_mcp()
    mcp.run(show_banner=False)
