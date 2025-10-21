import contextlib
import io
from typing import TYPE_CHECKING, Any, Callable

import click
from click.types import BOOL, FLOAT, INT, STRING, UUID

from mlflow.ai_commands.ai_command_utils import get_command_body, list_commands
from mlflow.cli.scorers import commands as scorers_cli
from mlflow.cli.traces import commands as traces_cli

if TYPE_CHECKING:
    from fastmcp import FastMCP
    from fastmcp.tools import FunctionTool


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
    properties: dict[str, Any] = {}
    required: list[str] = []
    for p in params:
        schema = {
            "type": param_type_to_json_schema_type(p.type),
        }
        if p.default is not None:
            schema["default"] = p.default
        if isinstance(p, click.Option):
            schema["description"] = (p.help or "").strip()
        if isinstance(p.type, click.Choice):
            schema["enum"] = [str(choice) for choice in p.type.choices]
        if p.required:
            required.append(p.name)
        properties[p.name] = schema

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def fn_wrapper(command: click.Command) -> Callable[..., str]:
    def wrapper(**kwargs: Any) -> str:
        # Capture stdout and stderr
        string_io = io.StringIO()
        with (
            contextlib.redirect_stdout(string_io),
            contextlib.redirect_stderr(string_io),
        ):
            command.callback(**kwargs)
        return string_io.getvalue().strip()

    return wrapper


def cmd_to_function_tool(cmd: click.Command) -> "FunctionTool":
    """
    Converts a Click command to a FunctionTool.
    """
    from fastmcp.tools import FunctionTool

    return FunctionTool(
        fn=fn_wrapper(cmd),
        name=cmd.callback.__name__,
        description=(cmd.help or "").strip(),
        parameters=get_input_schema(cmd.params),
    )


def register_prompts(mcp: "FastMCP") -> None:
    """Register AI commands as MCP prompts."""
    for command in list_commands():
        # Convert slash-separated keys to underscores for MCP names
        mcp_name = command["key"].replace("/", "_")

        # Create a closure to capture the command key
        def make_prompt(cmd_key: str):
            @mcp.prompt(name=mcp_name, description=command["description"])
            def ai_command_prompt() -> str:
                """Execute an MLflow AI command prompt."""
                return get_command_body(cmd_key)

            return ai_command_prompt

        # Register the prompt
        make_prompt(command["key"])


def create_mcp() -> "FastMCP":
    from fastmcp import FastMCP

    tools = [
        *[cmd_to_function_tool(cmd) for cmd in traces_cli.commands.values()],
        *[cmd_to_function_tool(cmd) for cmd in scorers_cli.commands.values()],
    ]
    mcp = FastMCP(
        name="Mlflow MCP",
        tools=tools,
    )

    register_prompts(mcp)
    return mcp


def run_server() -> None:
    mcp = create_mcp()
    mcp.run(show_banner=False)


if __name__ == "__main__":
    run_server()
