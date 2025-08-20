import contextlib
import io
from typing import Any, Callable, Literal

import click
from click import Parameter, ParamType
from click.types import BOOL, FLOAT, INT, STRING, UUID
from fastmcp import FastMCP
from fastmcp.tools import FunctionTool


@click.group()
def cli():
    """
    Test CLI
    """


@cli.command()
@click.argument("a")
@click.option(
    "--b",
    default=1,
    type=int,
    help="An optional integer parameter",
)
@click.option(
    "--c",
    default="a",
    type=click.Choice(["a", "b", "c"]),
    help="An optional enum parameter",
)
def test(
    a: str,
    b: int = 1,
    c: Literal["a", "b", "c"] = "a",
):
    """
    Test command
    """
    click.echo(f"Test command called with a={a!r} and b={b!r} and c={c!r}")


def param_type_to_json_schema_type(pt: ParamType) -> str:
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


def get_input_schema(params: list[Parameter]) -> dict[str, Any]:
    """
    Converts click params to JSON schema
    """
    res: dict[str, Any] = {}

    for p in params:
        schema = {
            "type": param_type_to_json_schema_type(p.type),
            "default": p.default,
            "required": p.required,
            "description": p.help if hasattr(p, "help") else None,
        }
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
    return FunctionTool(
        fn=wrapper(cmd),
        name=cmd.name,
        description=cmd.help,
        parameters={
            "name": cmd.name,
            "description": cmd.help,
            "inputSchema": get_input_schema(cmd.params),
        },
    )


def create_mcp() -> FastMCP:
    return FastMCP(
        name=cli.name,
        instructions=cli.help,
        tools=[cmd_to_function_tool(cmd) for cmd in cli.commands.values()],
    )


if __name__ == "__main__":
    mcp = create_mcp()
    mcp.run(show_banner=False)
