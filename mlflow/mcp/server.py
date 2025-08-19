import contextlib
import io
from typing import Any, Callable

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
@click.option("--b", default=1, type=int, help="An optional integer parameter")
def test(a: str, b: int = 1):
    """
    Test command
    """
    click.echo(f"Test command called with a={a} and b={b}")


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


mcp = FastMCP("Demo 🚀")


def wrapper(command: click.Command) -> Callable[..., str]:
    def wrapper(**kwargs: Any) -> str:
        string_io = io.StringIO()
        with contextlib.redirect_stdout(string_io), contextlib.redirect_stderr(string_io):
            command.callback(**kwargs)
        return string_io.getvalue()

    return wrapper


mcp.add_tool(
    FunctionTool(
        fn=wrapper(test),
        name=test.name,
        description=test.help,
        parameters={
            "name": test.name,
            "description": test.help,
            "inputSchema": get_input_schema(test.params),
        },
    )
)

if __name__ == "__main__":
    mcp.run()
