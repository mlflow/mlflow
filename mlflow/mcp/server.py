import contextlib
import io
import re
from pathlib import Path
from typing import Any, Callable

import click
import yaml
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


def cmd_to_function_tool(cmd: click.Command) -> FunctionTool:
    """
    Converts a Click command to a FunctionTool.
    """
    return FunctionTool(
        fn=fn_wrapper(cmd),
        name=cmd.callback.__name__,
        description=(cmd.help or "").strip(),
        parameters=get_input_schema(cmd.params),
    )


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}, content

    match = re.match(r"^---\n(.*?)\n---\n(.*)", content, re.DOTALL)
    if not match:
        return {}, content

    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}, content

    return metadata, match.group(2)


def _list_ai_commands() -> list[dict[str, Any]]:
    """List AI command files from mlflow/ai_commands/"""
    # Get mlflow/ai_commands directory
    ai_commands_dir = Path(__file__).parent.parent / "ai_commands"
    commands = []

    if not ai_commands_dir.exists():
        return commands

    for md_file in ai_commands_dir.glob("**/*.md"):
        try:
            content = md_file.read_text()
            metadata, _ = _parse_frontmatter(content)

            # Build command key from path
            relative_path = md_file.relative_to(ai_commands_dir)
            command_key = str(relative_path.with_suffix("")).replace("/", "_")

            commands.append(
                {
                    "key": command_key,
                    "original_key": str(relative_path.with_suffix("")),
                    "description": metadata.get("description", "No description"),
                }
            )
        except Exception:
            continue

    return commands


def _get_ai_command(original_key: str) -> str:
    """Get content of specific AI command."""
    ai_commands_dir = Path(__file__).parent.parent / "ai_commands"
    key_parts = original_key.split("/")
    command_path = ai_commands_dir.joinpath(*key_parts).with_suffix(".md")

    if not command_path.exists():
        return f"Command '{original_key}' not found"

    content = command_path.read_text()
    _, body = _parse_frontmatter(content)
    return body


def create_mcp() -> FastMCP:
    mcp = FastMCP(
        name="Mlflow MCP",
        tools=[cmd_to_function_tool(cmd) for cmd in traces_cli.commands.values()],
    )

    # Register AI commands as prompts
    for command in _list_ai_commands():
        # Create a closure to capture the command key
        def make_prompt(cmd_original_key: str):
            @mcp.prompt(name=command["key"], description=command["description"])
            def ai_command_prompt() -> str:
                """Execute an MLflow AI command prompt."""
                return _get_ai_command(cmd_original_key)

            return ai_command_prompt

        # Register the prompt
        make_prompt(command["original_key"])

    return mcp


def run_server() -> None:
    mcp = create_mcp()
    mcp.run(show_banner=False)


if __name__ == "__main__":
    run_server()
